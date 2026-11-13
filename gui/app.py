"""Trance LLM: chat, comparison and training around the native CUDA engine."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
from pathlib import Path
import time
import uuid
import tempfile

from PySide6.QtCore import Qt, QTimer, QLockFile
from PySide6.QtGui import QFont, QDesktopServices, QIcon
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame,
    QLabel, QPushButton, QComboBox, QListWidgetItem, QStackedWidget, QDoubleSpinBox,
    QSpinBox, QScrollArea, QFileDialog, QPlainTextEdit, QProgressBar, QSplitter, QMessageBox,
    QMenu, QInputDialog, QLineEdit, QListWidget, QGridLayout)

from .backends.router import BackendRouter, backend_kind
from .gguf import discover_gguf, read_gguf_info
from .runtimes import RuntimeManager
from . import APP_NAME
from .storage import ROOT, STATE, atomic_json, read_json, new_chat, completed_context, discover_models, model_info, validate_config
from .training import EventTail, latest_run, launch_training, stop_training, run_alive, recent_runs
from .widgets import Composer, Transcript, LossChart, ConversationList, dot_icon, STYLE
from . import theme
from .config_builder import ConfigBuilder, parameter_count
from .settings import SettingsPage
from .documentation import Documentation
from .theme import COLORS as C


def label(text, name=None):
    widget = QLabel(text)
    if name: widget.setObjectName(name)
    return widget


GREETINGS = ("Welcome", "What\u2019s up next?", "What are we making today?", "Ready when you are", "What\u2019s on your mind?",
             "Let\u2019s get started", "What shall we explore?", "Good to see you")
SUGGESTIONS = ("Write a fantasy story", "Explain how black holes work", "Make me an email template", "Plan a trip to Rome",
               "Teach me a card trick", "Brainstorm names for a caf\u00e9", "Give me a beginner workout", "Explain recursion simply",
               "Write a poem about the sea", "Summarize the plot of Hamlet", "Help me learn some Spanish", "Suggest a quick dinner",
               "Draft a polite apology", "Tell me a fun space fact", "Describe a futuristic city", "Explain how vaccines work")


def human_size(size):
    if size >= 1e9: return f"{size/1e9:.1f} GB"
    if size >= 1e7: return f"{size/1e6:.0f} MB"
    if size >= 1e6: return f"{size/1e6:.1f} MB"
    return f"{max(1, round(size/1e3))} KB"


def model_label(info):
    if info.get("kind") == "gguf":
        quant = f" {info['quantization']}" if info.get("quantization") else ""
        return f"{info['title']}  \u00b7  GGUF{quant}  \u00b7  {human_size(info['size_bytes'])}"
    return f"{friendly(info['path'])}  \u00b7  {info['parameters']/1e6:.1f}M"


def button(text, callback, primary=False):
    widget = QPushButton(text)
    if primary: widget.setObjectName("primary")
    widget.clicked.connect(callback)
    return widget


def friendly(path):
    name = Path(path).stem
    for suffix in ("_21m", "_30m"):
        if name.endswith(suffix): name = name[:-len(suffix)]
    return name.replace("trance", "Trance ", 1)


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600); mins, secs = divmod(rem, 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m {secs:02d}s"


class MainWindow(QMainWindow):
    def __init__(self, *, state_dir=None, binary=None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(str(Path(__file__).parent/"assets/trance.svg")))
        self.resize(1300, 860)
        self.setMinimumSize(1000, 690)
        self.statusBar().hide()
        self.statusBar().messageChanged.connect(lambda message: self.statusBar().setVisible(bool(message) and message != "Ready"))
        self.state_dir = Path(state_dir or STATE)
        self.chats = self._read_state("chats.json", [])
        self.preferences = self._read_state("settings.json", {})
        if not isinstance(self.chats, list): self.chats = []
        # Older versions saved empty drafts. Only conversations with messages
        # belong in history; preserve every conversation that has content.
        self.chats = [chat for chat in self.chats if isinstance(chat,dict) and chat.get("messages")]
        if not isinstance(self.preferences, dict): self.preferences = {}
        self._theme_name = self.preferences.get("theme") if self.preferences.get("theme") in theme.PALETTES else theme.DEFAULT_PALETTE
        self._apply_stylesheet(self._theme_name)
        for chat in self.chats:
            for message in chat.get("messages", []):
                if message.get("status") == "generating":
                    message.update(status="cancelled", note="Interrupted when the application closed")
        self.current_chat = None
        self.operation = None
        self.comparison = None
        self.compare_index = 0
        self.compare_cancelled = False
        self.training_dir = None
        self.tail = None
        self.training_active = False
        self.training_base_step = 0
        self.section_last = {0: 0, 1: 2, 2: 1}  # last page opened in each sidebar section
        self._busy = False
        self._engine_error = False
        self._gpu_ready = False
        self._closing = False
        self.models = []
        self.runtimes = RuntimeManager(self.state_dir, lambda: self.preferences.get("llama_server_path", ""), self._set_llama_path, self)
        self.engine = BackendRouter(self, binary=binary, resolver=self.runtimes.resolve)
        self.engine.delta.connect(self._delta)
        self.engine.started.connect(self._generation_started)
        self.engine.finished.connect(self._finished)
        self.engine.failed.connect(self._failed)
        self.engine.state_changed.connect(self.statusBar().showMessage)
        self.engine.ready.connect(self._engine_ready)
        self.engine.loaded.connect(self._model_loaded)
        self.runtimes.changed.connect(self._update_welcome)
        self._build()
        self.refresh_models()
        self._refresh_chats()
        if self.chats:
            self._select_chat(0)
        else:
            self._new_chat()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_training)
        self.timer.start(1000)
        previous = latest_run()
        if previous:
            self._attach_training(previous)
        self._show_page(0)
        self._update_status()
        self.statusBar().clearMessage()

    def _read_state(self, filename, default):
        path = self.state_dir / filename
        try:
            return read_json(path, default)
        except (ValueError, OSError):
            if path.exists():
                path.rename(path.with_name(path.stem + ".unreadable-" + str(int(time.time())) + ".json"))
            return default

    def _build(self):
        central = QWidget(); layout = QHBoxLayout(central); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        self.setCentralWidget(central)
        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(230)
        left = QVBoxLayout(sidebar); left.setContentsMargins(18,28,18,20); left.setSpacing(10)
        brand = QHBoxLayout(); brand.setSpacing(8)
        brand.addWidget(label("TRANCE", "brand")); brand.addWidget(label("LLM", "brandAccent")); brand.addStretch()
        left.addLayout(brand); left.addSpacing(12)
        self.nav = {}
        # One click reaches any section; the list below the rule follows the section.
        bar = QFrame(); bar.setObjectName("segBar")
        bar_layout = QHBoxLayout(bar); bar_layout.setContentsMargins(3,3,3,3); bar_layout.setSpacing(2)
        self.sections = []
        for i, name in enumerate(("Chat","Train","Tools")):
            seg = button(name, lambda checked=False, section=i: self._open_section(section))
            seg.setObjectName("seg"); seg.setCheckable(True); seg.setCursor(Qt.CursorShape.PointingHandCursor)
            bar_layout.addWidget(seg,1); self.sections.append(seg)
        left.addWidget(bar)
        rule = QFrame(); rule.setObjectName("sidebarRule"); rule.setFixedHeight(1); left.addWidget(rule)
        self.section_stack = QStackedWidget(); self.section_stack.setObjectName("sectionStack"); left.addWidget(self.section_stack,1)

        def panel():
            widget = QWidget(); widget.setObjectName("sidebarPanel")
            box = QVBoxLayout(widget); box.setContentsMargins(0,4,0,0); box.setSpacing(8)
            self.section_stack.addWidget(widget)
            return box

        def nav_button(box, page, name):
            nav = button(name, lambda checked=False, page=page: self._show_page(page)); nav.setObjectName("nav"); nav.setCheckable(True)
            box.addWidget(nav); self.nav[page] = nav

        chat_box = panel()
        self.new_chat_button = button("+  New chat", self._new_chat, primary=True); self.new_chat_button.setCheckable(True)
        self.temporary_chat_button = button("Temporary chat", lambda: self._new_chat(temporary=True))
        self.temporary_chat_button.setObjectName("outline"); self.temporary_chat_button.setCheckable(True)
        self.temporary_chat_button.setToolTip("A private chat that is never saved to history")
        self.chat_search = QLineEdit(); self.chat_search.setObjectName("chatSearch"); self.chat_search.setPlaceholderText("Search chats")
        self.chat_search.setClearButtonEnabled(True); self.chat_search.textChanged.connect(lambda _text: self._apply_chat_filter())
        self.chat_list = ConversationList(); self.chat_list.currentRowChanged.connect(self._select_chat)
        self.chat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self._chat_context_menu)
        self.chat_empty = label("No saved chats yet", "muted"); self.chat_empty.setWordWrap(True); self.chat_empty.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        chat_box.addWidget(self.new_chat_button); chat_box.addWidget(self.temporary_chat_button); chat_box.addSpacing(4)
        chat_box.addWidget(self.chat_search); chat_box.addWidget(self.chat_list,1); chat_box.addWidget(self.chat_empty,1,Qt.AlignmentFlag.AlignTop)

        train_box = panel()
        nav_button(train_box,2,"Train"); nav_button(train_box,3,"Configs")
        train_box.addSpacing(6); train_box.addWidget(label("RECENT RUNS","eyebrow"))
        self.run_list = QListWidget(); self.run_list.setObjectName("runList")
        self.run_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.run_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.run_list.itemClicked.connect(self._open_run)
        self.run_empty = label("No training runs yet", "muted"); self.run_empty.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        train_box.addWidget(self.run_list,1); train_box.addWidget(self.run_empty,1,Qt.AlignmentFlag.AlignTop)

        tools_box = panel()
        nav_button(tools_box,1,"Compare"); nav_button(tools_box,4,"Documentation"); nav_button(tools_box,5,"Settings"); tools_box.addStretch(1)

        card = QFrame(); card.setObjectName("statusCard")
        card_layout = QHBoxLayout(card); card_layout.setContentsMargins(12,10,12,10); card_layout.setSpacing(10)
        self.status_dot = label("\u25cf", "statusDot"); self.status_dot.setProperty("state", "idle")
        card_layout.addWidget(self.status_dot,0,Qt.AlignmentFlag.AlignTop)
        status_text = QVBoxLayout(); status_text.setSpacing(1)
        self.status_title = label("Engine idle", "statusTitle")
        self.gpu_status = label("CUDA engine", "muted"); self.gpu_status.setWordWrap(True)
        status_text.addWidget(self.status_title); status_text.addWidget(self.gpu_status); card_layout.addLayout(status_text,1)
        left.addWidget(card)
        layout.addWidget(sidebar)
        self.pages = QStackedWidget(); layout.addWidget(self.pages, 1)
        self._build_chat(); self._build_compare(); self._build_training()
        self.config_builder=ConfigBuilder(root=ROOT);self.pages.addWidget(self.config_builder)
        self.config_builder.saved.connect(lambda path:self._use_saved_config(path,show=False))
        self.config_builder.open_training.connect(self._use_saved_config)
        self.documentation=Documentation();self.pages.addWidget(self.documentation)
        self.settings_page = SettingsPage(self._theme_name, runtimes=self.runtimes); self.pages.addWidget(self.settings_page)
        self.settings_page.theme_selected.connect(self._set_theme)
        self.settings = QFrame(); self.settings.setObjectName("settings"); self.settings.setFixedWidth(264)
        outer = QVBoxLayout(self.settings); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setObjectName("generationScroll"); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); outer.addWidget(scroll)
        content = QWidget(); content.setObjectName("generationContent"); scroll.setWidget(content)
        settings = QVBoxLayout(content); settings.setContentsMargins(20,28,20,20); settings.setSpacing(20)
        settings.addWidget(label("GENERATION", "eyebrow"))
        self.temperature = QDoubleSpinBox(); self.temperature.setRange(.05,2); self.temperature.setSingleStep(.05); self.temperature.setValue(self.preferences.get("temperature",.8))
        self.top_p = QDoubleSpinBox(); self.top_p.setRange(.01,1); self.top_p.setSingleStep(.05); self.top_p.setValue(self.preferences.get("top_p",.95))
        self.top_k = QSpinBox(); self.top_k.setRange(0,1024); self.top_k.setValue(self.preferences.get("top_k",40))
        self.max_tokens = QSpinBox(); self.max_tokens.setRange(1,1023); self.max_tokens.setValue(self.preferences.get("max_new_tokens",160))
        self.seed = QSpinBox(); self.seed.setRange(0,2**31-1); self.seed.setValue(self.preferences.get("seed",4004))
        parameters = (
            ("Temperature",self.temperature,"Controls randomness. Lower values give more predictable replies; higher values add variety."),
            ("Top P",self.top_p,"Keeps likely next tokens up to this total probability. Lower values narrow the choices."),
            ("Top K",self.top_k,"Limits each choice to the K most likely tokens. Set 0 to allow the whole vocabulary."),
            ("Max tokens",self.max_tokens,"Caps the reply length. A token can be a word, part of a word or punctuation."),
            ("Seed",self.seed,"Use the same seed with an identical model, prompt and settings to repeat a response."),
        )
        self.parameter_descriptions = {}
        for title, widget, description in parameters:
            group = QVBoxLayout(); group.setSpacing(8)
            row = QHBoxLayout(); row.addWidget(label(title,"parameter")); row.addStretch()
            widget.setFixedWidth(116); row.addWidget(widget); group.addLayout(row)
            note = label(description,"parameterDescription"); note.setWordWrap(True)
            widget.setToolTip(description); group.addWidget(note); settings.addLayout(group)
            self.parameter_descriptions[title] = note
        settings.addStretch()
        layout.addWidget(self.settings)

    def _page(self, title):
        page = QWidget(); box = QVBoxLayout(page); box.setContentsMargins(30,28,30,24); box.setSpacing(16)
        box.addWidget(label(title,"title")); self.pages.addWidget(page)
        return page, box

    def _build_chat(self):
        page = QWidget(); box = QVBoxLayout(page); box.setContentsMargins(30,28,30,24); box.setSpacing(16)
        self.pages.addWidget(page)
        header = QHBoxLayout(); self.chat_title = label("New Chat","title"); header.addWidget(self.chat_title); header.addStretch()
        self.chat_export_button = button("Export chat…",self._export_chat); header.addWidget(self.chat_export_button); box.addLayout(header)
        self.model_controls = QFrame(); self.model_controls.setObjectName("modelControls")
        models = QHBoxLayout(self.model_controls); models.setContentsMargins(8,8,8,8); models.setSpacing(6)
        self.model_combo = QComboBox(); self.model_combo.setMinimumWidth(210); self.model_combo.setMaximumWidth(300)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(12); self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.browse_model_button = button("Browse…",self._browse_model)
        self.refresh_model_button = button("↻",self.refresh_models); self.refresh_model_button.setFixedWidth(34)
        self.refresh_model_button.setToolTip("Refresh the model list"); self.refresh_model_button.setAccessibleName("Refresh models")
        models.addWidget(self.model_combo,1); models.addWidget(self.browse_model_button); models.addWidget(self.refresh_model_button)
        floating = QHBoxLayout(); floating.addStretch(); floating.addWidget(self.model_controls); floating.addStretch(); box.addLayout(floating)
        self.chat_content = QStackedWidget()
        self.transcript = Transcript(); self.chat_content.addWidget(self.transcript)
        temporary_empty = QWidget(); center = QVBoxLayout(temporary_empty); center.setContentsMargins(20,20,20,20)
        self.temporary_notice = label("Chat Not Saved to History. Cleared when you switch chats or close the app.","temporaryNotice")
        self.temporary_notice.setWordWrap(True); self.temporary_notice.setMinimumWidth(300); self.temporary_notice.setMaximumWidth(440)
        self.temporary_notice.setMinimumHeight(72)
        self.temporary_notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addStretch(); center.addWidget(self.temporary_notice,0,Qt.AlignmentFlag.AlignHCenter); center.addStretch()
        self.chat_content.addWidget(temporary_empty)
        welcome = QWidget(); outer = QVBoxLayout(welcome); outer.setContentsMargins(20,20,20,20); outer.setSpacing(10)
        self.welcome_title = label("Welcome","welcomeTitle"); self.welcome_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.welcome_hint = label("Choose a model above to get started.","welcomeHint"); self.welcome_hint.setAlignment(Qt.AlignmentFlag.AlignCenter); self.welcome_hint.setWordWrap(True)
        outer.addStretch(5); outer.addWidget(self.welcome_title); outer.addWidget(self.welcome_hint); outer.addStretch(6)
        self._greeting, self._welcome_active, self._shown_suggestions = "", False, 0
        self._suggestions, self._suggestion_offset = list(SUGGESTIONS), 0
        self.chat_content.addWidget(welcome); box.addWidget(self.chat_content,1)
        self.composer_panel = QFrame(); self.composer_panel.setObjectName("composerPanel")
        panel = QVBoxLayout(self.composer_panel); panel.setContentsMargins(12,10,12,10); panel.setSpacing(6)
        self.composer = Composer(auto_size=True); self.composer.setObjectName("messageInput"); self.composer.setPlaceholderText("Message Trance…")
        self.composer.submitted.connect(self._send_chat); panel.addWidget(self.composer)
        self.response_controls = QWidget(); self.response_controls.setObjectName("responseControls")
        actions = QHBoxLayout(self.response_controls); actions.setContentsMargins(0,0,0,0); actions.setSpacing(6)
        self.stop_button = button("Stop response",self._stop); self.stop_button.setEnabled(False); actions.addWidget(self.stop_button)
        self.send_button = button("Send message",self._send_chat,True); actions.addWidget(self.send_button)
        for widget in (self.browse_model_button,self.refresh_model_button,self.stop_button,self.send_button): widget.setProperty("compact",True)
        footer = QHBoxLayout()
        self.chat_hint = label("Enter to send · Shift+Enter for a new line","composerHint"); self.chat_hint.setWordWrap(True)
        footer.addWidget(self.chat_hint,1); footer.addWidget(self.response_controls); panel.addLayout(footer)
        self.suggestion_row = QWidget(); self.suggestion_row.setObjectName("sidebarPanel")
        row = QHBoxLayout(self.suggestion_row); row.setContentsMargins(0,0,0,0); row.setSpacing(8)
        box.addWidget(self.suggestion_row); box.addWidget(self.composer_panel)
        self.suggestion_timer = QTimer(self); self.suggestion_timer.setInterval(8000); self.suggestion_timer.timeout.connect(self._rotate_suggestions); self.suggestion_timer.start()

    def _build_compare(self):
        _, box = self._page("Compare models")
        row = QHBoxLayout(); self.compare_a = QComboBox(); self.compare_b = QComboBox()
        row.addWidget(label("Model A")); row.addWidget(self.compare_a,1); row.addWidget(label("Model B")); row.addWidget(self.compare_b,1); box.addLayout(row)
        self.compare_hint = label("Trance 1.2 will appear here once it has been trained.","muted"); self.compare_hint.setWordWrap(True); box.addWidget(self.compare_hint)
        self.compare_prompt = Composer(); self.compare_prompt.setMaximumHeight(105); self.compare_prompt.setPlaceholderText("Enter a prompt to compare both models…"); self.compare_prompt.submitted.connect(self._start_compare); box.addWidget(self.compare_prompt)
        split = QSplitter(); self.compare_views = []
        for name in ("A", "B"):
            container = QWidget(); col = QVBoxLayout(container); col.setContentsMargins(0,0,0,0)
            col.addWidget(label(f"RESPONSE {name}","eyebrow")); view = Transcript(); col.addWidget(view,1); self.compare_views.append(view); split.addWidget(container)
            view.render([],"Awaiting a prompt","Select a model above to begin.")
        box.addWidget(split,1)
        row = QHBoxLayout(); self.export_compare = button("Export comparison…",self._export_compare); self.export_compare.setEnabled(False); row.addWidget(self.export_compare); row.addStretch()
        self.compare_stop = button("Stop",self._stop); self.compare_stop.setEnabled(False); row.addWidget(self.compare_stop)
        self.compare_run = button("Compare responses",self._start_compare,True); row.addWidget(self.compare_run); box.addLayout(row)

    def _build_training(self):
        _, box = self._page("Train Trance")
        row = QHBoxLayout(); self.config_combo = QComboBox()
        for path in sorted((ROOT/"configs").glob("*.json")):
            self.config_combo.addItem(path.name,str(path))
        index = self.config_combo.findData(str(ROOT/"configs/trance1.2_30m.json"))
        if index >= 0: self.config_combo.setCurrentIndex(index)
        row.addWidget(self.config_combo,1); row.addWidget(button("Browse config…",self._browse_config)); row.addWidget(button("Reload",self._load_config)); box.addLayout(row)
        self.config_combo.currentIndexChanged.connect(self._load_config)
        split = QSplitter(); editor_panel = QWidget(); left = QVBoxLayout(editor_panel); left.setContentsMargins(0,0,8,0)
        left.addWidget(label("RUN SUMMARY", "eyebrow")); self._left_layout = left
        self.summary_tiles = {}
        grid = QGridLayout(); grid.setSpacing(8)
        for n, (key, caption) in enumerate((("params","Parameters"),("weights","FP32 weights"),("shape","Layers \u00d7 width"),("context","Token context"),("steps","Steps"),("lr","Learning rate"))):
            tile = QFrame(); tile.setObjectName("statTile"); tile_box = QVBoxLayout(tile); tile_box.setContentsMargins(12,10,12,10); tile_box.setSpacing(2)
            value = label("\u2013","statValue"); name = label(caption,"statLabel"); name.setWordWrap(True)
            tile_box.addWidget(value); tile_box.addWidget(name); grid.addWidget(tile, n // 2, n % 2); self.summary_tiles[key] = (value, name)
        left.addLayout(grid)
        self.config_hint = label("", "configHint"); self.config_hint.setWordWrap(True); left.addWidget(self.config_hint)
        self.config_output = label("", "muted"); self.config_output.setWordWrap(True); left.addWidget(self.config_output)
        self.json_toggle = button("Edit JSON \u25b8", self._toggle_json); self.json_toggle.setObjectName("link"); self.json_toggle.setCheckable(True)
        self.json_toggle.setCursor(Qt.CursorShape.PointingHandCursor); left.addWidget(self.json_toggle)
        self.config_editor = QPlainTextEdit(); self.config_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap); self.config_editor.setFont(QFont("DejaVu Sans Mono",10)); self.config_editor.setVisible(False); left.addWidget(self.config_editor,1)
        self.json_note = label("Edits are saved as a run snapshot when training starts.","muted"); self.json_note.setVisible(False); left.addWidget(self.json_note)
        self._left_spacer = left.count(); left.addStretch(1); split.addWidget(editor_panel)
        metrics = QWidget(); right = QVBoxLayout(metrics); right.setContentsMargins(12,0,0,0)
        right.addWidget(label("TRAINING / VALIDATION LOSS", "eyebrow")); self.chart_legend = label(self._legend_html(), "muted")
        self.chart_legend.setTextFormat(Qt.TextFormat.RichText); right.addWidget(self.chart_legend)
        self.chart = LossChart(); right.addWidget(self.chart,1)
        self.progress = QProgressBar(); self.progress.setRange(0,1000); self.progress.setValue(0); right.addWidget(self.progress)
        self.train_metrics = label("No training run yet", "muted"); self.train_metrics.setWordWrap(True); right.addWidget(self.train_metrics)
        self.train_log = QPlainTextEdit(); self.train_log.setReadOnly(True); self.train_log.setMaximumBlockCount(500); self.train_log.setMaximumHeight(160); self.train_log.setPlaceholderText("Progress and checkpoint locations appear here."); right.addWidget(self.train_log)
        split.addWidget(metrics); split.setSizes([390,600]); box.addWidget(split,1)
        self.train_message = label("Training only starts when you click Start training.","muted"); self.train_message.setWordWrap(True); box.addWidget(self.train_message)
        row = QHBoxLayout(); row.addWidget(button("Open run folder", self._open_run_folder)); self.resume_button = button("Resume checkpoint",self._resume_training); self.resume_button.setEnabled(False); row.addWidget(self.resume_button); row.addStretch()
        self.train_stop = button("Stop && save checkpoint", self._stop_training); self.train_stop.setEnabled(False); row.addWidget(self.train_stop)
        self.train_start = button("Start training",self._start_training,True); row.addWidget(self.train_start); box.addLayout(row)
        self.config_editor.textChanged.connect(self._validate_editor)
        self._load_config()

    SECTION_OF_PAGE = {0: 0, 2: 1, 3: 1, 1: 2, 4: 2, 5: 2}  # page index -> sidebar section

    def _open_section(self, section):
        self._show_page(self.section_last[section])

    def _show_page(self,index):
        self.pages.setCurrentIndex(index)
        self.settings.setVisible(index in (0,1))
        section = self.SECTION_OF_PAGE[index]; self.section_last[section] = index
        for i, seg in enumerate(self.sections): seg.setChecked(i==section)
        self.section_stack.setCurrentIndex(section)
        if section == 1: self._refresh_runs()
        for i, nav in self.nav.items(): nav.setChecked(i==index)
        temporary = bool(self.current_chat and self.current_chat.get("temporary"))
        new = bool(self.current_chat and not self.current_chat.get("messages"))
        self.temporary_chat_button.setChecked(index==0 and temporary)
        self.new_chat_button.setChecked(index==0 and not temporary and new)
        self.chat_title.setText("Temporary Chat" if temporary else "New Chat" if new else "Chat")

    def sampling(self):
        return {"temperature":self.temperature.value(),"top_p":self.top_p.value(),"top_k":self.top_k.value(),
                "max_new_tokens":self.max_tokens.value(),"seed":self.seed.value()}

    def refresh_models(self):
        existing = {m["path"]:m for m in self.models}
        existing.update({m["path"]:m for m in discover_models()})
        existing.update({m["path"]:m for m in discover_gguf()})
        for saved in self.preferences.get("gguf_paths", []):
            try: info = read_gguf_info(Path(saved)); existing[info["path"]] = info
            except (OSError, ValueError): pass
        self.models = [m for m in existing.values() if Path(m["path"]).is_file()]
        for combo in (self.model_combo,self.compare_a,self.compare_b):
            previous = combo.currentData(); combo.blockSignals(True); combo.clear(); combo.addItem("Choose a model…","")
            for info in self.models:
                combo.addItem(model_label(info),info["path"])
            index = combo.findData(previous)
            if index < 1 and combo is not self.compare_b and self.models: index = 1
            combo.setCurrentIndex(max(0,index)); combo.blockSignals(False)
            combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0) + 40)   # the closed box is narrow; the list shows full names
        self._model_changed()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self,"Choose a model",str(ROOT/"models"),"Models (*.bin *.gguf);;Trance models (*.bin);;GGUF models (*.gguf)")
        if not path: return
        try:
            if backend_kind(path) == "gguf":
                info = read_gguf_info(Path(path)); saved = [p for p in self.preferences.get("gguf_paths", []) if p != info["path"]]
                self.preferences["gguf_paths"] = saved + [info["path"]]; self._save_preferences()
            else: info = model_info(Path(path))
            self.models.append(info); self.refresh_models()
            self.model_combo.setCurrentIndex(self.model_combo.findData(info["path"]))
        except (OSError,ValueError) as error: self.statusBar().showMessage(str(error))

    def _model_loaded(self, info):
        name = friendly(info["model"]) if backend_kind(info["model"]) == "trance" else Path(info["model"]).stem
        size = f" \u00b7 {info['parameters']/1e6:.2f}M parameters" if info.get("parameters") else ""
        self.statusBar().showMessage(f"Loaded {name}{size}")

    def _set_llama_path(self, value):
        self.preferences["llama_server_path"] = value; self._save_preferences()

    def _model_changed(self):
        path = self.model_combo.currentData()
        self._update_welcome()
        info = next((m for m in self.models if m["path"]==path),None)
        if info and info.get("kind") == "gguf":
            quant = f" \u00b7 {info['quantization']}" if info.get("quantization") else ""
            self.model_combo.setToolTip(f"GGUF ({info['architecture']}) \u00b7 {info['context_length']:,} token context{quant} \u00b7 {human_size(info['size_bytes'])}")
            if hasattr(self,"max_tokens") and info["context_length"]: self.max_tokens.setMaximum(info["context_length"]-1)
        elif info:
            self.model_combo.setToolTip(f"{info['parameters']/1e6:.2f}M parameters · {info['context_length']:,} token context · Step {info['step']:,}")
            if hasattr(self,"max_tokens"): self.max_tokens.setMaximum(info["context_length"]-1)
        else: self.model_combo.setToolTip("Select a trained model or browse to a checkpoint with its tokenizer.")

    def _refresh_chats(self):
        active = self.current_chat["id"] if self.current_chat else None
        scroll = self.chat_list.verticalScrollBar().value()
        self.chat_list.blockSignals(True); self.chat_list.clear()
        for chat in self.chats:
            item = QListWidgetItem(chat.get("title","Conversation")); item.setToolTip(chat.get("title","Conversation"))
            item.setData(Qt.ItemDataRole.UserRole,chat["id"]); self.chat_list.addItem(item)
        index = next((i for i,c in enumerate(self.chats) if c["id"]==active),-1)
        self.chat_list.setCurrentRow(index); self.chat_list.blockSignals(False)
        self.chat_list.verticalScrollBar().setValue(scroll)
        self._apply_chat_filter()

    def _apply_chat_filter(self):
        query = self.chat_search.text().strip().casefold()
        shown = 0
        for row, chat in enumerate(self.chats):  # list rows stay aligned with self.chats
            match = not query or query in chat.get("title","").casefold() or any(
                query in str(m.get("content","")).casefold() for m in chat.get("messages",[]))
            self.chat_list.item(row).setHidden(not match); shown += match
        self.chat_empty.setText("No saved chats yet" if not self.chats else "No chats match your search")
        self.chat_list.setVisible(shown > 0); self.chat_empty.setVisible(shown == 0)

    def _refresh_runs(self):
        self.run_list.blockSignals(True); self.run_list.clear()
        colors = {"running": C["accent"], "complete": C["accent_2"], "stopped": C["text_muted"], "failed": C["danger"], "unknown": C["text_disabled"]}
        words = {"running": "running", "complete": "done", "stopped": "stopped", "failed": "failed", "unknown": "unknown"}
        runs = recent_runs()
        for run in runs:
            name = friendly(run["model"]) if run["model"] else run["dir"].name
            item = QListWidgetItem(dot_icon(colors[run["state"]]), f"{name}  \u00b7  {words[run['state']]}")
            item.setData(Qt.ItemDataRole.UserRole, str(run["dir"])); item.setToolTip(str(run["dir"])); self.run_list.addItem(item)
            if self.training_dir and run["dir"] == self.training_dir: self.run_list.setCurrentItem(item)
        self.run_list.blockSignals(False)
        self.run_list.setVisible(bool(runs)); self.run_empty.setVisible(not runs)

    def _open_run(self, item):
        directory = Path(item.data(Qt.ItemDataRole.UserRole))
        if directory != self.training_dir:
            if self.training_active:
                self.train_message.setText("A training run is active. Stop it before opening an earlier run.")
                self._show_page(2); return
            self._attach_training(directory)
        self._show_page(2)

    @staticmethod
    def _apply_stylesheet(name):
        sheet = theme.apply_palette(name); app = QApplication.instance()
        if app.styleSheet() != sheet: app.setStyleSheet(sheet)   # re-polishing every widget is slow; skip no-ops

    def _set_theme(self, name):
        if name not in theme.PALETTES or name == self._theme_name: return
        self._theme_name = name
        self._apply_stylesheet(name)
        self.preferences["theme"] = name; self._save_preferences()
        self._restyle()

    def _restyle(self):
        """Colors baked into rendered HTML, icons and painted widgets need a nudge after a theme change."""
        for view in (self.transcript, *self.compare_views): view.restyle()
        self.chart_legend.setText(self._legend_html()); self.chart.update()
        self._refresh_runs(); self.documentation.restyle(); self._update_status()

    @staticmethod
    def _legend_html():
        return (f'<span style="color:{C["accent_2"]}">\u25cf</span> Training &nbsp;&nbsp;&nbsp; '
                f'<span style="color:{C["accent"]}">\u25cf</span> Validation')

    def _engine_ready(self, gpu):
        self._gpu_ready = True; self.gpu_status.setText(gpu.split(" (compute")[0]); self._update_status()

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget); widget.style().polish(widget)

    def _update_status(self):
        if self.training_active: state, title = "busy", f"Training \u00b7 {self.progress.value()//10}%"
        elif self._busy: state, title = "busy", "Generating\u2026"
        elif self._engine_error: state, title = "error", "Last request failed"
        elif self._gpu_ready: state, title = "ok", "Ready"
        else: state, title = "idle", "Engine idle"
        self.status_title.setText(title)
        if self.status_dot.property("state") != state:
            self.status_dot.setProperty("state", state); self._repolish(self.status_dot)
        seg = self.sections[1]; running = bool(self.training_active)
        if bool(seg.property("running")) != running or seg.text() != ("Train \u25cf" if running else "Train"):
            seg.setText("Train \u25cf" if running else "Train"); seg.setProperty("running", running); self._repolish(seg)

    def _new_chat(self, *, temporary=False):
        if self.engine.busy:
            self.statusBar().showMessage("Stop the current response before starting a new conversation"); return
        chat = new_chat(self.model_combo.currentData() or "")
        if temporary: chat.update(temporary=True,title="Temporary chat")
        self.current_chat=chat
        self._display_chat()

    def _display_chat(self):
        self._refresh_chats(); self.composer.clear()
        self.transcript.render(self.current_chat.get("messages",[]))
        self._show_chat_content()
        self.chat_hint.setText("Enter to send · Shift+Enter for a new line")
        self._show_page(0)

    def _show_chat_content(self):
        chat = self.current_chat
        if chat and not chat.get("messages"): index = 1 if chat.get("temporary") else 2   # notice / welcome
        else: index = 0
        self.chat_content.setCurrentIndex(index)
        entering = index == 2 and not self._welcome_active
        self._welcome_active = index == 2
        if entering: self._enter_welcome()
        self.suggestion_row.setVisible(self._welcome_active)
        if entering: self._fill_suggestions()
        self._update_welcome()

    def _update_welcome(self):
        hint = getattr(self, "welcome_hint", None)
        if hint is None: return
        model = self.model_combo.currentData()
        if not model: text = "Choose a model above to get started."
        elif backend_kind(model) == "gguf" and self.runtimes.resolve() is None:
            text = "This model needs the llama.cpp runtime. Install it in Tools \u2192 Settings."
        else: text = ""
        hint.setText(text); hint.setVisible(bool(text))

    def _enter_welcome(self):
        choices = [g for g in GREETINGS if g != self._greeting]
        self._greeting = random.choice(choices); self.welcome_title.setText(self._greeting)
        self._suggestions, self._suggestion_offset = random.sample(SUGGESTIONS, len(SUGGESTIONS)), 0

    def _fill_suggestions(self):
        """One row of prompt chips, as many as fit the width, starting at the current rotation offset."""
        row = self.suggestion_row.layout()
        while row.count():
            item = row.takeAt(0)
            if item.widget() is not None: item.widget().setParent(None)   # gone immediately, not just scheduled
        width = self.suggestion_row.width() or 620; metrics = self.fontMetrics(); used, shown = 0, 0
        row.addStretch(1)
        for n in range(len(self._suggestions)):
            text = self._suggestions[(self._suggestion_offset + n) % len(self._suggestions)]
            chip_width = metrics.horizontalAdvance(text) + 44
            if shown >= 4 or (shown and used + chip_width + 8 > width): break
            chip = button(text, lambda checked=False, p=text: self._use_starter(p)); chip.setObjectName("chip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor); row.addWidget(chip); used += chip_width + 8; shown += 1
        row.addStretch(1); self._shown_suggestions = shown

    def _rotate_suggestions(self):
        if (self._welcome_active and self.isVisible() and not self.composer.toPlainText() and not self.suggestion_row.underMouse()
                and self._shown_suggestions):
            self._suggestion_offset = (self._suggestion_offset + self._shown_suggestions) % len(self._suggestions)
            self._fill_suggestions()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "suggestion_row") and self._welcome_active: self._fill_suggestions()

    def _use_starter(self, prompt):
        self.composer.setPlainText(prompt); self.composer.setFocus()
        cursor = self.composer.textCursor(); cursor.movePosition(cursor.MoveOperation.End); self.composer.setTextCursor(cursor)

    def _save_preferences(self):
        preferences = {**self.preferences, **self.sampling()}
        try:
            atomic_json(self.state_dir/"settings.json",preferences)
            self.preferences = preferences
        except OSError as error:
            self.statusBar().showMessage(f"Could not save settings: {error}")

    def _chat_context_menu(self, position):
        if self.engine.busy: return
        item = self.chat_list.itemAt(position)
        if item is None: return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self.chat_list)
        rename = menu.addAction("Rename chat…")
        delete = menu.addAction("Delete chat")
        action = menu.exec(self.chat_list.viewport().mapToGlobal(position))
        if action == rename: self._rename_chat(chat_id)
        elif action == delete: self._delete_chat(chat_id)

    def _rename_chat(self, chat_id):
        if self.engine.busy: return
        chat = next((chat for chat in self.chats if chat["id"] == chat_id),None)
        if chat is None: return
        title, accepted = QInputDialog.getText(self,"Rename chat","Chat name:",
                                              QLineEdit.EchoMode.Normal,chat.get("title","Conversation"))
        title = " ".join(title.split())
        if not accepted or not title: return
        previous = chat.get("title","Conversation")
        chat["title"] = title
        if not self._save_chats(): chat["title"] = previous
        self._refresh_chats()

    def _delete_chat(self, chat_id):
        if self.engine.busy: return
        index = next((i for i,chat in enumerate(self.chats) if chat["id"] == chat_id),None)
        if index is None: return
        deleted = self.chats.pop(index)
        if not self._save_chats():
            self.chats.insert(index,deleted)
            return
        if self.current_chat and self.current_chat["id"] == chat_id:
            if self.chats: self._select_chat(min(index,len(self.chats)-1))
            else: self._new_chat()
        else: self._refresh_chats()

    def _select_chat(self,index):
        if index < 0 or index >= len(self.chats): return
        if self.engine.busy:
            self._refresh_chats(); return
        self.current_chat = self.chats[index]
        selected = self.model_combo.findData(self.current_chat.get("model",""))
        if selected >= 1: self.model_combo.setCurrentIndex(selected)
        self._display_chat()

    def _save_chats(self):
        # Only saved chats enter self.chats. This also lets a context-menu
        # action persist while an unrelated temporary chat is on screen.
        try:
            atomic_json(self.state_dir/"chats.json",self.chats)
            return True
        except OSError as error:
            self.statusBar().showMessage(f"Could not save history: {error}")
            return False

    def _set_busy(self,busy):
        self._busy = busy
        if busy: self._engine_error = False
        enabled = not busy and not self.training_active
        self.send_button.setEnabled(enabled); self.compare_run.setEnabled(enabled)
        self.stop_button.setEnabled(busy and self.operation=="chat"); self.compare_stop.setEnabled(busy and self.operation=="compare")
        for widget in (self.new_chat_button,self.temporary_chat_button,self.chat_search,self.chat_list): widget.setEnabled(not busy)
        self.composer.setReadOnly(busy or self.training_active)
        for widget in (self.model_combo,self.browse_model_button,self.refresh_model_button,self.compare_a,self.compare_b,self.compare_prompt,self.temperature,self.top_p,self.top_k,self.max_tokens,self.seed):
            widget.setEnabled(enabled)
        self.train_start.setEnabled(not busy and not self.training_active)
        self._update_status()

    def _send_chat(self):
        if self.engine.busy or self.training_active: return
        text = self.composer.toPlainText().strip(); model = self.model_combo.currentData()
        if not text: return
        if not model: self.statusBar().showMessage("Choose a trained model first"); return
        context = completed_context(self.current_chat["messages"])
        self.current_chat["model"]=model
        self.current_chat["messages"].extend([{"role":"user","content":text}, {"role":"assistant","content":"","status":"generating"}])
        if self.current_chat["title"]=="New conversation": self.current_chat["title"]=" ".join(text.split())[:45]
        if not self.current_chat.get("temporary") and not any(c["id"]==self.current_chat["id"] for c in self.chats):
            self.chats.insert(0,self.current_chat)
        self._refresh_chats(); self._save_chats(); self.composer.clear(); self.operation="chat"
        self._show_page(0)
        self._show_chat_content()
        self.transcript.render(self.current_chat["messages"]); self._set_busy(True)
        self.engine.generate(model,context+[{"role":"user","content":text}],self.sampling())

    def _start_compare(self):
        if self.engine.busy or self.training_active: return
        prompt=self.compare_prompt.toPlainText().strip(); a=self.compare_a.currentData(); b=self.compare_b.currentData()
        if not prompt: return
        if not a or not b: self.statusBar().showMessage("Choose both models before comparing"); return
        self.comparison={"id":uuid.uuid4().hex,"created_at":datetime.now().isoformat(),"prompt":prompt,"settings":self.sampling(),
                         "models":[a,b],"responses":[{"role":"assistant","content":"","status":"generating"},{"role":"assistant","content":"","status":"waiting"}]}
        self.compare_index=0; self.compare_cancelled=False; self.operation="compare"; self.export_compare.setEnabled(False)
        for view in self.compare_views: view.render([],"Waiting for response","The selected model will answer this prompt.")
        self._set_busy(True); self._compare_next()

    def _compare_next(self):
        if self._closing:
            return
        if self.compare_cancelled:
            self.comparison["responses"][self.compare_index].update(status="cancelled",note="Comparison stopped")
            self._save_comparison(); self.operation=None; self._set_busy(False)
            self.statusBar().showMessage("Comparison stopped")
            return
        self.operation="compare"
        self.engine.generate(self.comparison["models"][self.compare_index],
                             [{"role":"user","content":self.comparison["prompt"]}],self.comparison["settings"])

    def _active_message(self):
        if self.operation=="chat": return self.current_chat["messages"][-1]
        if self.operation=="compare": return self.comparison["responses"][self.compare_index]
        return None

    def _render_active(self):
        if self.operation=="chat": self.transcript.render(self.current_chat["messages"])
        elif self.operation=="compare": self.compare_views[self.compare_index].render([self._active_message()])

    def _delta(self,text):
        message=self._active_message()
        if message is not None:
            message["content"]+=text; self._render_active()

    def _generation_started(self,event):
        message=self._active_message()
        if "prompt_tokens" not in event: return      # other backends report context use when the reply finishes
        if message is not None:
            message["prompt_tokens"]=event["prompt_tokens"]; message["dropped_messages"]=event["dropped_messages"]
        if self.operation=="chat":
            dropped=event["dropped_messages"]//2
            self.chat_hint.setText(f"Context: {event['prompt_tokens']:,}/{event['context_length']:,} tokens" + (f" · {dropped} older exchanges omitted" if dropped else ""))

    def _finished(self,event):
        message=self._active_message()
        if message is None: return
        reason=event.get("reason"); message["status"]="cancelled" if reason=="cancelled" else "complete"
        tokens=event.get("generated_tokens",0); elapsed=event.get("elapsed_seconds",0)
        message["metrics"]={"generated_tokens":tokens,"elapsed_seconds":elapsed,"reason":reason}
        if event.get("prompt_tokens") is not None and event.get("context_length"):
            message["prompt_tokens"]=event["prompt_tokens"]
            if self.operation=="chat": self.chat_hint.setText(f"Context: {event['prompt_tokens']+tokens:,}/{event['context_length']:,} tokens")
        message["note"]=f"{tokens} tokens · {elapsed:.1f}s" + (" · stopped" if reason=="cancelled" else " · response limit reached" if reason=="length" else "")
        self._render_active()
        if self.operation=="chat": self._save_chats()
        elif self.compare_index==0 and not self.compare_cancelled and reason!="cancelled":
            self.compare_index=1; self.comparison["responses"][1]["status"]="generating"
            QTimer.singleShot(0,self._compare_next); return
        elif self.operation=="compare":
            self._save_comparison()
        self.operation=None; self._set_busy(False); self.statusBar().showMessage("Response stopped" if reason=="cancelled" else "Ready")

    def _failed(self,error):
        message=self._active_message()
        if message is not None:
            message.update(status="failed",note=error); self._render_active()
            if self.operation=="chat": self._save_chats()
            else: self._save_comparison()
        self.operation=None; self._set_busy(False); self._engine_error=True; self._update_status(); self.statusBar().showMessage(error)

    def _stop(self):
        self.compare_cancelled=True; self.engine.cancel()

    def _save_comparison(self):
        self.export_compare.setEnabled(True)
        try: atomic_json(self.state_dir/"comparisons"/(self.comparison["id"]+".json"),self.comparison)
        except OSError as error: self.statusBar().showMessage(f"Could not save comparison: {error}")

    def _export_chat(self):
        if not self.current_chat: return
        path,_=QFileDialog.getSaveFileName(self,"Export conversation",self.current_chat["title"]+".json","JSON (*.json)")
        if path:
            try: atomic_json(Path(path),self.current_chat)
            except OSError as error: self.statusBar().showMessage(str(error))

    def _export_compare(self):
        if not self.comparison: return
        path,_=QFileDialog.getSaveFileName(self,"Export comparison","trance-comparison.json","JSON (*.json)")
        if path:
            try: atomic_json(Path(path),self.comparison)
            except OSError as error: self.statusBar().showMessage(str(error))

    def _browse_config(self):
        path,_=QFileDialog.getOpenFileName(self,"Select training config",str(ROOT/"configs"),"JSON (*.json)")
        if path:
            index=self.config_combo.findData(path)
            if index<0: self.config_combo.addItem(Path(path).name,path); index=self.config_combo.count()-1
            self.config_combo.setCurrentIndex(index)

    def _use_saved_config(self,path,*,show=True):
        index=self.config_combo.findData(path)
        if index<0:self.config_combo.addItem(Path(path).name,path);index=self.config_combo.count()-1
        if show:
            self.config_combo.setCurrentIndex(index);self._load_config();self._show_page(2)

    def _load_config(self):
        path=self.config_combo.currentData()
        if not path: return
        try: self.config_editor.setPlainText(Path(path).read_text())
        except OSError as error: self.config_hint.setText(str(error))

    def _toggle_json(self, on=False):
        self.config_editor.setVisible(on); self.json_note.setVisible(on)
        self.json_toggle.setText("Edit JSON \u25be" if on else "Edit JSON \u25b8")
        self._left_layout.setStretch(self._left_spacer, 0 if on else 1)

    def _validate_editor(self):
        """Update the summary card; the tiles still show when only data/output checks fail."""
        try: config = validate_config(json.loads(self.config_editor.toPlainText()), check_output=False, check_files=False)
        except (ValueError, OSError) as error: self._show_summary(None, f"Not a valid configuration: {error}"); return
        try: validate_config(config); problem = ""
        except (ValueError, OSError) as error: problem = str(error)
        self._show_summary(config, problem)

    def _show_summary(self, config, problem):
        values = {key: ("\u2013", default) for key, default in (("params","Parameters"),("weights","FP32 weights"),("shape","Layers \u00d7 width"),("context","Token context"),("steps","Steps"),("lr","Learning rate"))}
        if config:
            count = parameter_count(config)
            values = {"params": (f"{count/1e6:.1f}M", "Parameters"), "weights": (f"{count*4/1048576:,.0f} MiB", "FP32 weights"),
                      "shape": (f"{config['layers']} \u00d7 {config['embedding_dim']}", f"Layers \u00d7 width \u00b7 {config['attention_heads']} heads"),
                      "context": (f"{config['context_length']:,}", f"Token context \u00b7 vocab {config['vocab_size']:,}"),
                      "steps": (f"{config['training_steps']:,}", f"Steps \u00b7 batch {config['batch_size']}"),
                      "lr": (f"{config['learning_rate']:g}", f"Learning rate \u00b7 clip {config['gradient_clip']:g}")}
        for key, (value, caption) in values.items():
            self.summary_tiles[key][0].setText(value); self.summary_tiles[key][1].setText(caption)
        self.config_hint.setText(problem or "Ready to train: data files found.")
        self.config_hint.setProperty("bad", bool(problem)); self._repolish(self.config_hint)
        self.config_output.setText(f"Output: {config['output_model']}" if config else "")

    def _start_training(self):
        if self.engine.busy: self.train_message.setText("Stop generation before starting training."); return
        try:
            config=validate_config(json.loads(self.config_editor.toPlainText()))
            self.engine.release(); directory=launch_training(config)
            self._attach_training(directory)
            self.train_message.setText("Training continues if you close this window. Stop & save checkpoint ends the run safely.")
        except (ValueError,OSError,RuntimeError) as error: self.train_message.setText(str(error))

    def _attach_training(self,directory):
        self.training_dir=directory; self.tail=EventTail(directory); self.chart.clear(); self.train_log.clear()
        self.progress.setValue(0); self.training_base_step=0; self._poll_training(); self._refresh_runs()

    def _poll_training(self):
        if not self.training_dir:
            directory=latest_run()
            if directory: self._attach_training(directory)
            return
        try:
            for event in self.tail.read():
                kind=event.get("event")
                if kind=="training_started": self.training_base_step=event.get("step",0)
                elif kind=="stage": self.train_metrics.setText(event["stage"]); self.train_log.appendPlainText(event["stage"])
                elif kind=="progress":
                    self.chart.add(event); step=event["step"]; end=event["end_step"]; elapsed=event.get("elapsed_seconds",0)
                    completed=max(0,step-self.training_base_step); total=max(1,end-self.training_base_step)
                    self.progress.setValue(int(1000*completed/total))
                    eta=(end-step)*elapsed/completed if completed else 0
                    remaining="~"+duration(eta) if completed>=25 else "estimating…"
                    text=f"Step {step:,} / {end:,} · elapsed {duration(elapsed)} · remaining {remaining}"
                    self.train_metrics.setText(text)
                    if event.get("valid_loss") is not None: self.train_log.appendPlainText(f"Step {step:,}: train {event['train_loss']:.4f} · validation {event['valid_loss']:.4f}")
                elif kind in ("checkpoint","stopped","complete"):
                    self.train_log.appendPlainText(f"{kind.capitalize()} · step {event.get('step',0):,}\n{event.get('path','')}")
                elif kind=="log": self.train_log.appendPlainText(event.get("message",""))
            status=read_json(self.training_dir/"status.json",{})
            active=run_alive(status)
            if status.get("status") in ("starting","running") and not active:
                self.train_message.setText("The training supervisor is no longer running. Check the run folder for logs and saved checkpoints.")
            elif status.get("status")=="failed": self.train_message.setText("Training failed. " + status.get("error","See the log and run folder for details."))
            elif status.get("status")=="stopped": self.train_message.setText("Stopped safely. The saved optimizer checkpoint can be resumed.")
            elif status.get("status")=="complete": self.train_message.setText("Training complete. Refresh the model list to chat with the saved model.")
            if active != self.training_active:
                self.training_active=active; self._set_busy(self.engine.busy)
                if not active: self.refresh_models()
                self._refresh_runs()
            self.train_stop.setEnabled(active and not (self.training_dir/"stop.request").exists())
            self.resume_button.setEnabled(not active and status.get("status")=="stopped" and bool(status.get("checkpoint")))
            self._update_status()
        except (ValueError,OSError) as error: self.train_message.setText(str(error))

    def _stop_training(self):
        if not self.training_dir: return
        try:
            stop_training(self.training_dir); self.train_message.setText("Stop requested. Waiting for the current step and checkpoint save…"); self.train_stop.setEnabled(False)
        except (ValueError,OSError) as error: self.train_message.setText(str(error))

    def _resume_training(self):
        try:
            status=read_json(self.training_dir/"status.json",{}); config=read_json(self.training_dir/"config.json")
            # Interrupted checkpoints are replaced on each subsequent stop.
            # Read the immutable run target, not the current contents of the
            # original resume file, to preserve the total budget after multiple stops.
            remaining=status["end_step"]-status["step"]
            if remaining<=0: raise ValueError("The configured step budget is already complete")
            checkpoint=model_info(ROOT/status["checkpoint"])
            config.update(resume_model=str(ROOT/status["checkpoint"]),training_steps=remaining,vocab_size=checkpoint["vocab_size"])
            self.config_editor.setPlainText(json.dumps(config,indent=2))
            self.train_message.setText(f"Resume configuration prepared with {remaining:,} remaining steps. Click Start training to resume.")
        except (ValueError,OSError,KeyError) as error: self.train_message.setText(str(error))

    def _open_run_folder(self):
        directory=self.training_dir or ROOT/"models"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def closeEvent(self,event):
        self._closing=True
        if self.operation:
            message=self._active_message()
            if message is not None: message.update(status="cancelled",note="Stopped when the application closed")
            if self.operation=="compare": self._save_comparison()
        self.engine.close(); self.runtimes.close(); self._save_chats()
        self._save_preferences()
        event.accept()


def main():
    parser=argparse.ArgumentParser(description="Trance LLM desktop GUI")
    parser.add_argument("--screenshot",type=Path,help="Render the app to a PNG and exit; no model computation")
    parser.add_argument("--page",choices=("chat","compare","training","configs","docs","settings"),default="chat")
    args=parser.parse_args()
    app=QApplication.instance() or QApplication([]); app.setApplicationName(APP_NAME); app.setStyle("Fusion"); app.setStyleSheet(STYLE)
    lock=None
    if not args.screenshot:
        STATE.mkdir(parents=True,exist_ok=True)
        # Keep the legacy filename so earlier releases share the same lock.
        lock=QLockFile(str(STATE/"studio.lock")); lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            QMessageBox.information(None,APP_NAME,f"{APP_NAME} is already open. Use the existing window.")
            return 0
    # Layout captures never overwrite live conversation history or preferences.
    capture_state=tempfile.TemporaryDirectory(prefix="trance-layout-") if args.screenshot else None
    window=MainWindow(state_dir=Path(capture_state.name) if capture_state else None)
    window.show(); window._show_page(("chat","compare","training","configs","docs","settings").index(args.page))
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            window.grab().save(str(args.screenshot)); window.close(); app.quit()
        QTimer.singleShot(350,capture)
    result=app.exec()
    if lock: lock.unlock()
    if capture_state: capture_state.cleanup()
    return result
