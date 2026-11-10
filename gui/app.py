"""Trance LLM: chat, comparison and training around the native CUDA engine."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
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
    QMenu, QInputDialog, QLineEdit)

from .engine import EngineClient
from . import APP_NAME
from .storage import ROOT, STATE, atomic_json, read_json, new_chat, completed_context, discover_models, model_info, validate_config
from .training import EventTail, latest_run, launch_training, stop_training, run_alive
from .widgets import Composer, Transcript, LossChart, ConversationList, STYLE
from .config_builder import ConfigBuilder
from .documentation import Documentation


def label(text, name=None):
    widget = QLabel(text)
    if name: widget.setObjectName(name)
    return widget


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
        self.history_expanded = self.preferences.get("history_expanded", False) is True
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
        self._closing = False
        self.models = []
        self.engine = EngineClient(self, binary=binary)
        self.engine.delta.connect(self._delta)
        self.engine.started.connect(self._generation_started)
        self.engine.finished.connect(self._finished)
        self.engine.failed.connect(self._failed)
        self.engine.state_changed.connect(self.statusBar().showMessage)
        self.engine.ready.connect(lambda gpu: self.gpu_status.setText(gpu.split(" (compute")[0]))
        self.engine.loaded.connect(lambda info: self.statusBar().showMessage(f"Loaded {friendly(info['model'])} · {info['parameters']/1e6:.2f}M parameters"))
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
        left.addWidget(label("CHAT", "eyebrow"))
        self.temporary_chat_button = button("Temporary chat", lambda: self._new_chat(temporary=True))
        self.new_chat_button = button("New chat", self._new_chat)
        self.history_button = button("Chat history", self._show_history)
        for nav in (self.temporary_chat_button,self.new_chat_button,self.history_button):
            nav.setObjectName("nav"); nav.setCheckable(True); left.addWidget(nav)
        self.chat_list = ConversationList(); self.chat_list.currentRowChanged.connect(self._select_chat)
        self.chat_list.setMinimumHeight(60);self.chat_list.setMaximumHeight(180)
        self.chat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self._chat_context_menu)
        left.addWidget(self.chat_list)
        self._set_history_expanded(self.history_expanded)
        for section, entries in (("TRAIN",((2,"Train"),(3,"Configs"))),
                                 ("TOOLS",((1,"Compare"),(4,"Documentation")))):
            left.addSpacing(10);left.addWidget(label(section,"eyebrow"))
            for index,name in entries:
                nav=button(name,lambda checked=False,page=index:self._show_page(page));nav.setObjectName("nav");nav.setCheckable(True)
                left.addWidget(nav);self.nav[index]=nav
        left.addStretch(1)
        left.addWidget(label("ON THIS COMPUTER", "eyebrow"))
        self.gpu_status = label("CUDA engine · idle", "muted"); self.gpu_status.setWordWrap(True); left.addWidget(self.gpu_status)
        layout.addWidget(sidebar)
        self.pages = QStackedWidget(); layout.addWidget(self.pages, 1)
        self._build_chat(); self._build_compare(); self._build_training()
        self.config_builder=ConfigBuilder(root=ROOT);self.pages.addWidget(self.config_builder)
        self.config_builder.saved.connect(lambda path:self._use_saved_config(path,show=False))
        self.config_builder.open_training.connect(self._use_saved_config)
        self.documentation=Documentation();self.pages.addWidget(self.documentation)
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
        self.chat_content.addWidget(temporary_empty); box.addWidget(self.chat_content,1)
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
        box.addWidget(self.composer_panel)

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
        left.addWidget(label("RUN CONFIGURATION", "eyebrow"))
        self.config_editor = QPlainTextEdit(); self.config_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap); self.config_editor.setFont(QFont("DejaVu Sans Mono",10)); left.addWidget(self.config_editor,1)
        self.config_hint = label("", "muted"); self.config_hint.setWordWrap(True); left.addWidget(self.config_hint)
        left.addWidget(label("Edits are saved as a run snapshot when training starts.","muted")); split.addWidget(editor_panel)
        metrics = QWidget(); right = QVBoxLayout(metrics); right.setContentsMargins(12,0,0,0)
        right.addWidget(label("TRAINING / VALIDATION LOSS", "eyebrow")); right.addWidget(label("Blue: training     Teal: validation", "muted"))
        self.chart = LossChart(); right.addWidget(self.chart,1)
        self.progress = QProgressBar(); self.progress.setRange(0,1000); self.progress.setValue(0); right.addWidget(self.progress)
        self.train_metrics = label("No training run yet", "muted"); self.train_metrics.setWordWrap(True); right.addWidget(self.train_metrics)
        self.train_log = QPlainTextEdit(); self.train_log.setReadOnly(True); self.train_log.setMaximumBlockCount(500); self.train_log.setMaximumHeight(160); self.train_log.setPlaceholderText("Progress and checkpoint locations appear here."); right.addWidget(self.train_log)
        split.addWidget(metrics); split.setSizes([390,600]); box.addWidget(split,1)
        self.train_message = label("Training only starts when you click Start training.","muted"); self.train_message.setWordWrap(True); box.addWidget(self.train_message)
        row = QHBoxLayout(); row.addWidget(button("Open run folder", self._open_run_folder)); self.resume_button = button("Resume checkpoint",self._resume_training); self.resume_button.setEnabled(False); row.addWidget(self.resume_button); row.addStretch()
        self.train_stop = button("Stop & save checkpoint", self._stop_training); self.train_stop.setEnabled(False); row.addWidget(self.train_stop)
        self.train_start = button("Start training",self._start_training,True); row.addWidget(self.train_start); box.addLayout(row)
        self.config_editor.textChanged.connect(self._validate_editor)
        self._load_config()

    def _show_page(self,index):
        self.pages.setCurrentIndex(index)
        self.settings.setVisible(index in (0,1))
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
        self.models = [m for m in existing.values() if Path(m["path"]).is_file()]
        for combo in (self.model_combo,self.compare_a,self.compare_b):
            previous = combo.currentData(); combo.blockSignals(True); combo.clear(); combo.addItem("Choose a model…","")
            for info in self.models:
                combo.addItem(f"{friendly(info['path'])}  ·  {info['parameters']/1e6:.1f}M",info["path"])
            index = combo.findData(previous)
            if index < 1 and combo is not self.compare_b and self.models: index = 1
            combo.setCurrentIndex(max(0,index)); combo.blockSignals(False)
        self._model_changed()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self,"Choose a Trance model",str(ROOT/"models"),"Trance models (*.bin)")
        if not path: return
        try:
            info = model_info(Path(path)); self.models.append(info); self.refresh_models()
            self.model_combo.setCurrentIndex(self.model_combo.findData(info["path"]))
        except (OSError,ValueError) as error: self.statusBar().showMessage(str(error))

    def _model_changed(self):
        path = self.model_combo.currentData()
        info = next((m for m in self.models if m["path"]==path),None)
        if info:
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
        empty_temporary = bool(self.current_chat and self.current_chat.get("temporary") and not self.current_chat.get("messages"))
        self.chat_content.setCurrentIndex(1 if empty_temporary else 0)

    def _show_history(self):
        self._set_history_expanded(not self.history_expanded)
        self._save_preferences()

    def _set_history_expanded(self, expanded):
        self.history_expanded = expanded
        self.chat_list.setVisible(expanded)
        self.history_button.setChecked(expanded)
        self.history_button.setText("Chat history  " + ("▾" if expanded else "▸"))
        self.history_button.setAccessibleName("Chat history")
        self.history_button.setToolTip("Collapse chat history" if expanded else "Expand chat history")

    def _save_preferences(self):
        preferences = {**self.preferences, **self.sampling(), "history_expanded":self.history_expanded}
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
        enabled = not busy and not self.training_active
        self.send_button.setEnabled(enabled); self.compare_run.setEnabled(enabled)
        self.stop_button.setEnabled(busy and self.operation=="chat"); self.compare_stop.setEnabled(busy and self.operation=="compare")
        for widget in (self.new_chat_button,self.temporary_chat_button,self.history_button,self.chat_list): widget.setEnabled(not busy)
        self.composer.setReadOnly(busy or self.training_active)
        for widget in (self.model_combo,self.browse_model_button,self.refresh_model_button,self.compare_a,self.compare_b,self.compare_prompt,self.temperature,self.top_p,self.top_k,self.max_tokens,self.seed):
            widget.setEnabled(enabled)
        self.train_start.setEnabled(not busy and not self.training_active)

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
        self.operation=None; self._set_busy(False); self.statusBar().showMessage(error)

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

    def _validate_editor(self):
        try:
            config=validate_config(json.loads(self.config_editor.toPlainText()))
            d=config['embedding_dim']; f=config['feed_forward_dim']; v=config['vocab_size']; t=config['context_length']; l=config['layers']
            count=2*v*d+t*d+2*d+v+l*(4*d*d+2*d*f+9*d+f)
            self.config_hint.setText(f"{count/1e6:.2f}M parameters · {config['training_steps']:,} steps\nOutput: {config['output_model']}")
        except (ValueError,OSError) as error: self.config_hint.setText(str(error))

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
        self.progress.setValue(0); self.training_base_step=0; self._poll_training()

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
            self.train_stop.setEnabled(active and not (self.training_dir/"stop.request").exists())
            self.resume_button.setEnabled(not active and status.get("status")=="stopped" and bool(status.get("checkpoint")))
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
        self.engine.close(); self._save_chats()
        self._save_preferences()
        event.accept()


def main():
    parser=argparse.ArgumentParser(description="Trance LLM desktop GUI")
    parser.add_argument("--screenshot",type=Path,help="Render the app to a PNG and exit; no model computation")
    parser.add_argument("--page",choices=("chat","compare","training","configs","docs"),default="chat")
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
    window.show(); window._show_page(("chat","compare","training","configs","docs").index(args.page))
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            window.grab().save(str(args.screenshot)); window.close(); app.quit()
        QTimer.singleShot(350,capture)
    result=app.exec()
    if lock: lock.unlock()
    if capture_state: capture_state.cleanup()
    return result
