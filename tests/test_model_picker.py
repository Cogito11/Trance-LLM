"""Loading models on purpose: the picker modal, Eject, the default model, Save Chat and the shared chat menu.

No GPU is needed. The native engine is replaced by tests/fake_engine.py (same JSON-lines protocol) and llama-server by
tests/fake_llama_server.py, so the real client code, process handling and window run end to end.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gguf import chat_model, write as write_gguf  # noqa: E402

from gui import theme  # noqa: E402
from gui.app import MainWindow  # noqa: E402
from gui.backends.router import BackendRouter  # noqa: E402
from gui.engine import EngineClient  # noqa: E402
from gui.runtimes import SERVER_NAME, Asset, Flavor  # noqa: E402
from test_runtimes import FileServer, make_tar  # noqa: E402
from gui.gguf import read_gguf_info  # noqa: E402
from gui.runtimes import RuntimeInfo  # noqa: E402
from gui.storage import atomic_json, new_chat, read_json  # noqa: E402
from gui.widgets import STYLE  # noqa: E402

HERE = Path(__file__).resolve().parent
FAKE_ENGINE, FAKE_SERVER = HERE / "fake_engine.py", HERE / "fake_llama_server.py"
RUNTIME = RuntimeInfo(FAKE_SERVER, "custom", "b4242", "fake")
SETTINGS = {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "max_new_tokens": 50, "seed": 11}


def pump(app, condition, timeout=20.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents(); time.sleep(0.005)
    return condition()


def count_processes(pattern: str) -> int:
    """How many processes match `pattern` (a leading [x] keeps pgrep from matching itself)."""
    return int(subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True).stdout.strip() or 0)


class Recorder:
    def __init__(self, source, *names):
        self.events = []
        for name in names:
            getattr(source, name).connect(lambda *a, n=name: self.events.append((n, a[0] if a else None)))

    def names(self):
        return [n for n, _ in self.events]


class AppCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion"); cls.app.setStyleSheet(STYLE)


class EngineLoadTests(AppCase):
    """EngineClient.load(): put a model in memory without generating anything."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "commands.log"
        patcher = patch.dict(os.environ, {"FAKE_ENGINE_LOG": str(self.log), "FAKE_ENGINE_LOAD_SECONDS": "0.1"})
        patcher.start(); self.addCleanup(patcher.stop)
        self.engine = EngineClient(binary=FAKE_ENGINE)
        self.rec = Recorder(self.engine, "loaded", "unloaded", "failed", "finished")
        self.busy_when_loaded = []
        self.engine.loaded.connect(lambda _info: self.busy_when_loaded.append(self.engine.busy))

    def tearDown(self):
        self.engine.close(); self.tmp.cleanup()

    def commands(self):
        return self.log.read_text().split("\n")[:-1] if self.log.exists() else []

    def load(self, name):
        self.engine.load(str(Path(self.tmp.name) / name))
        self.assertTrue(pump(self.app, lambda: not self.engine.busy), "engine stayed busy")
        self.app.processEvents()

    def test_load_reports_loaded_without_generating_and_is_idle_by_then(self):
        self.load("a.bin")
        self.assertEqual(self.rec.names(), ["loaded"])
        self.assertEqual([c.split()[0] for c in self.commands()], ["load"])      # no generate
        self.assertEqual(self.busy_when_loaded, [False])                          # listeners may offer Eject straight away

    def test_loading_the_model_that_is_already_loaded_sends_nothing(self):
        self.load("a.bin"); self.load("a.bin")
        self.assertEqual(self.rec.names(), ["loaded", "loaded"])
        self.assertEqual(len(self.commands()), 1)

    def test_loading_another_model_replaces_the_first(self):
        self.load("a.bin"); self.load("b.bin")
        self.assertEqual([Path(c.split()[1]).name for c in self.commands()], ["a.bin", "b.bin"])
        self.assertTrue(self.engine._loaded.endswith("b.bin"))

    def test_a_failed_load_leaves_nothing_loaded_and_the_engine_usable(self):
        self.load("a.bin"); self.load("BAD.bin")
        self.assertEqual(self.rec.names(), ["loaded", "failed"])
        self.assertEqual(self.rec.events[-1][1], "Cannot load model")
        self.assertEqual(self.engine._loaded, "")
        self.assertNotEqual(self.engine.process.state().name, "Running")          # restarted clean: nothing is guessed about GPU memory
        self.load("c.bin")                                                        # and the next load simply works
        self.assertEqual(self.rec.names()[-1], "loaded")

    def test_a_crash_is_reported_but_a_deliberate_release_is_not(self):
        self.load("a.bin")
        self.engine.release(); self.app.processEvents()
        self.assertNotIn("unloaded", self.rec.names())
        self.load("a.bin")
        self.engine.process.kill()
        self.assertTrue(pump(self.app, lambda: "unloaded" in self.rec.names()))

    def test_load_is_refused_while_busy_and_can_be_cancelled(self):
        with patch.dict(os.environ, {"FAKE_ENGINE_LOAD_SECONDS": "3"}):
            self.engine.load(str(Path(self.tmp.name) / "slow.bin"))
        with self.assertRaises(RuntimeError): self.engine.load("/x/other.bin")
        self.engine.cancel()
        self.assertTrue(pump(self.app, lambda: not self.engine.busy))
        self.assertEqual(self.rec.events[-1][0], "finished")
        self.assertEqual(self.rec.events[-1][1]["reason"], "cancelled")


class OneModelAtATimeTests(AppCase):
    """Whatever the user picks, at most one process holds a model, and nothing is left after close."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        patcher = patch.dict(os.environ, {"FAKE_ENGINE_LOAD_SECONDS": "0.1"}); patcher.start(); self.addCleanup(patcher.stop)
        self.t1, self.t2 = str(root / "a.bin"), str(root / "b.bin")
        self.g1, self.g2 = str(root / "a.gguf"), str(root / "b.gguf")
        for path in (self.t1, self.t2, self.g1, self.g2): Path(path).write_bytes(b"GGUF-stand-in")
        self.bad_gguf = root / "bad.gguf"; self.bad_gguf.write_bytes(b"BAD model")
        self.router = BackendRouter(binary=FAKE_ENGINE, resolver=lambda: RUNTIME)
        self.rec = Recorder(self.router, "loaded", "unloaded", "failed")

    def tearDown(self):
        self.router.close(); self.tmp.cleanup()

    def load(self, path):
        self.router.load(path)
        self.assertTrue(pump(self.app, lambda: not self.router.busy), "router stayed busy")
        self.app.processEvents()

    def holders(self):
        return count_processes("[f]ake_engine.py engine") + count_processes("[f]ake_llama_server.py")

    def test_every_swap_leaves_exactly_one_model_in_memory(self):
        self.assertEqual(self.holders(), 0)
        for path in (self.g1, self.g2, self.t1, self.t2, self.g1):
            self.load(path)
            self.assertEqual(self.router.loaded_model, str(Path(path).resolve()))
            self.assertEqual(self.holders(), 1, f"after loading {Path(path).name}")
        self.router.release()
        self.assertEqual((self.router.loaded_model, self.holders()), ("", 0))

    def test_a_native_to_gguf_swap_stops_the_native_engine_first(self):
        self.load(self.t1); self.assertEqual(self.router.trance.process.state().name, "Running")
        self.load(self.g1)
        self.assertEqual(self.router.trance.process.state().name, "NotRunning")
        self.assertTrue(self.router.llama.running)

    def test_the_replaced_model_is_announced_as_unloaded_before_the_new_one_loads(self):
        self.load(self.g1); self.rec.events.clear()
        self.load(self.g2)
        self.assertEqual(self.rec.names(), ["unloaded", "loaded"])

    def test_a_failed_load_leaves_nothing_loaded_not_the_old_model(self):
        self.load(self.g1)
        self.load(str(self.bad_gguf))
        self.assertEqual(self.rec.names()[-1], "failed")
        self.assertEqual((self.router.loaded_model, self.holders()), ("", 0))
        self.load(self.t1); self.load(str(Path(self.tmp.name) / "BAD.bin"))
        self.assertEqual((self.router.loaded_model, self.holders()), ("", 0))

    def test_close_stops_every_process(self):
        for path in (self.g1, self.t1):
            self.load(path)
        self.router.close(); time.sleep(0.2)
        self.assertEqual(self.holders(), 0)


class WindowCase(AppCase):
    """A real MainWindow on stand-in models, with the engine mocked unless a test asks for the fakes."""
    use_fakes = False

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="trance-picker-")
        self.state = Path(self.temp.name) / "state"; root = Path(self.temp.name) / "models"; root.mkdir()
        patcher = patch.dict(os.environ, {"FAKE_ENGINE_LOAD_SECONDS": "0.1"}); patcher.start(); self.addCleanup(patcher.stop)
        self.trance = []
        for name, params, ctx in (("trance1.1_21m.bin", 21_000_000, 512), ("trance1.2_30m.bin", 30_400_000, 1024)):
            path = root / name; path.write_bytes(b"x")
            self.trance.append({"path": str(path.resolve()), "name": name, "parameters": params, "context_length": ctx, "vocab_size": 1024,
                                "width": 512, "layers": 8, "heads": 8, "step": 1000, "optimizer": False, "feed_forward_dim": 2048})
        self.gguf = [read_gguf_info(write_gguf(root / f"{name}.gguf", *chat_model(title, "llama", 4096, 15)))
                     for name, title in (("qwen", "Qwen Tiny"), ("llama", "Llama Tiny"))]
        self.patches = [patch("gui.app.latest_run", return_value=None), patch("gui.app.discover_models", return_value=self.trance),
                        patch("gui.app.discover_gguf", return_value=self.gguf)]
        for p in self.patches: p.start()

    def tearDown(self):
        for window in getattr(self, "windows", []): window.close()
        self.app.processEvents()
        for p in reversed(self.patches): p.stop()
        sheet = theme.apply_palette("teal")
        if self.app.styleSheet() != sheet: self.app.setStyleSheet(sheet)
        self.temp.cleanup()

    def window(self, preferences=None, binary=None):
        if preferences is not None: atomic_json(self.state / "settings.json", preferences)
        w = MainWindow(state_dir=self.state, binary=binary or (FAKE_ENGINE if self.use_fakes else Path("/unused-cpu-test-engine")))
        w.timer.stop()
        if self.use_fakes:
            w.runtimes.resolve = lambda: RUNTIME
            w.engine.llama._resolver = lambda: RUNTIME
        w.show(); self.windows = getattr(self, "windows", []) + [w]
        return w

    def idle(self, w, timeout=20):
        self.assertTrue(pump(self.app, lambda: not w.engine.busy and w.operation is None, timeout), "window stayed busy")
        self.app.processEvents()

    def choose_from_menu(self, trigger, label):
        shown = []

        class Menu(QMenu):
            def exec(menu, position=None, *args):
                shown.extend((a.text(), a.isEnabled()) for a in menu.actions())
                return next((a for a in menu.actions() if a.text() == label and a.isEnabled()), None)
        with patch("gui.app.QMenu", Menu): trigger()
        return shown

    def row(self, w, path):
        lst = w.model_picker.list
        for i in range(lst.count()):
            if lst.item(i).data(Qt.ItemDataRole.UserRole) == path:
                return lst.visualItemRect(lst.item(i)).center()
        self.fail(f"{path} is not listed")

    def chat_with(self, w, messages=(("hello", "hi there"),), **extra):
        chat = new_chat(w.active_model); chat.update(extra)
        for q, a in messages:
            chat["messages"] += [{"role": "user", "content": q}, {"role": "assistant", "content": a, "status": "complete"}]
        return chat

    def show_chat(self, w, chat):
        w.current_chat = chat; w._display_chat()


class PickerAndLoadingTests(WindowCase):
    def test_the_header_button_says_no_model_loaded_until_one_is_loaded(self):
        w = self.window()
        self.assertEqual(w.model_button.label(), "No Model Loaded, Pick one to use")
        self.assertFalse(w.eject_button.isVisibleTo(w))
        x = lambda widget: widget.mapTo(w, widget.rect().topLeft()).x()
        y = lambda widget: widget.mapTo(w, widget.rect().center()).y()
        self.assertGreater(x(w.model_button), x(w.chat_title))                                   # in the same top row as the large heading
        self.assertLess(abs(y(w.model_button) - y(w.chat_title)), 20)

    def test_the_model_controls_are_centered_in_the_top_bar_and_never_change_size(self):
        w = self.window(); w.resize(1400, 800); self.app.processEvents()
        bar = w.pages.currentWidget()
        def measure():
            self.app.processEvents()
            left = w.model_controls.mapTo(bar, w.model_controls.rect().topLeft()).x()
            return w.model_controls.width(), left + w.model_controls.width() / 2 - bar.width() / 2
        states = [measure()]                                                                      # nothing loaded: the button fills the group
        for path in (self.trance[1]["path"], self.gguf[0]["path"]):
            w.engine.loaded_model = path; w._active_model_changed(); states.append(measure())     # loaded: button + Eject
        long = dict(self.gguf[0], title="An Extraordinarily Long Model Name " * 4)
        w.models[:] = [long]; w.engine.loaded_model = long["path"]; w._active_model_changed(); states.append(measure())
        self.show_chat(w, self.chat_with(w, temporary=True, title="Temporary chat")); states.append(measure())
        self.assertEqual({width for width, _ in states}, {280})                                   # one consistent size throughout
        for _, offset in states: self.assertLessEqual(abs(offset), 8)                             # and in the middle of the bar
        self.assertLess(w.model_button.width() + w.eject_button.width() + 6, 281)

    def test_a_loaded_model_shows_its_name_beside_a_separate_eject_button(self):
        w = self.window()
        w.engine.loaded_model = self.trance[1]["path"]; w._active_model_changed()
        self.assertEqual(w.model_button.label(), "Trance 1.2")
        self.assertTrue(w.eject_button.isVisibleTo(w)); self.assertEqual(w.eject_button.text(), "Eject")
        self.assertIsNot(w.eject_button, w.model_button)
        w.engine.loaded_model = self.gguf[0]["path"]; w._active_model_changed()
        self.assertEqual(w.model_button.label(), "Qwen Tiny")

    def test_a_long_model_name_is_elided_instead_of_widening_the_header(self):
        w = self.window(); w.resize(1160, 700)
        long = dict(self.gguf[0], title="An Extraordinarily Long Model Name " * 4)
        w.models[:] = [long]; w.engine.loaded_model = long["path"]; w._active_model_changed()
        self.app.processEvents()
        self.assertLessEqual(w.model_controls.width(), 460)
        self.assertLess(w.model_button.width(), w.model_button.sizeHint().width())

    def test_the_picker_lists_the_loaded_model_first_and_search_narrows_it(self):
        w = self.window()
        w.engine.loaded_model = self.gguf[1]["path"]; w._active_model_changed()
        QTest.mouseClick(w.model_button, Qt.MouseButton.LeftButton)
        picker = w.model_picker
        self.assertTrue(picker.isVisible())
        self.assertEqual(picker.visible_paths()[0], self.gguf[1]["path"])
        self.assertEqual(len(picker.visible_paths()), 4)
        self.assertTrue(picker.list.item(0).data(Qt.ItemDataRole.UserRole + 2))             # carries the Loaded badge
        QTest.keyClicks(picker.search, "trance 1.2")                                         # every word must match
        self.assertEqual(picker.visible_paths(), [self.trance[1]["path"]])
        picker.search.setText("nothing like this")
        self.assertEqual(picker.visible_paths(), []); self.assertIn("No models match", picker.empty.text())
        picker.search.clear(); self.assertEqual(len(picker.visible_paths()), 4)
        picker.dismiss(); self.assertFalse(picker.isVisible())

    def test_choosing_a_model_loads_it_and_closes_the_picker(self):
        w = self.window()
        with patch.object(w.engine, "load") as load:
            QTest.mouseClick(w.model_button, Qt.MouseButton.LeftButton)
            QTest.mouseClick(w.model_picker.list.viewport(), Qt.MouseButton.LeftButton, pos=self.row(w, self.trance[0]["path"]))
            load.assert_called_once_with(self.trance[0]["path"])
        self.assertFalse(w.model_picker.isVisible())
        self.assertEqual(w.model_button.label(), "Loading Trance 1.1\u2026")
        self.assertEqual((w.eject_button.text(), w.eject_button.isEnabled()), ("Cancel", True))   # a slow load can be abandoned
        self.assertFalse(w.model_button.isEnabled()); self.assertFalse(w.send_button.isEnabled())
        self.assertEqual(w.status_title.text(), "Loading model\u2026")
        self.assertEqual(w.welcome_hint.text(), "Loading the model\u2026")
        self.assertFalse(w.composer.isReadOnly())                                             # typing ahead is fine

    def test_keyboard_picks_the_highlighted_model(self):
        w = self.window()
        with patch.object(w.engine, "load") as load:
            w._open_model_picker()
            QTest.keyClick(w.model_picker.search, Qt.Key.Key_Down)
            QTest.keyClick(w.model_picker.search, Qt.Key.Key_Return)
            load.assert_called_once_with(self.trance[1]["path"])

    def test_choosing_the_loaded_model_again_does_nothing(self):
        w = self.window(); w.engine.loaded_model = self.trance[0]["path"]; w._active_model_changed()
        with patch.object(w.engine, "load") as load:
            w._open_model_picker()
            QTest.mouseClick(w.model_picker.list.viewport(), Qt.MouseButton.LeftButton, pos=self.row(w, self.trance[0]["path"]))
            load.assert_not_called()
        self.assertFalse(w._busy)

    def test_escape_and_the_backdrop_close_the_picker_but_the_card_does_not(self):
        w = self.window(); picker = w.model_picker
        w._open_model_picker(); QTest.keyClick(picker.search, Qt.Key.Key_Escape); self.assertFalse(picker.isVisible())
        w._open_model_picker(); QTest.mouseClick(picker.card, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
        self.assertTrue(picker.isVisible())                                                      # a click on the card itself stays open
        QTest.mouseClick(picker, Qt.MouseButton.LeftButton, pos=QPoint(3, 3))                    # the dimmed backdrop closes it
        self.assertFalse(picker.isVisible())
        w._open_model_picker(); QTest.mouseClick(picker.close_button, Qt.MouseButton.LeftButton); self.assertFalse(picker.isVisible())

    def test_the_picker_does_not_open_while_a_reply_is_written_or_training_runs(self):
        w = self.window()
        w._set_busy(True); QTest.mouseClick(w.model_button, Qt.MouseButton.LeftButton); self.assertFalse(w.model_picker.isVisible())
        self.assertFalse(w.model_button.isEnabled())
        w._set_busy(False); w.training_active = True; w._open_model_picker(); self.assertFalse(w.model_picker.isVisible())

    def test_eject_in_the_picker_matches_the_header_button(self):
        w = self.window(); w.engine.loaded_model = self.trance[0]["path"]; w._active_model_changed()
        w._open_model_picker(); self.assertTrue(w.model_picker.eject_button.isEnabled())
        with patch.object(w.engine, "release", wraps=w.engine.release) as release:
            QTest.mouseClick(w.model_picker.eject_button, Qt.MouseButton.LeftButton)
            release.assert_called_once()
        self.assertEqual(w.model_button.label(), "No Model Loaded, Pick one to use")
        self.assertTrue(w.model_picker.isVisible())                                           # stays open to pick another
        self.assertFalse(w.model_picker.eject_button.isEnabled()); self.assertEqual(w.model_picker.visible_paths()[0], self.trance[0]["path"])

    def test_refresh_and_browse_in_the_picker_reach_the_window(self):
        w = self.window()
        with patch.object(w, "refresh_models") as refresh:
            w.model_picker.refresh_requested.disconnect(); w.model_picker.refresh_requested.connect(w.refresh_models)
            w._open_model_picker(); QTest.mouseClick(w.model_picker.refresh_button, Qt.MouseButton.LeftButton)
            refresh.assert_called_once(); self.assertTrue(w.model_picker.isVisible())
        with patch.object(w, "_browse_model") as browse:
            w.model_picker.browse_requested.disconnect(); w.model_picker.browse_requested.connect(w._browse_model)
            QTest.mouseClick(w.model_picker.browse_button, Qt.MouseButton.LeftButton)
            browse.assert_called_once(); self.assertFalse(w.model_picker.isVisible())          # out of the way of the file dialog

    def test_sending_without_a_model_explains_and_never_reaches_the_engine(self):
        w = self.window()
        with patch.object(w.engine, "generate") as generate:
            w.composer.setPlainText("hello"); w._send_chat()
            generate.assert_not_called()
        self.assertIn("No model is loaded", w.statusBar().currentMessage())
        self.assertEqual(w.composer.toPlainText(), "hello")                                    # the draft is kept
        self.assertEqual(w.current_chat["messages"], [])

    def test_a_failed_load_restores_the_button_and_says_why(self):
        w = self.window()
        with patch.object(w.engine, "load"):
            w._load_model(self.trance[0]["path"])
        w._failed("Cannot load model")
        self.assertEqual(w.model_button.label(), "No Model Loaded, Pick one to use")
        self.assertTrue(w.model_button.isEnabled()); self.assertFalse(w.eject_button.isVisibleTo(w))
        self.assertEqual(w.statusBar().currentMessage(), "Cannot load model"); self.assertIsNone(w.operation)

    def test_cancelling_a_load_goes_back_to_nothing_loaded(self):
        w = self.window()
        with patch.object(w.engine, "load"), patch.object(w.engine, "cancel") as cancel:
            w._load_model(self.trance[0]["path"])
            QTest.mouseClick(w.eject_button, Qt.MouseButton.LeftButton)
            cancel.assert_called_once()
        w._finished({"reason": "cancelled", "generated_tokens": 0, "elapsed_seconds": 0})       # what the engine reports
        self.assertEqual((w.model_button.label(), w._busy, w.operation), ("No Model Loaded, Pick one to use", False, None))
        self.assertIn("cancelled", w.statusBar().currentMessage())

    def test_training_unloads_the_model_and_the_button_follows(self):
        w = self.window(); w.engine.trance.loaded.emit({"model": self.trance[0]["path"], "parameters": 1})
        self.assertEqual(w.model_button.label(), "Trance 1.1")
        w.engine.release()
        self.assertEqual(w.model_button.label(), "No Model Loaded, Pick one to use")

    def test_the_header_fits_in_the_tightest_state_at_the_minimum_window_width(self):
        w = self.window(); w.resize(w.minimumWidth(), w.minimumHeight())
        w.engine.loaded_model = self.trance[1]["path"]; w._active_model_changed()
        self.show_chat(w, self.chat_with(w, temporary=True, title="Temporary chat"))
        self.app.processEvents()
        header = w.pages.currentWidget().layout().itemAt(0).layout()
        self.assertTrue(w.save_chat_button.isVisibleTo(w) and w.chat_menu_button.isVisibleTo(w) and w.eject_button.isVisibleTo(w))
        self.assertLessEqual(header.minimumSize().width(), w.pages.currentWidget().width() - 60)
        self.assertGreaterEqual(w.chat_title.width(), w.chat_title.sizeHint().width())          # the heading is never clipped


class WithFakeProcessesTests(WindowCase):
    use_fakes = True

    def test_picking_models_through_the_window_swaps_them_one_at_a_time(self):
        w = self.window()
        self.assertEqual(w.model_button.label(), "No Model Loaded, Pick one to use")
        for path, label in ((self.gguf[0]["path"], "Qwen Tiny"), (self.trance[1]["path"], "Trance 1.2"),
                            (self.gguf[1]["path"], "Llama Tiny"), (self.trance[0]["path"], "Trance 1.1")):
            w._open_model_picker()
            QTest.mouseClick(w.model_picker.list.viewport(), Qt.MouseButton.LeftButton, pos=self.row(w, path))
            self.assertTrue(w._busy and w.model_button.label().startswith("Loading"))
            self.idle(w)
            self.assertEqual((w.model_button.label(), w.active_model), (label, path))
            self.assertTrue(w.eject_button.isVisibleTo(w) and w.eject_button.isEnabled())
            self.assertEqual(count_processes("[f]ake_engine.py engine") + count_processes("[f]ake_llama_server.py"), 1)
        QTest.mouseClick(w.eject_button, Qt.MouseButton.LeftButton)
        self.assertEqual(w.model_button.label(), "No Model Loaded, Pick one to use")
        self.assertEqual(count_processes("[f]ake_engine.py engine") + count_processes("[f]ake_llama_server.py"), 0)

    def test_chatting_uses_the_loaded_model_and_closing_the_window_frees_it(self):
        w = self.window(); w._load_model(self.gguf[0]["path"]); self.idle(w)
        w.composer.setPlainText("hello"); w._send_chat(); self.idle(w)
        self.assertEqual(w.current_chat["messages"][-1]["status"], "complete")
        self.assertEqual(w.current_chat["model"], self.gguf[0]["path"])
        self.assertEqual(count_processes("[f]ake_llama_server.py"), 1)
        w.close(); time.sleep(0.3)
        self.assertEqual(count_processes("[f]ake_llama_server.py"), 0)

    def test_opening_another_chat_never_swaps_the_loaded_model(self):
        w = self.window(); w._load_model(self.trance[0]["path"]); self.idle(w)
        old = self.chat_with(w); old["model"] = self.gguf[0]["path"]; w.chats.append(old); w._refresh_chats()
        w._select_chat(0)
        self.assertEqual((w.active_model, w.model_button.label(), w._busy), (self.trance[0]["path"], "Trance 1.1", False))


class DefaultModelTests(WindowCase):
    def test_settings_offers_every_model_and_remembers_the_choice(self):
        w = self.window()
        card = w.settings_page.default_model_card
        self.assertTrue(card.combo.itemText(0).startswith("None"))
        self.assertEqual(card.combo.count(), 1 + len(w.models))
        self.assertEqual(card.combo.currentData(), "")
        index = card.combo.findData(self.trance[1]["path"]); card.combo.setCurrentIndex(index); card.combo.activated.emit(index)
        self.assertEqual(read_json(self.state / "settings.json")["default_model"], self.trance[1]["path"])
        card.combo.setCurrentIndex(0); card.combo.activated.emit(0)
        self.assertEqual(read_json(self.state / "settings.json")["default_model"], "")

    def test_the_default_model_loads_by_itself_once_the_window_is_up(self):
        with patch("gui.engine.EngineClient.load") as load:
            w = self.window({"default_model": self.trance[1]["path"]})
            self.assertEqual(w.active_model, "")                      # the window appears first
            QTest.qWait(50)
            load.assert_called_once_with(self.trance[1]["path"])
        self.assertEqual(w.settings_page.default_model_card.combo.currentData(), self.trance[1]["path"])

    def test_no_default_means_nothing_loads_at_launch(self):
        with patch("gui.engine.EngineClient.load") as load, patch("gui.backends.llama_server.LlamaServerBackend.load") as llama:
            w = self.window({"default_model": ""}); QTest.qWait(50)
            load.assert_not_called(); llama.assert_not_called()
        self.assertEqual(w.model_button.label(), "No Model Loaded, Pick one to use")

    def test_a_default_that_has_gone_is_reported_and_kept(self):
        gone = str(Path(self.temp.name) / "models" / "deleted.bin")
        with patch("gui.engine.EngineClient.load") as load:
            w = self.window({"default_model": gone}); QTest.qWait(50)
            load.assert_not_called()
        self.assertIn("Default model not found: deleted.bin", w.statusBar().currentMessage())
        self.assertIn("not found", w.settings_page.default_model_card.combo.currentText())
        self.assertEqual(read_json(self.state / "settings.json").get("default_model"), gone)

    def test_a_default_gguf_without_the_runtime_explains_instead_of_failing_silently(self):
        with patch("gui.runtimes.RuntimeManager.resolve", return_value=None):
            w = self.window({"default_model": self.gguf[0]["path"]}); QTest.qWait(50)
        self.assertIn("llama.cpp runtime", w.statusBar().currentMessage()); self.assertEqual(w.active_model, "")
        self.assertFalse(w.runtime_prompt.isVisible())                    # a launch must not greet the person with a dialog

    def test_the_default_is_not_loaded_onto_a_gpu_that_is_training(self):
        with patch("gui.engine.EngineClient.load") as load:
            w = self.window({"default_model": self.trance[0]["path"]}); w.training_active = True
            w._load_default_model(); load.assert_not_called()


class ChatActionTests(WindowCase):
    def setUp(self):
        super().setUp()
        self.w = self.window()
        self.w.engine.loaded_model = self.trance[0]["path"]; self.w._active_model_changed()

    def test_buttons_appear_only_when_they_mean_something(self):
        w = self.w
        self.assertEqual((w.chat_menu_button.isVisibleTo(w), w.save_chat_button.isVisibleTo(w)), (False, False))      # empty new chat
        w._new_chat(temporary=True)
        self.assertEqual((w.chat_menu_button.isVisibleTo(w), w.save_chat_button.isVisibleTo(w)), (False, False))      # empty temporary chat
        self.show_chat(w, self.chat_with(w, temporary=True, title="Temporary chat"))
        self.assertEqual((w.chat_menu_button.isVisibleTo(w), w.save_chat_button.isVisibleTo(w)), (True, True))
        self.show_chat(w, self.chat_with(w))
        self.assertEqual((w.chat_menu_button.isVisibleTo(w), w.save_chat_button.isVisibleTo(w)), (True, False))       # saved chat: no Save
        self.assertEqual((w.chat_menu_button.text(), w.chat_menu_button.toolTip()), ("\u22ef", "Chat options"))
        self.assertFalse(hasattr(w, "chat_export_button"))

    def test_save_chat_is_left_of_the_three_dots_in_the_top_row(self):
        w = self.w; self.show_chat(w, self.chat_with(w, temporary=True, title="Temporary chat")); self.app.processEvents()
        x = lambda widget: widget.mapTo(w, widget.rect().topLeft()).x()
        y = lambda widget: widget.mapTo(w, widget.rect().center()).y()
        self.assertLess(x(w.save_chat_button), x(w.chat_menu_button))
        self.assertLess(x(w.chat_title), x(w.model_button)); self.assertLess(x(w.model_button), x(w.save_chat_button))
        self.assertLess(abs(y(w.save_chat_button) - y(w.chat_title)), 30); self.assertLess(abs(y(w.chat_menu_button) - y(w.chat_title)), 30)

    def test_save_chat_turns_a_temporary_chat_into_a_normal_one(self):
        w = self.w
        chat = self.chat_with(w, (("Draft a polite apology", "Dear Sam, I am sorry."), ("shorter", "Sorry, Sam.")), temporary=True, title="Temporary chat")
        self.show_chat(w, chat)
        w.composer.setPlainText("an unsent draft")
        self.assertEqual(w.chat_title.text(), "Temporary Chat"); self.assertTrue(w.temporary_chat_button.isChecked())
        QTest.mouseClick(w.save_chat_button, Qt.MouseButton.LeftButton)
        self.assertNotIn("temporary", chat)
        self.assertEqual([c["id"] for c in w.chats], [chat["id"]])
        self.assertEqual(chat["title"], "Draft a polite apology")                             # named from the first message, like any chat
        self.assertEqual(w.chat_list.count(), 1); self.assertEqual(w.chat_list.currentRow(), 0)
        saved = read_json(self.state / "chats.json")
        self.assertEqual((len(saved), len(saved[0]["messages"]), "temporary" in saved[0]), (1, 4, False))
        self.assertEqual((w.chat_title.text(), w.temporary_chat_button.isChecked()), ("Chat", False))
        self.assertFalse(w.save_chat_button.isVisibleTo(w)); self.assertTrue(w.chat_menu_button.isVisibleTo(w))
        self.assertEqual(w.composer.toPlainText(), "an unsent draft")                        # nothing else was disturbed
        self.assertIs(w.current_chat, chat)
        reopened = MainWindow(state_dir=self.state, binary=Path("/unused-cpu-test-engine")); reopened.timer.stop()
        try: self.assertEqual([c["title"] for c in reopened.chats], ["Draft a polite apology"])
        finally: reopened.close()

    def test_a_renamed_temporary_chat_keeps_its_name_when_saved(self):
        w = self.w; chat = self.chat_with(w, temporary=True, title="Temporary chat"); self.show_chat(w, chat)
        with patch("gui.app.QInputDialog.getText", return_value=("  My   plan ", True)):
            self.choose_from_menu(lambda: w._open_chat_menu(), "Rename")
        self.assertEqual(chat["title"], "My plan"); self.assertEqual(w.chats, [])             # renamed, still not in history
        self.assertEqual(read_json(self.state / "chats.json", []), [])
        QTest.mouseClick(w.save_chat_button, Qt.MouseButton.LeftButton)
        self.assertEqual([c["title"] for c in w.chats], ["My plan"])

    def test_save_chat_that_cannot_write_history_stays_temporary(self):
        w = self.w; chat = self.chat_with(w, temporary=True, title="Temporary chat"); self.show_chat(w, chat)
        with patch.object(w, "_save_chats", return_value=False):
            w._save_temporary_chat()
        self.assertTrue(chat["temporary"]); self.assertEqual((chat["title"], w.chats), ("Temporary chat", []))
        self.assertTrue(w.save_chat_button.isVisibleTo(w))

    def test_save_chat_is_ignored_for_chats_that_are_not_temporary(self):
        w = self.w; chat = self.chat_with(w); w.chats.append(chat); self.show_chat(w, chat)
        w._save_temporary_chat(); self.assertEqual(len(w.chats), 1)

    def test_header_menu_is_the_same_rename_export_delete_as_the_sidebar(self):
        w = self.w; chat = self.chat_with(w); w.chats.append(chat); w._refresh_chats(); self.show_chat(w, chat)
        header = self.choose_from_menu(lambda: w._open_chat_menu(), "")
        sidebar = self.choose_from_menu(lambda: w._chat_context_menu(w.chat_list.visualItemRect(w.chat_list.item(0)).center()), "")
        self.assertEqual([t for t, _ in header], ["Rename", "Export", "Delete"])
        self.assertEqual(header, sidebar)

    def test_rename_and_delete_are_unavailable_but_export_still_works_while_a_reply_is_written(self):
        w = self.w; self.show_chat(w, self.chat_with(w))
        with patch.object(type(w.engine), "busy", property(lambda self: True)):
            shown = self.choose_from_menu(lambda: w._open_chat_menu(), "")
        self.assertEqual(shown, [("Rename", False), ("Export", True), ("Delete", False)])

    def test_header_menu_actions_target_the_open_chat(self):
        w = self.w; chat = self.chat_with(w); w.chats.append(chat); w._refresh_chats(); self.show_chat(w, chat)
        for label, method in (("Rename", "_rename_chat"), ("Export", "_export_chat"), ("Delete", "_delete_chat")):
            with patch.object(w, method) as callback:
                self.choose_from_menu(lambda: QTest.mouseClick(w.chat_menu_button, Qt.MouseButton.LeftButton), label)
                callback.assert_called_once_with(chat["id"])

    def test_delete_from_the_header_removes_a_saved_chat_and_discards_a_temporary_one(self):
        w = self.w; saved = self.chat_with(w, (("keep me?", "no"),)); w.chats.append(saved); w._save_chats(); w._refresh_chats(); self.show_chat(w, saved)
        self.choose_from_menu(lambda: w._open_chat_menu(), "Delete")
        self.assertEqual((w.chats, read_json(self.state / "chats.json")), ([], []))
        self.assertEqual(w.chat_title.text(), "New Chat")
        temporary = self.chat_with(w, temporary=True, title="Temporary chat"); self.show_chat(w, temporary)
        self.choose_from_menu(lambda: w._open_chat_menu(), "Delete")
        self.assertNotEqual(w.current_chat["id"], temporary["id"]); self.assertEqual(w.chat_title.text(), "New Chat")
        self.assertEqual(w.chats, [])

    def test_export_from_the_sidebar_exports_the_clicked_chat_not_the_open_one(self):
        w = self.w
        other = self.chat_with(w, (("the other one", "yes"),)); other["title"] = "Other: chat/1?"; w.chats.append(other)
        open_chat = self.chat_with(w, (("the open one", "yes"),)); w.chats.append(open_chat); w._refresh_chats(); self.show_chat(w, open_chat)
        target = Path(self.temp.name) / "out.json"
        with patch("gui.app.QFileDialog.getSaveFileName", return_value=(str(target), "JSON")) as dialog:
            self.choose_from_menu(lambda: w._chat_context_menu(w.chat_list.visualItemRect(w.chat_list.item(0)).center()), "Export")
        self.assertEqual(read_json(target)["id"], other["id"])
        self.assertEqual(dialog.call_args.args[2], "Other- chat-1.json")                      # a safe file name from the title
        self.assertEqual(w.current_chat["id"], open_chat["id"])

    def test_every_menu_wording_is_the_same_everywhere(self):
        source = Path(__file__).resolve().parents[1].joinpath("gui", "app.py").read_text()
        for old in ("Rename chat\u2026", "Delete chat", "Export chat\u2026"):
            self.assertNotIn(f'"{old}"', source)


class RuntimePromptTests(WindowCase):
    """Picking a GGUF model while no llama.cpp runtime is installed."""

    def setUp(self):
        super().setUp()
        self.w = self.window()
        self.w.runtimes._path_lookup = lambda name: None                      # a llama-server on this machine's PATH must not leak in
        self.w.runtimes.bundled_dir = Path(self.temp.name) / "no-bundled-runtime"
        self.flavor = Flavor("vulkan", "Vulkan (NVIDIA, AMD, Intel)", "d", (Asset("x.tar.gz", "0" * 64),), 36)
        self.w.runtimes.flavors = [self.flavor]
        self.model = self.gguf[0]["path"]

    def pick(self, path=None):
        with patch.object(self.w.engine, "load") as load:
            self.w._open_model_picker()
            QTest.mouseClick(self.w.model_picker.list.viewport(), Qt.MouseButton.LeftButton, pos=self.row(self.w, path or self.model))
            self.assertFalse(load.called)
        self.app.processEvents()

    def test_picking_a_gguf_opens_a_popup_that_offers_the_download_and_loads_nothing(self):
        w = self.w; self.assertIsNone(w.runtimes.resolve())
        self.pick()
        prompt = w.runtime_prompt
        self.assertTrue(prompt.isVisible()); self.assertEqual(w._runtime_model, self.model)
        self.assertIn("Qwen Tiny", prompt.body.text()); self.assertIn("GGUF", prompt.body.text())
        self.assertIn("Vulkan (NVIDIA, AMD, Intel)", prompt.detail.text()); self.assertIn("github.com", prompt.detail.text())
        self.assertEqual(prompt.install_button.text(), "Download (36 MB)")
        self.assertTrue(prompt.runtimes_button.isVisibleTo(prompt) and prompt.cancel_button.isVisibleTo(prompt))
        self.assertEqual((w.model_button.label(), w._busy, w.operation), ("No Model Loaded, Pick one to use", False, None))
        self.assertIn("llama.cpp runtime", w.statusBar().currentMessage())                # the status line stays as a second cue

    def test_trance_models_never_open_the_popup(self):
        w = self.w
        with patch.object(w.engine, "load") as load:
            w._load_model(self.trance[0]["path"]); load.assert_called_once()
        self.assertFalse(w.runtime_prompt.isVisible())

    def test_cancel_escape_and_the_backdrop_close_it_and_forget_the_model(self):
        w = self.w; prompt = w.runtime_prompt
        for close in (lambda: QTest.mouseClick(prompt.cancel_button, Qt.MouseButton.LeftButton),
                      lambda: QTest.keyClick(prompt.cancel_button, Qt.Key.Key_Escape),
                      lambda: QTest.mouseClick(prompt, Qt.MouseButton.LeftButton, pos=QPoint(3, 3))):
            self.pick(); self.assertTrue(prompt.isVisible()); close()
            self.assertFalse(prompt.isVisible()); self.assertEqual(w._runtime_model, "")
        self.assertFalse(w.runtimes.installing)

    def test_open_runtimes_goes_to_the_runtimes_page_and_closes_the_popup(self):
        w = self.w; self.pick()
        QTest.mouseClick(w.runtime_prompt.runtimes_button, Qt.MouseButton.LeftButton)
        self.assertFalse(w.runtime_prompt.isVisible()); self.assertIs(w.pages.currentWidget(), w.runtimes_page)

    def test_without_a_prebuilt_runtime_only_the_runtimes_page_is_offered(self):
        w = self.w; w.runtimes.flavors = []; self.pick()
        prompt = w.runtime_prompt
        self.assertFalse(prompt.install_button.isVisibleTo(prompt)); self.assertIn("No prebuilt runtime", prompt.detail.text())
        self.assertEqual(prompt.runtimes_button.objectName(), "primary")

    def test_download_starts_the_recommended_runtime_and_cannot_be_closed_by_accident(self):
        w = self.w; self.pick(); prompt = w.runtime_prompt
        with patch.object(w.runtimes, "start_install") as start, patch.object(w.runtimes, "cancel_install") as cancel:
            QTest.mouseClick(prompt.install_button, Qt.MouseButton.LeftButton)
            start.assert_called_once_with("vulkan")
            self.assertTrue(prompt.installing); self.assertTrue(prompt.progress.isVisibleTo(prompt))
            self.assertFalse(prompt.install_button.isVisibleTo(prompt)); self.assertEqual(prompt.cancel_button.text(), "Cancel download")
            self.assertIn("Qwen Tiny", prompt.body.text())
            QTest.keyClick(prompt.cancel_button, Qt.Key.Key_Escape); QTest.mouseClick(prompt, Qt.MouseButton.LeftButton, pos=QPoint(3, 3))
            self.assertTrue(prompt.isVisible())                                          # neither abandons the download
            w.runtimes.progress.emit(24_000_000, 36_000_000, "Downloading 1/1")
            self.assertEqual((prompt.progress.value(), prompt.progress.maximum()), (24_000_000, 36_000_000))
            self.assertIn("24 / 36 MB", prompt.progress.format())
            QTest.mouseClick(prompt.cancel_button, Qt.MouseButton.LeftButton)
            cancel.assert_called_once()
        self.assertFalse(prompt.isVisible()); self.assertEqual(w._runtime_model, "")

    def test_a_failed_download_says_why_and_can_be_retried_but_a_cancel_is_quiet(self):
        w = self.w; self.pick(); prompt = w.runtime_prompt
        with patch.object(w.runtimes, "start_install"):
            QTest.mouseClick(prompt.install_button, Qt.MouseButton.LeftButton)
        w.runtimes.install_failed.emit("Could not reach github.com (timed out).")
        self.assertTrue(prompt.error.isVisibleTo(prompt)); self.assertIn("Could not reach", prompt.error.text())
        self.assertEqual(prompt.install_button.text(), "Try again"); self.assertFalse(prompt.installing)
        self.assertTrue(prompt.isVisible()); self.assertFalse(prompt.progress.isVisibleTo(prompt))
        QTest.keyClick(prompt.cancel_button, Qt.Key.Key_Escape); self.assertFalse(prompt.isVisible())      # closable again
        w.runtimes.install_failed.emit("late error after closing"); self.assertFalse(prompt.isVisible())    # nothing pops back up
        self.pick()
        with patch.object(w.runtimes, "start_install"):
            QTest.mouseClick(prompt.install_button, Qt.MouseButton.LeftButton)
        w.runtimes.install_failed.emit("")                                                 # what a cancelled download reports
        self.assertFalse(prompt.error.isVisibleTo(prompt)); self.assertTrue(prompt.installing)

    def test_a_download_already_running_from_settings_is_joined(self):
        w = self.w
        with patch.object(type(w.runtimes), "installing", property(lambda manager: True)):
            self.pick()
        self.assertTrue(w.runtime_prompt.installing); self.assertFalse(w.runtime_prompt.install_button.isVisibleTo(w.runtime_prompt))

    def test_the_picker_tags_gguf_rows_only_while_no_runtime_is_installed(self):
        w = self.w
        flags = lambda: {Path(e["path"]).suffix: e.get("needs_runtime") for e in w._model_entries()}
        self.assertEqual(flags(), {".bin": None, ".gguf": True})                           # Trance rows never carry it
        with patch.object(w.runtimes, "resolve", return_value=RUNTIME):
            self.assertEqual(flags(), {".bin": None, ".gguf": False})
        w._open_model_picker()
        self.assertEqual(sum(bool(w.model_picker.list.item(i).data(Qt.ItemDataRole.UserRole + 3)) for i in range(w.model_picker.list.count())), 2)

    def test_the_overlays_follow_the_window_size(self):
        w = self.w; w.resize(1400, 800); self.app.processEvents()
        self.assertEqual(w.runtime_prompt.geometry(), w.rect()); self.assertEqual(w.model_picker.geometry(), w.rect())


class RuntimeInstallFlowTests(WindowCase):
    """The whole path with a real RuntimeManager: popup -> Download -> installed -> the picked model loads by itself."""

    def setUp(self):
        super().setUp()
        self.www = Path(self.temp.name) / "www" / "bTEST"; self.www.mkdir(parents=True)
        make_tar(self.www / "runtime.tar.gz", {SERVER_NAME: FAKE_SERVER.read_bytes(), "fake_openai.py": (HERE / "fake_openai.py").read_bytes()})
        digest = hashlib.sha256((self.www / "runtime.tar.gz").read_bytes()).hexdigest()
        self.files = FileServer(Path(self.temp.name) / "www"); self.addCleanup(self.files.stop)
        self.w = self.window(binary=FAKE_ENGINE)
        runtimes = self.w.runtimes
        runtimes._path_lookup = lambda name: None; runtimes.bundled_dir = Path(self.temp.name) / "no-bundled-runtime"
        runtimes.flavors = [Flavor("vulkan", "Vulkan", "d", (Asset("runtime.tar.gz", digest),), 1)]
        runtimes.tag, runtimes.base_url = "bTEST", self.files.url

    def test_download_then_the_model_loads_without_another_click(self):
        w = self.w; self.assertIsNone(w.runtimes.resolve())
        w._open_model_picker()
        QTest.mouseClick(w.model_picker.list.viewport(), Qt.MouseButton.LeftButton, pos=self.row(w, self.gguf[0]["path"]))
        self.assertTrue(w.runtime_prompt.isVisible())
        QTest.mouseClick(w.runtime_prompt.install_button, Qt.MouseButton.LeftButton)
        self.assertTrue(pump(self.app, lambda: not w.runtime_prompt.isVisible() and not w.engine.busy and w.active_model, 40))
        self.assertEqual((w.model_button.label(), w.active_model), ("Qwen Tiny", self.gguf[0]["path"]))
        self.assertEqual(w.runtimes.resolve().source, "managed")
        installed = self.state / "runtimes" / "llama.cpp" / "bTEST-vulkan"                  # inside the project's data folder
        self.assertTrue((installed / SERVER_NAME).is_file() and (installed / "runtime.json").is_file())
        self.assertFalse(any(e.get("needs_runtime") for e in w._model_entries()))
        self.assertEqual(w._runtime_model, ""); self.assertFalse(w._busy)
        self.assertEqual(count_processes("[b]TEST-vulkan/llama-server"), 1)                 # the downloaded runtime is the one running

    def test_a_download_that_does_not_match_its_hash_is_refused_and_nothing_loads(self):
        w = self.w
        w.runtimes.flavors = [Flavor("vulkan", "Vulkan", "d", (Asset("runtime.tar.gz", "f" * 64),), 1)]
        w._open_model_picker()
        QTest.mouseClick(w.model_picker.list.viewport(), Qt.MouseButton.LeftButton, pos=self.row(w, self.gguf[0]["path"]))
        QTest.mouseClick(w.runtime_prompt.install_button, Qt.MouseButton.LeftButton)
        self.assertTrue(pump(self.app, lambda: w.runtime_prompt.error.isVisibleTo(w.runtime_prompt), 30))
        self.assertTrue(w.runtime_prompt.isVisible()); self.assertEqual(w.active_model, "")
        self.assertIsNone(w.runtimes.resolve()); self.assertFalse((self.state / "runtimes" / "llama.cpp" / "bTEST-vulkan").exists())


class EngineCardTests(WindowCase):
    def test_the_trance_engine_is_listed_first_among_the_runtimes_and_is_read_only(self):
        w = self.window(); page = w.runtimes_page; card = page.engine_card
        layout = card.parentWidget().layout()
        self.assertLess(layout.indexOf(card), layout.indexOf(page.runtime_card))
        self.assertEqual(card.findChildren(__import__("PySide6.QtWidgets", fromlist=["QPushButton"]).QPushButton), [])
        self.assertEqual(card.findChild(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel, "badge").text(), "Built in")

    def test_it_reports_whether_the_binary_is_found_where_it_is_and_the_gpu_once_known(self):
        engine = Path(self.temp.name) / "trance"
        w = self.window(binary=engine)
        card = w.runtimes_page.engine_card
        self.assertIn("Not found", card.status.text()); self.assertEqual(card.gpu.text(), "Unavailable")
        engine.write_bytes(b"x"); card.refresh()
        self.assertEqual(card.status.text(), "Ready to use"); self.assertEqual(card.location.full_text(), str(engine))
        self.assertEqual(card.gpu.text(), card.GPU_UNKNOWN)
        w.engine.trance.ready.emit("NVIDIA GeForce RTX 4060 (compute capability 8.9)")
        self.assertEqual(card.gpu.text(), "NVIDIA GeForce RTX 4060")

    def test_a_bundled_engine_says_so_and_a_long_path_is_elided_with_the_full_one_in_the_tooltip(self):
        from gui.settings import EngineCard
        deep = Path(self.temp.name, *["a-very-long-folder-name"] * 6, "trance"); deep.parent.mkdir(parents=True); deep.write_bytes(b"x")
        card = EngineCard(deep, bundled=True); card.resize(420, 200); card.show(); self.app.processEvents()
        self.assertEqual(card.location.full_text(), "Bundled with the app"); self.assertEqual(card.location.toolTip(), str(deep))
        card.bundled = False; card.refresh(); card.resize(300, 200); self.app.processEvents()
        self.assertEqual(card.location.full_text(), str(deep)); self.assertLess(len(card.location.text()), len(str(deep)))
        self.assertIn("\u2026", card.location.text())


if __name__ == "__main__":
    unittest.main()
