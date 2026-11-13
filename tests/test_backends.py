"""Backend tests that need no GPU: a fake OpenAI server, a fake llama-server process, and the router.

One opt-in test drives the real llama-server: set TRANCE_TEST_LLAMA_SERVER to its path
(requires the `gguf` and `numpy` packages to build a tiny model).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_openai  # noqa: E402

from gui.backends.llama_server import NO_RUNTIME, LlamaServerBackend  # noqa: E402
from gui.backends.openai_stream import ChatStream, build_payload  # noqa: E402
from gui.backends.router import BackendRouter, backend_kind  # noqa: E402
from gui.runtimes import RuntimeInfo  # noqa: E402

FAKE_SERVER = Path(__file__).resolve().parent / "fake_llama_server.py"
SETTINGS = {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "max_new_tokens": 50, "seed": 11}


def pump(app, condition, timeout=15.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents(); time.sleep(0.005)
    return condition()


class Recorder:
    """Collects every signal from a backend or stream in order."""
    def __init__(self, source, *names):
        self.events, self.text = [], []
        for name in names:
            getattr(source, name).connect(lambda *a, n=name: self._on(n, *a))

    def _on(self, name, *args):
        if name == "delta": self.text.append(args[0])
        self.events.append((name, args[0] if args else None))

    def names(self):
        return [n for n, _ in self.events if n != "delta"]

    def payload(self, name):
        return next((v for n, v in self.events if n == name), None)


class AppCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class ChatStreamTests(AppCase):
    def setUp(self):
        self.server, self.port = fake_openai.start(token="secret")
        self.url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        self.auth = {"Authorization": "Bearer secret"}

    def tearDown(self):
        self.server.shutdown()

    def run_stream(self, payload, headers=None, url=None):
        stream = ChatStream(url or self.url, payload, self.auth if headers is None else headers)
        rec = Recorder(stream, "started_stream", "delta", "done", "failed")
        stream.start()
        self.assertTrue(pump(self.app, lambda: stream.isFinished()), "stream never finished")
        self.app.processEvents()
        return rec

    def test_streams_text_and_reports_usage(self):
        rec = self.run_stream(build_payload("fake-model", [{"role": "user", "content": "hi"}], SETTINGS))
        self.assertEqual("".join(rec.text), "Hello from the fake server.")
        self.assertEqual(rec.names(), ["started_stream", "done"])
        done = rec.payload("done")
        self.assertEqual((done["reason"], done["generated_tokens"]), ("stop", 6))
        self.assertIsNone(done["prompt_tokens"])
        sent = self.server.RequestHandlerClass.seen[0]
        self.assertEqual(sent["headers"]["Authorization"], "Bearer secret")
        body = sent["body"]
        self.assertEqual((body["temperature"], body["top_p"], body["top_k"], body["max_tokens"], body["seed"]), (0.7, 0.9, 30, 50, 11))
        self.assertTrue(body["stream"])

    def test_length_limit_is_reported(self):
        rec = self.run_stream(build_payload("fake-model", [], {"max_new_tokens": 3}))
        self.assertEqual(rec.payload("done")["reason"], "length")
        self.assertEqual("".join(rec.text), "Hello from the")

    def test_server_errors_become_readable_failures(self):
        rec = self.run_stream(build_payload("boom", [], SETTINGS))
        self.assertEqual(rec.names(), ["failed"])
        self.assertIn("context too long", rec.payload("failed"))

    def test_wrong_api_key_is_rejected(self):
        rec = self.run_stream(build_payload("fake-model", [], SETTINGS), headers={})
        self.assertIn("Invalid API Key", rec.payload("failed"))

    def test_connection_refused_is_explained(self):
        self.server.shutdown(); self.server.server_close()
        rec = self.run_stream(build_payload("fake-model", [], SETTINGS))
        self.assertIn("Could not connect", rec.payload("failed"))
        self.server, self.port = fake_openai.start()   # for tearDown

    def test_cancel_stops_a_long_stream_quickly(self):
        self.server.RequestHandlerClass.delay = 0.02
        stream = ChatStream(self.url, build_payload("fake-model", [], {"max_new_tokens": 100000}), self.auth)
        rec = Recorder(stream, "delta", "done", "failed")
        stream.start()
        self.assertTrue(pump(self.app, lambda: len(rec.text) >= 5))
        t0 = time.monotonic(); stream.cancel()
        self.assertTrue(pump(self.app, lambda: stream.isFinished(), 5))
        self.app.processEvents()
        self.assertLess(time.monotonic() - t0, 2)
        self.assertEqual(rec.payload("done")["reason"], "cancelled")
        self.assertIsNone(rec.payload("failed"))


@unittest.skipIf(os.name == "nt", "the fake llama-server is a POSIX script")
class LlamaServerBackendTests(AppCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "tiny.gguf"; self.model.write_bytes(b"GGUF-not-really")
        self.other = Path(self.tmp.name) / "other.gguf"; self.other.write_bytes(b"GGUF-other")
        self.bad = Path(self.tmp.name) / "bad.gguf"; self.bad.write_bytes(b"BAD model")
        self.runtime = RuntimeInfo(FAKE_SERVER, "custom", "b4242", "fake")
        self.backend = LlamaServerBackend(lambda: self.runtime, load_timeout=20)
        self.rec = Recorder(self.backend, "delta", "started", "finished", "failed", "loaded", "ready", "state_changed")

    def tearDown(self):
        self.backend.close(); self.tmp.cleanup()

    def ask(self, model=None, settings=SETTINGS):
        self.rec.events.clear(); self.rec.text.clear()
        self.backend.generate(str(model or self.model), [{"role": "user", "content": "hi"}], settings)
        self.assertTrue(pump(self.app, lambda: not self.backend.busy), "backend stayed busy")
        self.app.processEvents()

    def test_loads_streams_and_reuses_the_server(self):
        self.ask()
        self.assertEqual(self.rec.names(), ["state_changed", "loaded", "ready", "state_changed", "state_changed", "started", "finished"])
        self.assertEqual("".join(self.rec.text), "Hello from the fake server.")
        self.assertEqual(self.rec.payload("state_changed"), "Loading model\u2026")
        self.assertEqual(self.rec.payload("ready"), "llama.cpp b4242 \u00b7 fake")
        self.assertEqual(self.rec.payload("loaded")["model"], str(self.model.resolve()))
        port = self.backend._port
        self.ask()                                   # second message: no reload
        self.assertNotIn("loaded", self.rec.names())
        self.assertEqual(self.backend._port, port)
        self.assertEqual(self.rec.payload("finished")["reason"], "stop")
        self.assertEqual(self.rec.payload("finished")["context_length"], 8192)   # the app shows context use from this

    def test_server_gets_loopback_binding_and_an_api_key(self):
        self.ask()
        args = self.backend.process.arguments()
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertTrue(len(args[args.index("--api-key") + 1]) >= 16)
        self.assertIn("--no-webui", args)

    def test_the_server_does_not_inherit_the_apps_bundled_libraries(self):
        from gui import runtime
        bundle = self.tmp.name + "/_internal"
        with patch.object(runtime, "FROZEN", True), patch.object(runtime, "BUNDLE", Path(bundle)), \
             patch.dict(os.environ, {"LD_LIBRARY_PATH": f"{bundle}/cuda:{bundle}:/usr/lib/mine"}):
            self.ask()
        self.assertEqual(self.backend.process.processEnvironment().value("LD_LIBRARY_PATH"), "/usr/lib/mine")

    def test_switching_models_restarts_the_server(self):
        self.ask(); first = self.backend._port
        self.ask(self.other)
        self.assertNotEqual(self.backend._port, first)
        self.assertEqual(self.rec.names().count("loaded"), 1)

    def test_load_failure_shows_the_servers_own_message(self):
        self.ask(self.bad)
        self.assertEqual(self.rec.names()[-1], "failed")
        self.assertIn("unknown model architecture", self.rec.payload("failed"))
        self.assertFalse(self.backend.running)

    def test_missing_runtime_explains_how_to_install_it(self):
        backend = LlamaServerBackend(lambda: None); rec = Recorder(backend, "failed")
        backend.generate(str(self.model), [], SETTINGS)
        self.assertTrue(pump(self.app, lambda: not backend.busy))
        self.assertEqual(rec.payload("failed"), NO_RUNTIME)

    def test_cancel_mid_generation_keeps_the_server_usable(self):
        self.rec.events.clear()
        self.backend.generate(str(self.model), [{"role": "user", "content": "hi"}], {**SETTINGS, "max_new_tokens": 100000})
        self.assertTrue(pump(self.app, lambda: len(self.rec.text) >= 3))
        self.backend.cancel()
        self.assertTrue(pump(self.app, lambda: not self.backend.busy))
        self.app.processEvents()
        self.assertEqual(self.rec.payload("finished")["reason"], "cancelled")
        self.assertTrue(self.backend.running)
        self.ask()
        self.assertEqual(self.rec.payload("finished")["reason"], "stop")

    def test_cancel_while_loading_abandons_the_server(self):
        self.backend.generate(str(self.model), [], SETTINGS)
        self.backend.cancel()
        self.assertFalse(self.backend.busy)
        self.assertEqual(self.rec.payload("finished")["reason"], "cancelled")
        self.assertFalse(self.backend.running)

    def test_release_frees_the_server_and_refuses_while_busy(self):
        self.ask()
        self.assertTrue(self.backend.running)
        self.backend.release()
        self.assertFalse(self.backend.running)
        self.backend.generate(str(self.model), [], SETTINGS)
        with self.assertRaises(RuntimeError): self.backend.release()
        with self.assertRaises(RuntimeError): self.backend.generate(str(self.model), [], SETTINGS)
        self.backend.cancel()

    def test_close_terminates_the_process(self):
        self.ask(); process = self.backend.process
        self.backend.close()
        self.assertEqual(process.state().name, "NotRunning")


class RouterTests(AppCase):
    def setUp(self):
        self.router = BackendRouter(binary=Path("/unused-cpu-test-engine"))

    def tearDown(self):
        self.router.close()

    def test_models_are_routed_by_file_type(self):
        self.assertEqual(backend_kind("/m/Qwen3.Q4_K_M.GGUF"), "gguf")
        self.assertEqual(backend_kind("/m/trance1.2.bin"), "trance")
        with patch.object(self.router.trance, "generate") as t, patch.object(self.router.llama, "generate") as g, \
             patch.object(self.router.trance, "release") as tr, patch.object(self.router.llama, "release") as gr:
            self.router.generate("/m/a.gguf", [], SETTINGS)
            g.assert_called_once(); t.assert_not_called(); tr.assert_called_once()    # the native engine lets go of the GPU
            self.router.generate("/m/a.bin", [], SETTINGS)
            t.assert_called_once(); gr.assert_called_once()

    def test_signals_from_either_backend_reach_the_window(self):
        rec = Recorder(self.router, "delta", "finished", "failed", "loaded")
        self.router.llama.delta.emit("x"); self.router.trance.finished.emit({"reason": "stop"})
        self.router.llama.failed.emit("nope")
        self.assertEqual(rec.text, ["x"]); self.assertEqual(rec.names(), ["finished", "failed"])

    def test_busy_is_shared_and_blocks_new_requests(self):
        with patch.object(type(self.router.llama), "busy", property(lambda self: True)):
            self.assertTrue(self.router.busy)
            with self.assertRaises(RuntimeError): self.router.generate("/m/a.bin", [], SETTINGS)
            with self.assertRaises(RuntimeError): self.router.release()


@unittest.skipUnless(os.environ.get("TRANCE_TEST_LLAMA_SERVER"), "set TRANCE_TEST_LLAMA_SERVER to run against the real llama-server")
class RealLlamaServerTests(AppCase):
    def test_real_binary_streams_from_a_tiny_gguf(self):
        try:
            from tiny_gguf import build
            model = build(Path(tempfile.mkdtemp()) / "tiny.gguf")
        except ImportError:
            self.skipTest("needs the gguf and numpy packages")
        backend = LlamaServerBackend(lambda: RuntimeInfo(Path(os.environ["TRANCE_TEST_LLAMA_SERVER"]), "custom"))
        rec = Recorder(backend, "delta", "finished", "failed", "loaded")
        try:
            backend.generate(str(model), [{"role": "user", "content": "hello"}], {**SETTINGS, "max_new_tokens": 12})
            self.assertTrue(pump(self.app, lambda: not backend.busy, 60))
            self.app.processEvents()
            self.assertEqual(rec.names(), ["loaded", "finished"], rec.events)
            self.assertGreater(len(rec.text), 3)
            self.assertEqual(rec.payload("finished")["reason"], "length")
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
