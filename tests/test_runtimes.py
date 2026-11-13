"""Runtime catalog, safe extraction, installation and resolution (no real downloads)."""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from gui import runtimes
from gui.runtimes import (FLAVORS, PINNED_TAG, SERVER_NAME, Asset, Flavor, InstallError, RuntimeManager, find_server,
                          probe_version, recommend, safe_extract)

FAKE_SERVER = Path(__file__).resolve().parent / "fake_llama_server.py"


def pump(app, condition, timeout=15.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents(); time.sleep(0.005)
    return condition()


def make_tar(path: Path, files: dict, *, top="llama-bTEST", links: dict | None = None):
    with tarfile.open(path, "w:gz") as tar:
        if top:
            d = tarfile.TarInfo(top); d.type = tarfile.DIRTYPE; d.mode = 0o755; tar.addfile(d)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name); info.size = len(data); info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name); info.type = tarfile.SYMTYPE; info.linkname = target
            tar.addfile(info)


def make_zip(path: Path, files: dict, *, top=""):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(f"{top}/{name}" if top else name, data)


class CatalogTests(unittest.TestCase):
    def test_every_pinned_asset_is_well_formed(self):
        self.assertRegex(PINNED_TAG, r"^b\d+$")
        for (system, arch), flavors in FLAVORS.items():
            ids = [f.id for f in flavors]
            self.assertEqual(len(ids), len(set(ids)), (system, arch))
            for flavor in flavors:
                self.assertGreaterEqual(len(flavor.assets), 1)
                for asset in flavor.assets:
                    self.assertRegex(asset.sha256, r"^[0-9a-f]{64}$", asset.name)
                    self.assertTrue(asset.name.endswith(".zip" if system == "windows" else ".tar.gz"), asset.name)
                    if asset.name.startswith("llama-"): self.assertIn(PINNED_TAG, asset.name)
                if flavor.id.startswith("cuda"):
                    self.assertEqual(sum(a.name.startswith("cudart-") for a in flavor.assets), 1, "CUDA builds ship their runtime libraries")

    def test_platform_detection(self):
        cases = {("linux", "x86_64"): ("linux", "x64"), ("darwin", "arm64"): ("macos", "arm64"),
                 ("win32", "AMD64"): ("windows", "x64"), ("linux", "aarch64"): ("linux", "arm64")}
        for (sys_platform, machine), expected in cases.items():
            with patch.object(sys, "platform", sys_platform), patch("platform.machine", return_value=machine):
                self.assertEqual(runtimes.current_platform(), expected)

    def test_recommendation_prefers_the_best_available_flavor(self):
        linux = FLAVORS[("linux", "x64")]
        self.assertEqual(recommend(linux, nvidia=True, vulkan=True), "cuda12")
        self.assertEqual(recommend(linux, nvidia=False, vulkan=True), "vulkan")
        self.assertEqual(recommend(linux, nvidia=False, vulkan=False), "cpu")
        self.assertEqual(recommend(FLAVORS[("macos", "arm64")], nvidia=False, vulkan=False), "metal")
        self.assertEqual(recommend(FLAVORS[("macos", "x64")], nvidia=False, vulkan=False), "cpu")


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tar_strips_the_shared_top_folder_and_keeps_links(self):
        make_tar(self.root / "a.tar.gz", {SERVER_NAME: b"x", "libllama.so.0": b"lib"}, links={"libllama.so": "libllama.so.0"})
        safe_extract(self.root / "a.tar.gz", self.root / "out")
        self.assertTrue((self.root / "out" / SERVER_NAME).is_file())
        self.assertTrue((self.root / "out" / "libllama.so").is_symlink())
        self.assertEqual((self.root / "out" / "libllama.so").read_bytes(), b"lib")

    def test_zip_with_files_at_the_root_is_left_alone_and_archives_merge(self):
        make_zip(self.root / "main.zip", {"llama-server.exe": b"exe"}, top="")
        make_zip(self.root / "cudart.zip", {"cudart64_12.dll": b"dll"}, top="")
        for name in ("main.zip", "cudart.zip"): safe_extract(self.root / name, self.root / "out")
        self.assertEqual(sorted(p.name for p in (self.root / "out").iterdir()), ["cudart64_12.dll", "llama-server.exe"])

    def test_a_cuda_runtime_lands_next_to_the_server_even_if_only_one_archive_has_a_top_folder(self):
        make_zip(self.root / "main.zip", {"llama-server.exe": b"exe"}, top="llama-bTEST")
        make_zip(self.root / "cudart.zip", {"cudart64_12.dll": b"dll"}, top="")
        for name in ("main.zip", "cudart.zip"): safe_extract(self.root / name, self.root / "out")
        self.assertEqual((self.root / "out" / "llama-server.exe").read_bytes(), b"exe")
        self.assertTrue((self.root / "out" / "cudart64_12.dll").is_file())

    def test_path_traversal_is_refused(self):
        with tarfile.open(self.root / "evil.tar.gz", "w:gz") as tar:
            info = tarfile.TarInfo("top/../../escape.txt"); info.size = 1; tar.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(InstallError): safe_extract(self.root / "evil.tar.gz", self.root / "out")
        make_zip(self.root / "evil.zip", {"../escape.txt": b"x"})
        with self.assertRaises(InstallError): safe_extract(self.root / "evil.zip", self.root / "out2")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_links_that_point_outside_are_refused(self):
        make_tar(self.root / "link.tar.gz", {"a": b"x"}, links={"bad": "../../../etc/passwd"})
        with self.assertRaises(InstallError): safe_extract(self.root / "link.tar.gz", self.root / "out")

    def test_unknown_archive_types_are_refused(self):
        (self.root / "x.rar").write_bytes(b"")
        with self.assertRaises(InstallError): safe_extract(self.root / "x.rar", self.root / "out")


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):   # clients that cancel mid-download are expected
        pass


class FileServer:
    def __init__(self, directory: Path):
        handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
        handler.log_message = lambda *a: None
        self.server = _QuietServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown(); self.server.server_close()


@unittest.skipIf(os.name == "nt", "builds POSIX tar.gz fixtures")
class InstallAndResolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.www = self.root / "www" / "bTEST"; self.www.mkdir(parents=True)
        script = FAKE_SERVER.read_bytes()
        make_tar(self.www / "runtime.tar.gz", {SERVER_NAME: script, "libx.so": b"lib"})
        make_tar(self.www / "extra.tar.gz", {"libcudart.so": b"cuda"})
        sha = lambda name: hashlib.sha256((self.www / name).read_bytes()).hexdigest()
        self.flavor = Flavor("cpu", "CPU", "detail", (Asset("runtime.tar.gz", sha("runtime.tar.gz")), Asset("extra.tar.gz", sha("extra.tar.gz"))), 1)
        self.files = FileServer(self.root / "www")
        self.custom = {"path": ""}
        self.found_on_path = {"path": None}
        self.manager = RuntimeManager(self.root / "state", lambda: self.custom["path"], lambda v: self.custom.__setitem__("path", v),
                                      flavors=[self.flavor], tag="bTEST", base_url=self.files.url, bundled_dir=self.root / "bundled",
                                      path_lookup=lambda name: self.found_on_path["path"])
        self.failures, self.progress = [], []
        self.manager.install_failed.connect(self.failures.append)
        self.manager.progress.connect(lambda *a: self.progress.append(a))

    def tearDown(self):
        self.manager.close(); self.files.stop(); self.tmp.cleanup()

    def install(self):
        self.manager.start_install("cpu")
        self.assertTrue(pump(self.app, lambda: not self.manager.installing), "install never finished")
        self.app.processEvents()

    def test_download_verify_unpack_and_resolve(self):
        self.assertIsNone(self.manager.resolve())
        self.assertIn("Not installed", self.manager.status())
        self.install()
        info = self.manager.resolve()
        self.assertEqual((info.source, info.version, info.flavor), ("managed", "bTEST", "cpu"))
        self.assertTrue(os.access(info.path, os.X_OK))
        self.assertTrue((info.path.parent / "libcudart.so").is_file(), "the extra archive is merged next to the server")
        self.assertEqual(self.failures, [])
        self.assertTrue(any(total > 0 for _done, total, _text in self.progress))
        self.assertEqual(probe_version(info.path), "b4242")        # the installed binary really runs
        self.assertFalse(list((self.root / "state" / "runtimes" / "llama.cpp").glob(".*partial")), "no half-installed leftovers")

    def test_a_corrupted_download_is_rejected_and_nothing_is_installed(self):
        data = bytearray((self.www / "runtime.tar.gz").read_bytes()); data[-5] ^= 0xFF
        (self.www / "runtime.tar.gz").write_bytes(bytes(data))
        self.install()
        self.assertTrue(self.failures and "Checksum mismatch" in self.failures[0], self.failures)
        self.assertIsNone(self.manager.resolve())

    def test_missing_files_and_unreachable_servers_are_explained(self):
        (self.www / "extra.tar.gz").unlink()
        self.install()
        self.assertIn("404", self.failures[-1])
        self.files.stop(); self.failures.clear()
        self.install()
        self.assertIn("Could not reach", self.failures[-1])

    def test_cancel_leaves_no_install_and_no_error(self):
        slow = self.root / "www" / "bTEST" / "runtime.tar.gz"
        slow.write_bytes(slow.read_bytes() + os.urandom(6_000_000))     # big enough to still be downloading
        flavor = Flavor("cpu", "CPU", "d", (Asset("runtime.tar.gz", hashlib.sha256(slow.read_bytes()).hexdigest()),), 1)
        self.manager.flavors = [flavor]
        self.manager.start_install("cpu"); self.manager.cancel_install()
        self.assertTrue(pump(self.app, lambda: not self.manager.installing))
        self.app.processEvents()
        self.assertEqual(self.failures, [])
        self.assertIsNone(self.manager.resolve())

    def test_resolution_order_custom_then_download_then_bundled_then_path(self):
        on_path = self.root / "path-llama-server"; on_path.write_bytes(b"#!/bin/sh\n"); on_path.chmod(0o755)
        self.found_on_path["path"] = str(on_path)
        self.assertEqual(self.manager.resolve().source, "path")
        bundled = self.root / "bundled"; bundled.mkdir(); (bundled / SERVER_NAME).write_bytes(b"x")
        self.assertEqual(self.manager.resolve().source, "bundled")
        self.install()
        self.assertEqual(self.manager.resolve().source, "managed")
        mine = self.root / "mine"; mine.write_bytes(b"#!/bin/sh\n"); mine.chmod(0o755)
        self.manager.set_custom(str(mine))
        self.assertEqual(self.manager.resolve().source, "custom")
        self.assertEqual(self.custom["path"], str(mine))
        self.manager.clear_custom()
        self.assertEqual(self.manager.resolve().source, "managed")
        self.manager.remove_managed()
        self.assertEqual(self.manager.resolve().source, "bundled")

    def test_a_missing_custom_path_is_ignored(self):
        self.custom["path"] = str(self.root / "gone")
        self.assertIsNone(self.manager.resolve())

    def test_installing_a_download_clears_an_old_custom_choice(self):
        mine = self.root / "mine"; mine.write_bytes(b"#!/bin/sh\n"); mine.chmod(0o755)
        self.manager.set_custom(str(mine))
        self.install()
        self.assertEqual(self.custom["path"], "")
        self.assertEqual(self.manager.resolve().source, "managed")


class ExternalEnvironmentTests(unittest.TestCase):
    def test_bundled_library_paths_are_removed_only_when_frozen(self):
        from gui import runtime
        bundle = "/opt/app/_internal"
        env = {"LD_LIBRARY_PATH": f"{bundle}/cuda:{bundle}:/usr/lib/mine:/opt/other", "HOME": "/h"}
        with patch.object(runtime, "FROZEN", False):
            self.assertEqual(runtime.external_environment(env), env)
        with patch.object(runtime, "FROZEN", True), patch.object(runtime, "BUNDLE", Path(bundle)):
            cleaned = runtime.external_environment(env)
            self.assertEqual(cleaned["LD_LIBRARY_PATH"], "/usr/lib/mine:/opt/other")   # the user's own entries survive
            self.assertEqual(cleaned["HOME"], "/h")
            only_ours = runtime.external_environment({"LD_LIBRARY_PATH": f"{bundle}/cuda:{bundle}"})
            self.assertNotIn("LD_LIBRARY_PATH", only_ours)
            self.assertEqual(runtime.external_environment({"LD_LIBRARY_PATH": "/opt/app/_internal_other"})["LD_LIBRARY_PATH"], "/opt/app/_internal_other")
        self.assertEqual(env["LD_LIBRARY_PATH"].count(":"), 3, "the input mapping is not modified")


class VersionProbeTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX script")
    def test_build_number_is_parsed_from_version_output(self):
        self.assertEqual(probe_version(FAKE_SERVER), "b4242")
        self.assertEqual(probe_version(Path("/does/not/exist")), "")


if __name__ == "__main__":
    unittest.main()
