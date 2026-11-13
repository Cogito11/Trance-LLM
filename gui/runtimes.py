"""Find, download and manage the llama.cpp runtime used for GGUF models.

Resolution order (first hit wins): a path the user chose, a runtime this app
downloaded, a runtime bundled with the app (none today, but the slot exists so
bundling later is a packaging change, not a code change), then ``llama-server``
on PATH.

Downloads use a *pinned* llama.cpp build tag and pinned SHA-256 hashes, fetched
straight from the GitHub release download URLs. That avoids the rate-limited
GitHub API and means a tampered or truncated file is rejected before it is
unpacked. To move to a newer build, run ``tools/pin_llama_runtime.py``.
"""
from __future__ import annotations

import ctypes.util
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from .runtime import BUNDLE, external_environment

PINNED_TAG = "b12317"
RELEASE_BASE = "https://github.com/ggml-org/llama.cpp/releases/download"
SERVER_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str


@dataclass(frozen=True)
class Flavor:
    id: str
    label: str
    detail: str
    assets: tuple
    approx_mb: int


_SHA = {
    f"llama-{PINNED_TAG}-bin-ubuntu-x64.tar.gz": "8c9287285c949ba03bc39065221dd10e741e85508208af96b5c1ba5b9d5761e3",
    f"llama-{PINNED_TAG}-bin-ubuntu-arm64.tar.gz": "1146da2c1b9f5501a500255fac8345cf3d763dca312d77a7da5568e27744fddf",
    f"llama-{PINNED_TAG}-bin-ubuntu-vulkan-x64.tar.gz": "b2eaef4559b3815228b6115fc11204b323a42a3d2139c975add2922509ee33ae",
    f"llama-{PINNED_TAG}-bin-ubuntu-vulkan-arm64.tar.gz": "cd0a26b9862541f56cc00f303a407b503b9ed0b37a222401b73f17abd1bf5ce9",
    f"llama-{PINNED_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz": "c872818ef519140e2dd003765668fbd852446c224a29e95d086847d9733fe2c9",
    f"llama-{PINNED_TAG}-bin-ubuntu-cuda-13.5-x64.tar.gz": "f78e642cb9b4a694a8ac8a864741238dcf4e61fa2e029f5b8f659633c331709a",
    f"cudart-llama-{PINNED_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz": "fc9aeae93fea730e4c3fb9c813b9dbaf50acbadd6d9177a94cc7f4460e53a074",
    f"cudart-llama-{PINNED_TAG}-bin-ubuntu-cuda-13.5-x64.tar.gz": "da597a61b1965446791297a4f0d89cc1f0c1c4454d5f02e25d9791a32290b5d7",
    f"llama-{PINNED_TAG}-bin-macos-arm64.tar.gz": "ac453373fa09110df9edab032a00936dd834f85421c5333cb3a1ff9cbdc410ea",
    f"llama-{PINNED_TAG}-bin-macos-x64.tar.gz": "0b2bb073bdf5bbe4e49587dc7665c2e0fe2c30b550313aadfae9eb1fffad4f32",
    f"llama-{PINNED_TAG}-bin-win-cpu-x64.zip": "96e5008bde3367ef049a0f327e0674b149a4e3385af90b3f046cc7fe0301d322",
    f"llama-{PINNED_TAG}-bin-win-vulkan-x64.zip": "968a487604622f4194d0451d77a92b6a7cff1c29338e88841dd38472c81013ca",
    f"llama-{PINNED_TAG}-bin-win-cuda-12.4-x64.zip": "b6678853912e0cd1ac5adb2258db6b021c1833c4d139001fa1bb3ed7481745ea",
    f"llama-{PINNED_TAG}-bin-win-cuda-13.5-x64.zip": "8c9542f737e3cb2e6592069cb62ff5158cf39de747e225cdd4bb7e7528561969",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "cudart-llama-bin-win-cuda-13.5-x64.zip": "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
}


def _flavor(id, label, detail, mb, *names):
    return Flavor(id, label, detail, tuple(Asset(n, _SHA[n]) for n in names), mb)


_T = PINNED_TAG
_CPU = ("cpu", "CPU only", "Works everywhere, but is the slowest option.")
_VULKAN = ("vulkan", "Vulkan (NVIDIA, AMD, Intel)", "Small download with good speed on most graphics cards.")
FLAVORS = {
    ("linux", "x64"): [
        _flavor(*_CPU, 23, f"llama-{_T}-bin-ubuntu-x64.tar.gz"),
        _flavor(*_VULKAN, 36, f"llama-{_T}-bin-ubuntu-vulkan-x64.tar.gz"),
        _flavor("cuda12", "NVIDIA CUDA 12", "Fastest on NVIDIA cards and works with most drivers. Includes the CUDA runtime.", 792,
                f"llama-{_T}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{_T}-bin-ubuntu-cuda-12.8-x64.tar.gz"),
        _flavor("cuda13", "NVIDIA CUDA 13", "For recent NVIDIA drivers. Includes the CUDA runtime.", 621,
                f"llama-{_T}-bin-ubuntu-cuda-13.5-x64.tar.gz", f"cudart-llama-{_T}-bin-ubuntu-cuda-13.5-x64.tar.gz"),
    ],
    ("linux", "arm64"): [
        _flavor(*_CPU, 23, f"llama-{_T}-bin-ubuntu-arm64.tar.gz"),
        _flavor(*_VULKAN, 36, f"llama-{_T}-bin-ubuntu-vulkan-arm64.tar.gz"),
    ],
    ("macos", "arm64"): [
        _flavor("metal", "Apple silicon (Metal)", "Uses the Mac's GPU.", 17, f"llama-{_T}-bin-macos-arm64.tar.gz"),
    ],
    ("macos", "x64"): [
        _flavor(*_CPU, 17, f"llama-{_T}-bin-macos-x64.tar.gz"),
    ],
    ("windows", "x64"): [
        _flavor(*_CPU, 25, f"llama-{_T}-bin-win-cpu-x64.zip"),
        _flavor(*_VULKAN, 38, f"llama-{_T}-bin-win-vulkan-x64.zip"),
        _flavor("cuda12", "NVIDIA CUDA 12", "Fastest on NVIDIA cards and works with most drivers. Includes the CUDA runtime.", 709,
                f"llama-{_T}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"),
        _flavor("cuda13", "NVIDIA CUDA 13", "For recent NVIDIA drivers. Includes the CUDA runtime.", 605,
                f"llama-{_T}-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"),
    ],
}


def current_platform() -> tuple[str, str]:
    system = {"linux": "linux", "darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64" if machine in ("x86_64", "amd64") else machine
    return system, arch


def detect_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def detect_vulkan() -> bool:
    return bool(ctypes.util.find_library("vulkan") or ctypes.util.find_library("vulkan-1"))


def recommend(flavors, *, nvidia: bool, vulkan: bool) -> str:
    ids = [f.id for f in flavors]
    if "metal" in ids: return "metal"
    if nvidia and "cuda12" in ids: return "cuda12"
    if vulkan and "vulkan" in ids: return "vulkan"
    return "cpu" if "cpu" in ids else ids[0]


@dataclass(frozen=True)
class RuntimeInfo:
    path: Path
    source: str          # "custom", "managed", "bundled" or "path"
    version: str = ""
    flavor: str = ""

    def describe(self) -> str:
        extra = " \u00b7 ".join(p for p in (self.version, self.flavor) if p)
        names = {"custom": "Your chosen llama-server", "managed": "Downloaded runtime", "bundled": "Bundled runtime",
                 "path": "llama-server found on PATH"}
        return names[self.source] + (f" ({extra})" if extra else "")


def find_server(directory: Path) -> Path | None:
    for root, _dirs, files in os.walk(directory):
        if SERVER_NAME in files:
            return Path(root) / SERVER_NAME
    return None


_VERSION_CACHE: dict = {}


def probe_version(binary: Path) -> str:
    """Ask a llama-server binary for its build number (cached; may take a moment on GPU builds)."""
    try:
        key = (str(binary), binary.stat().st_mtime)
    except OSError:
        return ""
    if key not in _VERSION_CACHE:
        version = ""
        try:
            out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=15,
                                 cwd=str(binary.parent), env=external_environment(), check=False)
            text = (out.stdout or "") + (out.stderr or "")
            match = re.search(r"build (\d+)", text)
            version = f"b{match.group(1)}" if match else ""
        except (OSError, subprocess.SubprocessError):
            pass
        _VERSION_CACHE[key] = version
    return _VERSION_CACHE[key]


# ---- safe extraction -------------------------------------------------------------------------------------------

class InstallError(Exception):
    pass


def _inside(root: Path, target: Path) -> bool:
    root, target = os.path.realpath(root), os.path.realpath(target)
    return os.path.commonpath([root, target]) == root


def _strip_prefix(entries: list[tuple[str, bool]]) -> str:
    """The single top-level folder every entry lives under, or "" when entries sit at the archive root.

    ``entries`` are (name, is_directory) pairs. The folder itself counts as inside it.
    """
    tops = set()
    for name, is_dir in entries:
        parts = Path(name).parts
        if not parts: continue
        if len(parts) == 1 and not is_dir: return ""     # a file at the root: nothing to strip
        tops.add(parts[0])
    return next(iter(tops)) if len(tops) == 1 else ""


def _check_name(name: str) -> None:
    parts = Path(name).parts
    if ".." in parts or Path(name).is_absolute() or name.startswith(("/", "\\")):
        raise InstallError(f"Unsafe path in archive: {name}")


def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack .tar.gz or .zip into ``dest``, dropping one shared top folder and refusing path escapes."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members: _check_name(member.name)
            prefix = _strip_prefix([(m.name, m.isdir()) for m in members])
            for member in members:
                parts = Path(member.name).parts
                if prefix and parts and parts[0] == prefix: parts = parts[1:]
                if not parts: continue
                if member.isdev() or member.isfifo():
                    continue
                target = dest.joinpath(*parts)
                if not _inside(dest, target):
                    raise InstallError(f"Unsafe path in archive: {member.name}")
                if member.issym() or member.islnk():
                    link = os.path.normpath(os.path.join(os.path.dirname(target), member.linkname)) if member.issym() \
                        else str(dest.joinpath(*Path(member.linkname).parts[1 if prefix else 0:]))
                    if not _inside(dest, Path(link)):
                        raise InstallError(f"Unsafe link in archive: {member.name}")
                member.name = "/".join(parts)
                if member.islnk(): member.linkname = "/".join(Path(member.linkname).parts[1 if prefix else 0:])
                if hasattr(tarfile, "data_filter"): tar.extract(member, dest, filter="data")
                else: tar.extract(member, dest)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            for info in infos: _check_name(info.filename)
            prefix = _strip_prefix([(i.filename, i.is_dir()) for i in infos])
            for info in infos:
                parts = Path(info.filename).parts
                if prefix and parts and parts[0] == prefix: parts = parts[1:]
                if not parts: continue
                target = dest.joinpath(*parts)
                if not _inside(dest, target):
                    raise InstallError(f"Unsafe path in archive: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True); continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                mode = (info.external_attr >> 16) & 0o777
                if mode: os.chmod(target, mode)
    else:
        raise InstallError(f"Unsupported archive type: {archive.name}")


# ---- installer -------------------------------------------------------------------------------------------------

class _Cancelled(Exception):
    pass


class _InstallThread(QThread):
    progress = Signal(int, int, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, flavor: Flavor, dest: Path, tag: str, base_url: str, parent=None):
        super().__init__(parent)
        self.flavor, self.dest, self.tag, self.base_url = flavor, dest, tag, base_url
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        work = self.dest.parent / f".{self.dest.name}.partial"
        try:
            shutil.rmtree(work, ignore_errors=True)
            stage = work / "stage"; stage.mkdir(parents=True)
            count = len(self.flavor.assets)
            for index, asset in enumerate(self.flavor.assets, 1):
                archive = work / asset.name
                self._download(asset, archive, index, count)
                self.progress.emit(0, 0, f"Unpacking {index}/{count}\u2026")
                safe_extract(archive, stage)
                archive.unlink()
            binary = find_server(stage)
            if binary is None:
                raise InstallError("The download did not contain llama-server.")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            meta = {"tag": self.tag, "flavor": self.flavor.id, "binary": binary.relative_to(stage).as_posix(), "installed": time.time()}
            (stage / "runtime.json").write_text(json.dumps(meta), encoding="utf-8")
            if self._cancel.is_set(): raise _Cancelled
            if self.dest.exists(): shutil.rmtree(self.dest)
            os.replace(stage, self.dest)
            self.done.emit(str(self.dest))
        except _Cancelled:
            self.failed.emit("")
        except urllib.error.HTTPError as error:
            self.failed.emit(f"The download server answered {error.code} for this file.")
        except urllib.error.URLError as error:
            self.failed.emit(f"Could not reach github.com ({error.reason}). Check your connection and try again.")
        except (OSError, InstallError, tarfile.TarError, zipfile.BadZipFile) as error:
            self.failed.emit(str(error) or error.__class__.__name__)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _download(self, asset: Asset, archive: Path, index: int, count: int):
        url = f"{self.base_url}/{self.tag}/{asset.name}"
        request = urllib.request.Request(url, headers={"User-Agent": "TranceLLM-runtime-installer"})
        digest, done = hashlib.sha256(), 0
        with urllib.request.urlopen(request, timeout=30) as response, open(archive, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            while True:
                if self._cancel.is_set(): raise _Cancelled
                chunk = response.read(1 << 20)
                if not chunk: break
                out.write(chunk); digest.update(chunk); done += len(chunk)
                self.progress.emit(done, total, f"Downloading {index}/{count}")
        if digest.hexdigest() != asset.sha256:
            raise InstallError(f"Checksum mismatch for {asset.name}; the file was discarded.")


class _VersionThread(QThread):
    found = Signal()

    def __init__(self, binary: Path, parent=None):
        super().__init__(parent); self.binary = binary

    def run(self):
        probe_version(self.binary); self.found.emit()


class RuntimeManager(QObject):
    """Owns runtime discovery and installation; the UI only reads state and calls these methods."""
    changed = Signal()
    progress = Signal(int, int, str)
    install_failed = Signal(str)

    def __init__(self, state_dir: Path, get_custom, set_custom, parent=None, *, flavors=None, tag=PINNED_TAG,
                 base_url=RELEASE_BASE, bundled_dir: Path | None = None, path_lookup=shutil.which):
        super().__init__(parent)
        self.root = Path(state_dir) / "runtimes" / "llama.cpp"
        self._get_custom, self._set_custom = get_custom, set_custom
        self.flavors = FLAVORS.get(current_platform(), []) if flavors is None else flavors
        self.tag, self.base_url = tag, base_url
        self.bundled_dir = Path(bundled_dir) if bundled_dir else BUNDLE / "llama.cpp"
        self._path_lookup = path_lookup
        self._thread: _InstallThread | None = None
        self._probe: _VersionThread | None = None

    # -- discovery
    def _managed(self) -> RuntimeInfo | None:
        best, newest = None, -1.0
        for meta_file in self.root.glob("*/runtime.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                binary = meta_file.parent / meta["binary"]
                if binary.is_file() and meta["installed"] > newest:
                    best, newest = RuntimeInfo(binary, "managed", meta.get("tag", ""), meta.get("flavor", "")), meta["installed"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return best

    def resolve(self) -> RuntimeInfo | None:
        custom = self._get_custom()
        if custom and Path(custom).is_file():
            return RuntimeInfo(Path(custom), "custom", probe_version(Path(custom)) if self._cached(Path(custom)) else "")
        managed = self._managed()
        if managed: return managed
        bundled = find_server(self.bundled_dir) if self.bundled_dir.is_dir() else None
        if bundled: return RuntimeInfo(bundled, "bundled", self.tag)
        found = self._path_lookup("llama-server")
        if found:
            path = Path(found)
            return RuntimeInfo(path, "path", probe_version(path) if self._cached(path) else "")
        return None

    @staticmethod
    def _cached(path: Path) -> bool:
        try: return (str(path), path.stat().st_mtime) in _VERSION_CACHE
        except OSError: return False

    def has_download(self) -> bool:
        return self._managed() is not None

    def custom_path(self) -> str:
        return self._get_custom() or ""

    def refresh(self):
        """Re-evaluate what is installed; probes versions of external binaries off the UI thread."""
        info = self.resolve()
        if info and info.source in ("custom", "path") and not info.version and (self._probe is None or not self._probe.isRunning()):
            self._probe = _VersionThread(info.path, self); self._probe.found.connect(self.changed); self._probe.start()
        self.changed.emit()

    # -- state for the UI
    @property
    def installing(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def recommended(self) -> str:
        return recommend(self.flavors, nvidia=detect_nvidia(), vulkan=detect_vulkan()) if self.flavors else ""

    def status(self) -> str:
        info = self.resolve()
        if info: return info.describe() + f"\n{info.path}"
        if not self.flavors: return "No prebuilt runtime is available for this system. Choose an existing llama-server."
        return "Not installed. GGUF models need the llama.cpp runtime."

    # -- actions
    def set_custom(self, path: str):
        self._set_custom(path); self.refresh()

    def clear_custom(self):
        self._set_custom(""); self.refresh()

    def remove_managed(self):
        if self.installing: return
        shutil.rmtree(self.root, ignore_errors=True); self.refresh()

    def start_install(self, flavor_id: str):
        if self.installing: return
        flavor = next((f for f in self.flavors if f.id == flavor_id), None)
        if flavor is None:
            self.install_failed.emit("That runtime is not available for this system."); return
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread = _InstallThread(flavor, self.root / f"{self.tag}-{flavor.id}", self.tag, self.base_url, self)
        self._thread.progress.connect(self.progress)
        self._thread.done.connect(self._installed)
        self._thread.failed.connect(self._install_failed)
        self._thread.start(); self.changed.emit()

    def cancel_install(self):
        if self.installing: self._thread.cancel()

    def _installed(self, _dest):
        self._thread.wait(); self._thread = None
        if self._get_custom(): self._set_custom("")   # a fresh download is an explicit choice; stop preferring the old path
        self.refresh()

    def _install_failed(self, message):
        self._thread.wait(); self._thread = None
        self.changed.emit()
        if message: self.install_failed.emit(message)

    def close(self):
        if self.installing:
            self._thread.cancel(); self._thread.wait(5000)
