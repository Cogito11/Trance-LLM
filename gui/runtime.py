"""Separate bundled read-only program assets from the writable project."""
from __future__ import annotations

import os
from pathlib import Path
import sys

FROZEN = bool(getattr(sys,"frozen",False))
BUNDLE = Path(getattr(sys,"_MEIPASS",Path(__file__).resolve().parents[1]))
PROJECT = Path(os.environ.get("TRANCE_PROJECT_DIR",Path(__file__).resolve().parents[1])).expanduser().resolve()
ENGINE = Path(os.environ.get("TRANCE_ENGINE_BINARY", BUNDLE / "engine" / ("trance.exe" if os.name=="nt" else "trance")
                             if FROZEN else PROJECT / ("trance.exe" if os.name=="nt" else "trance"))).resolve()


def supervisor_program() -> str:
    # An AppImage worker needs its own mount, so it can outlive the GUI's mount.
    appimage=os.environ.get("APPIMAGE")
    return str(Path(appimage).resolve()) if FROZEN and appimage and Path(appimage).is_file() else sys.executable


def supervisor_arguments(directory: Path) -> list[str]:
    return ["--project",str(PROJECT),"--training-worker",str(directory)] if FROZEN else ["-m","gui.training_worker",str(directory)]


def external_environment(environ=None) -> dict:
    """Environment for programs the app does not ship (llama-server).

    The AppImage puts its own library folders (bundled Qt, Python and cuBLAS) on LD_LIBRARY_PATH. A child
    program must not inherit them or it could load our copies instead of its own, so entries inside the
    bundle are dropped and everything the user had set before launching is kept.
    """
    env = dict(os.environ if environ is None else environ)
    if FROZEN:
        bundle = os.path.realpath(BUNDLE)
        for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            kept = [p for p in env.get(name, "").split(os.pathsep)
                    if p and os.path.commonpath([bundle, os.path.realpath(p)]) != bundle]
            if kept: env[name] = os.pathsep.join(kept)
            else: env.pop(name, None)
    return env
