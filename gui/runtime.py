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
