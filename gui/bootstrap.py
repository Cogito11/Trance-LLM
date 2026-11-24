"""Frozen app entry: select a writable project before importing its paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


_PACKAGES = (("libGL.so", "libgl1"), ("libEGL.so", "libegl1"), ("libxkbcommon-x11", "libxkbcommon-x11-0"), ("libxkbcommon", "libxkbcommon0"),
             ("libfontconfig", "libfontconfig1"), ("libxcb-cursor", "libxcb-cursor0"), ("libdbus", "libdbus-1-3"))


def explain_startup_error(error):
    """A plain-English message when the app cannot start because the system lacks a graphics library, else None.

    The packaged app carries Qt and Python but uses the machine's own OpenGL stack, so a bare server image
    (or a minimal container) can lack it. Without this the user sees a Python traceback.
    """
    text=str(error)
    if "cannot open shared object file" not in text: return None
    library=text.split(":")[0].strip() or "a system library"
    package=next((pkg for prefix,pkg in _PACKAGES if library.startswith(prefix)),None)
    hint=f"On Ubuntu or Debian: sudo apt install {package}" if package else "Install the package that provides it with your system's package manager."
    return (f"Trance LLM could not start because this system is missing {library}, which the desktop interface needs.\n"
            f"{hint}\n(Desktops normally have it; minimal server images and containers often do not.)")


def appimage_project():
    """The folder holding the AppImage, when configs/ and models/ sit beside it (a downloaded release).

    This is what makes double-clicking the AppImage work with its sibling folders, wherever it was unpacked.
    """
    appimage=os.environ.get("APPIMAGE")
    if not appimage: return None
    folder=Path(appimage).resolve().parent
    if (folder/"configs").is_dir() and (folder/"models").is_dir() and os.access(folder,os.W_OK): return folder
    return None


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project",type=Path)
    parser.add_argument("--training-worker",type=Path)
    parser.add_argument("--diagnostics",action="store_true")
    parser.add_argument("--verify-detached-worker",type=Path,help=argparse.SUPPRESS)
    options,rest=parser.parse_known_args()
    frozen=bool(getattr(sys,"frozen",False))
    project=options.project
    if project is None and os.environ.get("TRANCE_PROJECT_DIR"):
        project=Path(os.environ["TRANCE_PROJECT_DIR"])
    if project is None and frozen:
        project=appimage_project()
    if project is None and frozen:
        # Only read settings until the user explicitly chooses a project.
        from PySide6.QtCore import QSettings
        saved=QSettings("Trance","LLM").value("project","") or QSettings("Trance","Studio").value("project","")
        if saved and Path(saved).is_dir(): project=Path(saved)
        elif (Path.cwd()/"configs").is_dir() and (Path.cwd()/"models").is_dir(): project=Path.cwd()
        elif options.training_worker or options.diagnostics or options.verify_detached_worker or "--screenshot" in rest:
            raise SystemExit("Supply --project PATH for a worker, diagnostics, or layout capture")
        else:
            from PySide6.QtWidgets import QApplication,QFileDialog
            app=QApplication.instance() or QApplication([])
            selected=QFileDialog.getExistingDirectory(None,"Choose your Trance project folder",str(Path.home()))
            if not selected: return 0
            project=Path(selected)
            if not (project/"configs").is_dir() or not (project/"models").is_dir():
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(None,"Choose a Trance project","Select a folder containing configs and models. Your existing Trance project can be used.")
                return 1
            QSettings("Trance","LLM").setValue("project",str(project))
    if project is not None:
        project=project.expanduser().resolve()
        if not project.is_dir(): raise SystemExit(f"Project folder does not exist: {project}")
        os.environ["TRANCE_PROJECT_DIR"]=str(project)
    sys.argv=[sys.argv[0],*rest]
    if options.diagnostics:
        from .runtime import PROJECT,ENGINE,BUNDLE
        from .documentation import load_articles
        print(json.dumps({"frozen":frozen,"project":str(PROJECT),"engine":str(ENGINE),"engine_exists":ENGINE.is_file(),"bundle":str(BUNDLE),"documentation_articles":len(load_articles())}))
        return 0
    if options.verify_detached_worker:
        # Packaging tests supply a mock CPU-only engine and an isolated project.
        # Exercise the same detached launch used by the Start training button,
        # then exit the parent immediately to test worker resource lifetime.
        from PySide6.QtCore import QCoreApplication
        app=QCoreApplication.instance() or QCoreApplication([])
        from .training import launch_training
        config=json.loads(options.verify_detached_worker.read_text())
        directory=launch_training(config)
        print(json.dumps({"run_dir":str(directory)}))
        return 0
    if options.training_worker:
        from .training_worker import run
        return run(options.training_worker)
    from .app import main as gui_main
    return gui_main()
