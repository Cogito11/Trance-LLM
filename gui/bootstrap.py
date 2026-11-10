"""Frozen app entry: select a writable project before importing its paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


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
