#!/usr/bin/env python3
"""Build a portable Linux bundle without recompiling or launching the CUDA engine.
Run: .venv-packaging/bin/python tools/build_appimage.py
The official appimagetool is downloaded separately to packaging/downloads/.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gui import APP_NAME, APP_SLUG, APP_VERSION, APPIMAGE_NAME
BUILD=ROOT/"build/packaging"
DIST=ROOT/"dist"
TOOL=ROOT/"packaging/downloads/appimagetool-x86_64.AppImage"
TOOL_SHA="95cbe7cce9717fce90c484e34052ee7c7f1d7635b33c12525b4776826a7d29b6"
RUNTIME=ROOT/"packaging/downloads/runtime-x86_64"
RUNTIME_SHA="156f4bdbde9c52d01814600013e0a273f0118dc2de98975f3c8c63427ec79074"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine",type=Path,default=ROOT/"trance",help="Native engine to package; it is never rebuilt or modified by this script")
    options=parser.parse_args()
    if sys.platform!="linux" or platform.machine()!="x86_64":
        raise SystemExit("This build targets Linux x86_64. Windows EXEs must be built on Windows.")
    if not TOOL.is_file() or sha(TOOL)!=TOOL_SHA:
        raise SystemExit("Download the pinned official appimagetool first; its SHA256 must match the build script.")
    if not RUNTIME.is_file() or sha(RUNTIME)!=RUNTIME_SHA:
        raise SystemExit("Download the pinned official AppImage runtime first.")
    engine=options.engine.resolve()
    if not engine.is_file(): raise SystemExit("Build the native trance binary before packaging.")
    BUILD.mkdir(parents=True,exist_ok=True); DIST.mkdir(exist_ok=True)
    inputs=BUILD/"inputs"; cuda=inputs/"cuda"; cuda.mkdir(parents=True,exist_ok=True)
    staged_engine=inputs/"engine/trance"
    staged_engine.parent.mkdir(exist_ok=True)
    shutil.copy2(engine,staged_engine)
    copied={}
    for name in ("libcublas.so.12","libcublasLt.so.12"):
        source=Path("/lib/x86_64-linux-gnu")/name
        if not source.exists(): raise SystemExit(f"Missing CUDA runtime library: {source}")
        dest=cuda/name; shutil.copy2(source.resolve(),dest); copied[name]=sha(dest)
    args=[sys.executable,"-m","PyInstaller","--noconfirm","--clean","--onedir","--name",APP_SLUG,
          "--paths",str(ROOT),"--distpath",str(BUILD/"frozen"),"--workpath",str(BUILD/"pyinstaller"),
          "--specpath",str(BUILD),"--hidden-import","gui.training_worker",
          "--add-data",f"{ROOT/'gui/assets'}:gui/assets","--add-data",f"{staged_engine}:engine",
          "--add-data",f"{ROOT/'gui/articles'}:gui/articles",
          "--add-data",f"{cuda}:cuda","--add-data",f"{ROOT/'LICENSE'}:licenses",
          str(ROOT/"packaging/entrypoint.py")]
    subprocess.run(args,cwd=ROOT,check=True)
    appdir=BUILD/"TranceLLM.AppDir"
    if appdir.exists(): shutil.rmtree(appdir)
    shutil.copytree(BUILD/"frozen"/APP_SLUG,appdir,symlinks=True)
    shutil.copy2(ROOT/"packaging/AppRun",appdir/"AppRun"); (appdir/"AppRun").chmod(0o755)
    shutil.copy2(ROOT/"packaging/trance-llm.desktop",appdir/"trance-llm.desktop")
    shutil.copy2(ROOT/"gui/assets/trance.svg",appdir/"trance-llm.svg")
    (appdir/".DirIcon").symlink_to("trance-llm.svg")
    (appdir/"_internal/engine/trance").chmod(0o755)
    # PyInstaller discovers the engine's ELF dependencies and copies cuBLAS
    # again at the top level. Keep only the explicitly copied, unmodified
    # redistributables; AppRun places that cuda directory first in the search path.
    for duplicate in (appdir/"_internal").glob("libcublas*.so*"):
        duplicate.unlink()
    # Include the X11 library repaired locally for this project. All other
    # Qt dependencies are collected by the PyInstaller Qt hooks.
    cursor=ROOT/".venv-gui/native-deps/root/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0"
    if not cursor.exists(): cursor=Path("/lib/x86_64-linux-gnu/libxcb-cursor.so.0")
    if cursor.exists(): shutil.copy2(cursor.resolve(),appdir/"_internal/libxcb-cursor.so.0")
    licenses=appdir/"_internal/licenses"; licenses.mkdir(exist_ok=True)
    for name in ("libcublas12","libcublaslt12"):
        shutil.copy2(Path("/usr/share/doc")/name/"copyright",licenses/(name+"-copyright.txt"))
    qt_license=Path(__import__("PySide6").__file__).parent/"Qt/licenses"
    if qt_license.is_dir(): shutil.copytree(qt_license,licenses/"Qt",dirs_exist_ok=True)
    for name,digest in copied.items():
        assert sha(appdir/"_internal/cuda"/name)==digest,"CUDA runtime must be copied without modification"
    assert sha(appdir/"_internal/engine/trance")==sha(engine),"Packaged engine must match the selected native binary"
    tool_dir=BUILD/"appimagetool"
    tool_dir.mkdir(exist_ok=True)
    if not (tool_dir/"squashfs-root/AppRun").exists():
        subprocess.run([str(TOOL),"--appimage-extract"],cwd=tool_dir,stdout=subprocess.DEVNULL,check=True)
    output=DIST/APPIMAGE_NAME
    temporary_output=DIST/("."+APPIMAGE_NAME+".building")
    temporary_output.unlink(missing_ok=True)
    env={**os.environ,"ARCH":"x86_64"}
    # Never overwrite an AppImage that may be mounted by a GUI or worker.
    # Rename a fully built file into place so existing mounts keep their inode.
    try:
        subprocess.run([str(tool_dir/"squashfs-root/AppRun"),"--runtime-file",str(RUNTIME),"--no-appstream","--mksquashfs-opt","-processors",
                        "--mksquashfs-opt","2",str(appdir),str(temporary_output)],env=env,cwd=ROOT,check=True)
        temporary_output.chmod(0o755)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
    manifest={"app_name":APP_NAME,"version":APP_VERSION,"platform":"linux-x86_64","build_os":platform.platform(),
              "artifact":output.name,"bytes":output.stat().st_size,"sha256":sha(output),
              "appimagetool_sha256":TOOL_SHA,"runtime_sha256":RUNTIME_SHA,"engine_sha256":sha(engine),"cuda_libraries":copied,
              "python_version":platform.python_version(),
              "python_dependencies":subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines(),
              "includes_models":False,"includes_training_data":False}
    (DIST/(output.name+".json")).write_text(json.dumps(manifest,indent=2)+"\n")
    (DIST/(output.name+".sha256")).write_text(f"{manifest['sha256']}  {output.name}\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
