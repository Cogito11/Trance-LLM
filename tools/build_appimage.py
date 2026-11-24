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
# Versioned releases only: the "continuous" builds are replaced upstream, which silently breaks the pinned hashes.
TOOL_URL="https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
RUNTIME_URL="https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64"
TOOL=ROOT/"packaging/downloads/appimagetool-x86_64.AppImage"
TOOL_SHA="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
RUNTIME=ROOT/"packaging/downloads/runtime-x86_64"
RUNTIME_SHA="2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()


# PyInstaller only *warns* when it cannot find these, and the AppImage then opens no window on a machine without them.
REQUIRED_BUNDLED_LIBS = ("libxcb-shape.so.0","libxcb-xkb.so.1","libxcb-keysyms.so.1","libxcb-icccm.so.4","libxkbcommon-x11.so.0","libxcb-cursor.so.0")
BUILD_PACKAGES = "libxcb-shape0 libxcb-xkb1 libxcb-keysyms1 libxcb-icccm4 libxkbcommon-x11-0 libxcb-cursor0 libxcb-image0 libxcb-render-util0 libxcb-util1"


def missing_bundled_libs(internal):
    return [name for name in REQUIRED_BUNDLED_LIBS if not (Path(internal)/name).exists()]


def cublas_search_dirs(environ=None):
    """Where the cuBLAS libraries may live: an explicit override, the CUDA toolkit, then the distro package location."""
    env = os.environ if environ is None else environ
    dirs = []
    if env.get("TRANCE_CUBLAS_DIR"): dirs.append(Path(env["TRANCE_CUBLAS_DIR"]))
    for root in (env.get("CUDA_PATH"), "/usr/local/cuda"):
        if root: dirs += [Path(root)/"lib64", Path(root)/"targets/x86_64-linux/lib"]
    dirs.append(Path("/lib/x86_64-linux-gnu"))
    return dirs


def find_cublas(name, environ=None):
    for directory in cublas_search_dirs(environ):
        if (directory/name).exists(): return directory/name
    return None


def find_cublas_licenses(environ=None):
    """(file name, source) pairs for the license text that must ship with the redistributed cuBLAS libraries."""
    env = os.environ if environ is None else environ
    found = []
    docs = Path(env.get("TRANCE_DOC_ROOT", "/usr/share/doc"))
    for copyright in sorted(docs.glob("libcublas*/copyright")):
        found.append((copyright.parent.name+"-copyright.txt", copyright))
    for root in (env.get("CUDA_PATH"), "/usr/local/cuda"):
        eula = Path(root)/"EULA.txt" if root else None
        if eula and eula.is_file() and not any(n == "cuda-EULA.txt" for n, _ in found): found.append(("cuda-EULA.txt", eula))
    return found


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
        source=find_cublas(name)
        if source is None:
            raise SystemExit(f"Missing CUDA runtime library {name}. Looked in: "+", ".join(map(str,cublas_search_dirs()))+". Set TRANCE_CUBLAS_DIR to the folder that holds it.")
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
    missing=missing_bundled_libs(appdir/"_internal")
    if missing: raise SystemExit("The AppImage would not open a window on machines without: "+", ".join(missing)+f". Install them on the build machine first: sudo apt install {BUILD_PACKAGES}")
    licenses=appdir/"_internal/licenses"; licenses.mkdir(exist_ok=True)
    notices = find_cublas_licenses()
    if not notices: raise SystemExit("Cannot find the license text for the cuBLAS libraries (a libcublas* copyright file or the CUDA toolkit EULA.txt); it must ship with them.")
    for filename, source in notices:
        shutil.copy2(source,licenses/filename)
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
