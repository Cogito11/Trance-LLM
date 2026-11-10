#!/usr/bin/env bash
# Rebuild and verify the AppImage without running any real CUDA workloads.
set -euo pipefail
trance_build_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$trance_build_root"
trance_build_compile=0
if [[ "${1:-}" == "--compile-engine" ]]; then
    trance_build_compile=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: ./tools/build_release.sh [--compile-engine]" >&2
    exit 2
fi
python3 -m venv .venv-packaging
.venv-packaging/bin/python -m pip install --disable-pip-version-check -r packaging/requirements.txt
.venv-packaging/bin/python tools/download_appimage_tools.py
trance_build_engine="$trance_build_root/trance"
if [[ "$trance_build_compile" == 1 ]]; then
    mkdir -p build/native
    "${TRANCE_NVCC:-nvcc}" -ccbin "${TRANCE_HOST_COMPILER:-g++-12}" -std=c++14 -O3 \
        -o build/native/trance.tmp trance.cu -lcublas
    mv -- build/native/trance.tmp build/native/trance
    trance_build_engine="$trance_build_root/build/native/trance"
fi
nice -n 10 .venv-packaging/bin/python tools/build_appimage.py --engine "$trance_build_engine"
nice -n 10 .venv-packaging/bin/python tools/verify_appimage.py
# All intermediate files can be recreated. Keep only the finished release,
# downloaded checked tools, and optional separately compiled native engine.
rm -rf -- "$trance_build_root/build/packaging"
echo "Verified release saved in $trance_build_root/dist"
