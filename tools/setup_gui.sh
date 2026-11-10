#!/usr/bin/env bash
# Project-local setup. Never installs system packages or uses sudo.
set -euo pipefail
trance_setup_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$trance_setup_root"
python3 -m venv .venv-gui
.venv-gui/bin/python -m pip install -r gui/requirements.txt

if [ "$(uname -s)" = Linux ] && command -v ldd >/dev/null; then
    trance_qt_plugin="$(.venv-gui/bin/python -c 'from pathlib import Path; import PySide6; print(Path(PySide6.__file__).parent / "Qt/plugins/platforms/libqxcb.so")')"
    trance_setup_libs="$trance_setup_root/.venv-gui/native-deps/root/usr/lib/x86_64-linux-gnu"
    mapfile -t trance_missing_libs < <(LD_LIBRARY_PATH="$trance_setup_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$trance_qt_plugin" | awk '/not found/ {print $1}' | sort -u)
    if [ "${#trance_missing_libs[@]}" -gt 0 ]; then
        if ! command -v apt-get >/dev/null || ! command -v dpkg-deb >/dev/null; then
            echo "Missing Qt desktop libraries: ${trance_missing_libs[*]}" >&2
            echo "Install their distribution packages before launching the GUI." >&2
            exit 1
        fi
        declare -A trance_lib_packages=(
            [libxcb-cursor.so.0]=libxcb-cursor0 [libxcb-icccm.so.4]=libxcb-icccm4
            [libxcb-image.so.0]=libxcb-image0 [libxcb-keysyms.so.1]=libxcb-keysyms1
            [libxcb-render-util.so.0]=libxcb-render-util0 [libxcb-xinerama.so.0]=libxcb-xinerama0
            [libxcb-xkb.so.1]=libxcb-xkb1 [libxkbcommon-x11.so.0]=libxkbcommon-x11-0
        )
        trance_packages=()
        for trance_lib in "${trance_missing_libs[@]}"; do
            trance_package="${trance_lib_packages[$trance_lib]:-}"
            if [ -z "$trance_package" ]; then
                echo "Missing Qt library $trance_lib. Install its distribution package." >&2
                exit 1
            fi
            trance_packages+=("$trance_package")
        done
        mkdir -p .venv-gui/native-deps
        (
            cd .venv-gui/native-deps
            apt-get download "${trance_packages[@]}"
            for trance_archive in ./*.deb; do dpkg-deb --extract "$trance_archive" root; done
        )
    fi
fi
echo "GUI environment ready. Build the CUDA engine if needed, then run ./trance-gui"
