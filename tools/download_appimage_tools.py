#!/usr/bin/env python3
"""Download pinned official AppImage tools and check every byte before use."""
from pathlib import Path
import urllib.request

from build_appimage import TOOL, TOOL_SHA, RUNTIME, RUNTIME_SHA, sha


def download(destination: Path, url: str, digest: str):
    if destination.is_file() and sha(destination) == digest:
        print(f"Verified cached {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TranceLLM-packaging"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha(temporary) != digest:
            raise ValueError(f"Checksum mismatch for {destination.name}; upstream continuous release may have changed")
        temporary.replace(destination)
        destination.chmod(0o755)
        print(f"Downloaded and verified {destination.name}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    download(TOOL, "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage", TOOL_SHA)
    download(RUNTIME, "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64", RUNTIME_SHA)
