# Trance LLM AppImage

`dist/TranceLLM-0.7.1-x86_64.AppImage` is a portable Linux desktop application. It contains Python, Qt, the native Trance engine and the CUDA cuBLAS runtime libraries. Launching it does not require a Python environment or a CUDA compiler. An NVIDIA GPU and a compatible NVIDIA driver are required for inference and training.

This build was produced on Linux Mint 22.3 / Ubuntu 24.04 with glibc 2.39. Older Linux distributions and other machines have not been verified. This is an x86_64 build.

## Run

Make the file executable, then double-click it or launch it from your project directory:

```bash
chmod +x dist/TranceLLM-0.7.1-x86_64.AppImage
./dist/TranceLLM-0.7.1-x86_64.AppImage --project "$PWD"
```

When launched elsewhere, it asks for a project folder containing `configs/` and `models/`, and remembers the selected folder. The project must be writable. Keep each model's `.bin` and `.bin.tok` together. Configs, models, training data and chat history remain outside the AppImage so they can change without rebuilding it. Training configs use paths relative to the selected project.

If the system cannot mount AppImages through FUSE, use the built-in extraction mode:

```bash
./dist/TranceLLM-0.7.1-x86_64.AppImage --appimage-extract-and-run --project "$PWD"
```

Training starts only through the Start training button. Each detached worker launches its own AppImage runtime and survives closing the GUI. Reopening the application with the same project reconnects to that run. Stop & save checkpoint creates a stop-request file; the engine stops at a safe training step boundary and writes an optimizer checkpoint. Existing source-launched GUI runs are also recognized.

Release 0.7.1 includes the Beta 19 engine repairs for chat BPE prefix consistency, assistant turn separators and end-of-file training sampling. Earlier AppImages still contain their older engine; open 0.7.1 to use these fixes. Existing model/tokenizer files remain compatible. The [Trance 1.2 training record](../TRANCE_1_2.md) documents the completed stages incorporated into the model.

To check the downloaded file, run from `dist/`:

```bash
sha256sum -c TranceLLM-0.7.1-x86_64.AppImage.sha256
```

## Release downloads

The rolling release (`.github/workflows/rolling-release.yml`, run on every push to `main` or by hand from the Actions tab) publishes one small zip per platform and nothing else:

- `trance-linux-x86_64.zip`: the AppImage, the `trance` engine, and `configs/`, `data/` and `models/` folders beside them, plus `LICENSE`.
- `trance-windows-x86_64.zip`: `trance.exe`, the cuBLAS DLLs, the same three folders, plus `LICENSE`.

No source, tests or build tooling is included. The AppImage uses the folders next to it as its project when `configs/` and `models/` are there, so it works when double-clicked from any location. The order is: `--project`, the `TRANCE_PROJECT_DIR` environment variable, the AppImage's own folder, the project saved from an earlier launch, the current directory, then a folder chooser.

## Rebuild

The release script sets up the isolated packaging environment, checks the downloaded tools, builds a fresh application, verifies it with CPU-only checks, and removes its intermediate packaging files. It packages the existing `trance` engine by default and leaves that binary and active training jobs alone. On a fresh checkout, prepare the native engine and GUI system dependencies using the project instructions first. This build expects CUDA 12 cuBLAS libraries and their package copyright notices under the standard Ubuntu paths, and uses the project-local `libxcb-cursor.so.0` if available.

```bash
./tools/build_release.sh
```

To compile the native engine from source too, use:

```bash
./tools/build_release.sh --compile-engine
```

This compiles with `nvcc -ccbin g++-12 -std=c++14 -O3` into `build/native/trance`, then packages that binary. It never replaces the project's live `./trance` executable. `TRANCE_NVCC` and `TRANCE_HOST_COMPILER` can override the two compiler programs. Compilation does not execute a GPU workload. Both the existing-engine packaging path and the full compilation path have been verified.

The process is repeatable with the documented Linux toolchain and cached checked tool assets. All packaging Python dependencies are pinned in `packaging/requirements.txt`; app name and version are defined once in `gui/__init__.py`. Each release includes the engine/runtime hashes, Python version and installed packaging dependencies in its manifest. The builder publishes the completed AppImage by renaming a temporary file, so it does not overwrite an existing mounted file. Byte-identical output across machines or build times is not guaranteed: native toolchain, platform libraries and filesystem metadata can differ.

The downloader verifies pinned SHA-256 checksums from official AppImage release assets. If upstream replaces a continuous-release asset, it fails rather than accepting a different binary. Review and update the pin before rebuilding with a new tool version. Packaging output and downloaded tools are ignored by Git. The output includes a checksum and JSON manifest identifying the native engine and runtime libraries.

`verify_appimage.py` checks the actual executable's startup, captures its layout, verifies Configs and Documentation plus the bundled article count, and uses an isolated mock engine to verify that a worker survives parent exit and completes a checkpointed stop. It checks the finished image's checksum. It does not execute CUDA or modify a real training job. `dist/AppImage-verification.json` records the result. The exact compiled engine included in release 0.7.1 passed 18 native checks, six native integration tests and four Qt/CUDA integration tests on this project's RTX 4060. Its hash matches the bundle manifest. GPU execution through the frozen GUI is separate from these checks and is not exercised by the package verifier.

Windows EXEs require a Windows build of both the CUDA engine and the frozen Python application. A `.deb` can wrap the Linux bundle with desktop integration in a future packaging pass.

## Build environment notes

- **Pinned tools.** `tools/download_appimage_tools.py` downloads the versioned `appimagetool` 1.9.1 and `type2-runtime` 20251108 releases and checks their SHA-256 hashes. (Earlier versions pinned the upstream "continuous" builds, which are replaced over time and stop matching.) After changing a pin in `tools/build_appimage.py`, run the downloader once so your cached copies are refreshed.
- **cuBLAS location.** The build looks for `libcublas.so.12` and `libcublasLt.so.12` in `TRANCE_CUBLAS_DIR` if set, then `$CUDA_PATH/lib64`, `/usr/local/cuda/lib64` and the toolkit's `targets` folder, then `/lib/x86_64-linux-gnu`. Their license text is collected from any `/usr/share/doc/libcublas*/copyright` and the toolkit's `EULA.txt`; the build stops if none is found, because it must ship with the libraries. `TRANCE_DOC_ROOT` overrides `/usr/share/doc` (used by tests).
- **Hosted runners.** AppImages are run with `APPIMAGE_EXTRACT_AND_RUN=1` because hosted runners have no FUSE.
