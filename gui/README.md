# Trance LLM

A local PySide6 desktop app around the custom Trance CUDA engine. Verified on Linux with an RTX 4060. Chat, history and model comparison run locally. The app makes no network requests on its own: the only built-in network use is the optional **Download** button for the llama.cpp runtime (see [GGUF models](#gguf-models-and-the-llamacpp-runtime)), which contacts github.com when you click it. Setup also downloads Python packages and, when needed, project-local X11 runtime libraries.

## Launch

For the portable Linux build, use `dist/TranceLLM-0.6.0-x86_64.AppImage`. It bundles Python, Qt and the native CUDA engine, and uses a writable external project folder. See [AppImage instructions](../packaging/README.md) for launching, rebuilding and verification.

From the project root:

```bash
./tools/setup_gui.sh
nvcc -O3 -o trance trance.cu -lcublas
./trance-gui
```

If CUDA 12.0 rejects a newer system GCC, use the installed GCC 12:

```bash
nvcc -ccbin g++-12 -std=c++14 -O3 -o trance trance.cu -lcublas
```

The current project environment and native binary are already prepared. `./trance-gui` works from any current directory. It resolves project paths itself, including paths containing spaces. The GUI environment is separate from the Hugging Face data environment. A single-instance lock prevents two desktop windows from overwriting the same conversation history.

On Windows, create a virtual environment, install `gui/requirements.txt`, build `trance.exe`, and run `python -m gui` from the project root using that environment. Windows is not verified here. Response cancellation on Windows restarts the inference process; POSIX cancellation preserves its loaded model. Training stopping uses a portable stop-request file.

## Chat and comparison

- The sidebar brand is two solid colors (the palette's primary and secondary accents). Chat headings switch between New Chat, Chat and Temporary Chat according to the active conversation. Export chat sits at the top right.
- The model picker, Browse and Refresh float in a centered group below the chat header. Model architecture and step details appear in the picker's tooltip. A new empty chat shows a welcome message with four starter prompts (clicking one fills the composer so you can edit it before sending) and a reminder to choose a model; a temporary chat shows its own not-saved notice instead. Send message and Stop response sit inside the compact message box. It grows for wrapped or multiline text, scrolls for long messages, and shrinks again when cleared. The generation panel includes descriptions for every parameter.
- The sidebar has a Chat / Train / Tools switcher at the top. Choosing a section opens the last page you used in it, and the list below a thin rule follows the section: Chat shows New chat, Temporary chat (a dashed outline button), a search box and the saved conversations (full height, scrolling); Train shows Train, Configs and Recent runs; Tools shows Compare, Documentation and Settings. Search matches titles and message text. Hover or select a chat to see a ⋯ button; it opens the same Rename/Delete menu as right-click. Opening an earlier training run from Recent runs is refused while another run is active. A status card at the bottom shows Engine idle / Ready / Generating… / Training · N% / Last request failed, with the GPU name beneath, and the Train tab shows a ● badge while a run is active.
- Right-click a saved conversation to rename or delete it. These actions target the clicked conversation without switching away from a draft or temporary chat. Deleting the active conversation opens a neighboring saved conversation or a fresh unsaved draft if history is empty. Failed history writes preserve the previous conversation/title.
- New chats stay as drafts until the first message is sent; empty drafts do not appear in history. Empty entries left by older releases are filtered on load and removed from the history file when it is next saved. Conversations with messages are preserved.
- Temporary chats keep conversation context in memory and never enter the history file. The empty conversation area displays the centered notice "Chat Not Saved to History. Cleared when you switch chats or close the app." Sending a message replaces the notice with the conversation. Switching to another chat or closing the app discards it. Visiting Train or Compare retains the current temporary chat until you switch chats. Export chat is still available when you explicitly want to save a copy.
- The model list discovers compatible `.bin` models and checkpoints in `models/`, paired with their `.tok` files. Browse can add a model elsewhere. Refresh discovers newly trained models.
- Enter sends a message; Shift+Enter inserts a newline. Responses stream while being generated. Stop cancels inference; closing the window cancels an active response.
- Completed previous exchanges enter the next prompt. The engine counts tokens with the selected model's tokenizer and drops the oldest whole exchanges to reserve the requested response length. An over-length latest message is rejected with a readable error. Interrupted or failed exchanges remain in history but are excluded from future context.
- Temperature, top P, top K, maximum response tokens and seed are adjustable. These controls freeze while generation is active.
- Comparison loads and runs model A, then model B, with the same prompt/settings. Only one model is resident. Responses, timings and settings are saved locally and can be exported. Two identical selections with a fixed seed should produce matching output.
- Saved conversation history, generation settings and comparison results live under `data/gui/`. They persist across launches and can be exported as JSON. Rendered text is escaped; model output is never interpreted as executable markup.

## Training

Training starts **only** when Start training is clicked. Preparing or viewing a config does not start a run. The Trance 1.2 config is selected by default.

The Train page opens with a run summary card (parameters, FP32 weight size, layers × width, context, steps and learning rate, plus whether data files are ready); **Edit JSON** reveals the raw configuration. The page validates supported settings, data paths, model architecture and output paths. An existing final model cannot be overwritten from the GUI. Resume requires an optimizer checkpoint with a matching architecture. Each run receives an immutable config snapshot under `data/gui/runs/<run>/`, along with `events.jsonl`, `status.json` and supervisor logs.

A detached supervisor owns the training process. Closing the GUI leaves an authorized run running; reopening the app reconnects to it. Starting a run releases inference GPU memory, and chat/comparison are disabled while that run is active. The dashboard displays loss, steps, elapsed time, estimated remaining time and checkpoint paths. ETA begins after 25 completed steps; validation follows the config's eval interval.

**Stop & save checkpoint** requests a stop at a safe step boundary. The native trainer writes `OUTPUT.interrupted.bin` with Adam state and a matching tokenizer. Model/checkpoint weights are committed through a temporary file so a failed weight write preserves the previous checkpoint. Preparation may need to finish before the first stop checkpoint can be saved.

**Resume checkpoint** prepares an editable config with the remaining step budget; click Start training to resume. Sampling RNG state is not serialized by the existing model format, so optimizer resume does not reproduce the exact uninterrupted sampling trajectory. Resuming writes to the configured final output only after completing the remaining budget.

The dashboard tracks runs started through this GUI. It does not attach to arbitrary existing CLI jobs.

## Configs

Open **Configs** in the sidebar to create a configuration without editing raw JSON. The Model, Training and Data & output tabs expose the supported settings with tooltips; JSON preview shows exactly what will be saved. Starter presets include a small experiment, a tiny learning model and the Trance 1.2 architecture with a 50,000-step experiment budget and checkpoints every 5,000 steps. These presets do not change existing configs or guarantee an ideal training duration.

Load template imports an existing JSON config without modifying it. Give the new config its own name and output model. Save config writes a new file under `configs/` and preserves an existing file with that name. It checks architecture, existing data files, output-model protection and any resume checkpoint. The parameter/weights estimate is not a GPU memory estimate.

Saving does not begin training. Open in Train selects the saved config for review; Start training remains the explicit launch action. Changing builder fields after saving requires another save before opening that version in Train. Config creation and documentation remain available during an active run.

## Documentation

The **Documentation** section bundles 14 offline articles under `gui/articles/`, grouped into Getting started, Inside Trance, and Training and use. They explain tokens/BPE, embeddings/positions, causal attention, residuals/LayerNorm/feed-forward blocks, the output head, training/loss/overfitting, data splits, sampling, configs, checkpoints, CUDA memory and background processes.

Search includes article text and favors title matches. Related-article links navigate inside the app. External source links open a browser only when clicked. The files are bundled into the AppImage; no network connection or project-specific documentation files are needed to read them.

## Engine protocol

`./trance engine` is a persistent JSON-lines stdin/stdout mode. It emits a `ready` event, accepts numeric request IDs, and supports `load`, `generate`, `unload`, and `shutdown`. Commands and events share a request ID. Fatal CUDA or weight-loading errors exit the process and are shown by the GUI; malformed requests return an error and leave the engine usable.

```json
{"id":1,"command":"load","model":"models/trance1.1_21m.bin"}
{"id":2,"command":"generate","messages":[{"role":"user","content":"Hi!"}],"temperature":0.8,"top_k":40,"top_p":0.95,"max_new_tokens":160,"seed":4004}
```

Generation emits `generation_started`, `delta` and `done`. Delta payloads contain hex-encoded bytes, decoded incrementally in the GUI because BPE boundaries can split a UTF-8 character. `done` contains stop reason, generated-token count and elapsed time. On POSIX, SIGUSR1 cancels the active generation.

Beta 19 ends a chat generation prefix at the atomic `<assistant>` marker. The model predicts the following formatting newline because its BPE token may also contain the first answer letters; the engine hides that one leading newline in displayed output while retaining it in token history. Completed assistant turns have their trailing newline separator normalized before reconstructing history. These changes work with existing model/tokenizer files. The trainer also preserves conversation starts near end-of-file and masks padded targets instead of shifting the sampled window backwards. The [Trance 1.2 training record](../TRANCE_1_2.md) documents the completed stages incorporated into the model.

`./trance train --config CONFIG --json-events --stop-file PATH` emits stage/progress/checkpoint/complete/stopped events. Creating PATH requests a checkpointed stop. SIGINT/SIGTERM also request graceful stopping during training. Normal terminal commands retain their human-readable output. The existing transformer operations and Adam update math are unchanged.

## Verification

```bash
./trance test
g++ -std=c++14 -O2 tests/test_json.cpp -o /tmp/trance-json-tests
/tmp/trance-json-tests
QT_QPA_PLATFORM=offscreen .venv-gui/bin/python -m unittest discover -s tests -v
```

The Python suite includes storage/config tests, real-model streaming/context/cancellation checks, and Qt interface checks. It uses this project's existing Trance 1.1 model and prepared Trance 1.2 data; an empty release package needs those artifacts before running the Python suite. Training checks use temporary miniature models only; they never train Trance 1.2 or alter the existing model. The close/reopen test verifies that the detached miniature job survives closing the GUI, reconnects, stops with an optimizer checkpoint and prepares a remaining-budget resume config.

While another training run is active, use the CPU-only checks:

```bash
QT_QPA_PLATFORM=offscreen .venv-gui/bin/python -m unittest discover -s tests -p 'test_gui_navigation.py' -v
QT_QPA_PLATFORM=offscreen .venv-gui/bin/python -m unittest discover -s tests -p 'test_gui_storage.py' -v
QT_QPA_PLATFORM=offscreen .venv-gui/bin/python -m unittest discover -s tests -p 'test_packaging_runtime.py' -v
```

To capture the app for layout inspection without starting inference or training:

```bash
QT_QPA_PLATFORM=offscreen ./trance-gui --screenshot /tmp/trance-llm.png
```

## GGUF models and the llama.cpp runtime

Besides Trance `.bin` checkpoints, chat and comparison can use **GGUF** models, the format used by llama.cpp, LM Studio and Ollama. Trance checkpoints keep running on the built-in CUDA engine, which is also what trains models; GGUF models run through llama.cpp, the same engine LM Studio uses.

- **Finding models.** GGUF files are listed automatically from `models/`, LM Studio's model folders and the Hugging Face cache. Use **Browse…** to add one from anywhere (it is remembered). Split models are listed once, and multimodal projector files are ignored.
- **The runtime.** GGUF needs a `llama-server` program. Tools → Settings → Runtimes shows which one is in use. The app looks, in order, for: a path you chose, a runtime it downloaded, a bundled one, then `llama-server` on your PATH. If none is found, **Download** fetches a pinned llama.cpp build (CPU, Vulkan, CUDA 12 or 13, or Metal depending on your system; the recommended one is preselected). Files come from the official `ggml-org/llama.cpp` GitHub releases and are checked against SHA-256 hashes built into the app before they are unpacked. CUDA builds include their CUDA runtime, so they are several hundred MB.
- **Privacy.** The only network use is that download, and only when you click it. Afterwards everything runs on your machine: the server listens on 127.0.0.1 only, on a random port, protected by a random per-launch API key, and it is stopped when you switch engines, start training, or close the app.
- **Behavior notes.** One model is loaded at a time and stays loaded between messages. Starting training or switching to a Trance model frees the GPU first. Context is capped at 8,192 tokens (or the model's own limit if smaller). Sampling settings map to temperature, top-p, top-k, max tokens and seed; the server applies the model's own chat template. The context hint shows usage after each reply.
- **AppImage note.** The AppImage puts its own library folders (bundled Qt, Python and cuBLAS) on `LD_LIBRARY_PATH`; the llama.cpp server is started without those entries so it loads its own libraries, while anything you had set yourself is kept.
- **Maintenance.** `python tools/pin_llama_runtime.py` re-verifies the pinned hashes; with `--tag bNNNN --write` it moves the app to a newer llama.cpp build.

## Settings and themes

Tools → Settings lets you choose a color theme: Teal (default), Indigo or Amber. A theme changes only the accent colors; it applies immediately, including to open chats, and is saved in `data/gui/settings.json` as `theme`. An unknown saved value falls back to Teal. Palettes live in `gui/theme.py` (`PALETTES`); add an entry there and a card appears automatically on the Settings page.
