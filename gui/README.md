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

- **Replies are formatted.** Model replies are rendered as markdown (headings, bold and italic, lists, tables, code blocks, links and quotes) with the app's colors. **Callouts** become tinted boxes: GitHub alerts (`> [!NOTE]`, `TIP`, `IMPORTANT`, `WARNING`, `CAUTION`, with an optional custom title after the marker) and the common Obsidian types such as `info`, `success`, `danger` and `example`. **Math** written as `$...$`, `$$...$$`, `\\(...\\)` or `\\[...\\]` is shown as readable Unicode: Greek letters, operators, roots, superscripts and subscripts, accents and simple fractions (`x = (\u2212b \u00b1 \u221a(b\u00b2 \u2212 4ac))/2a`). This is a text approximation, not typesetting: fractions are written inline, and anything it does not understand (matrices, unknown commands) is left as the original LaTeX. Dollar amounts in ordinary text, and anything inside code, are never treated as math. Your own messages are shown as typed. Raw HTML in a reply is shown as text rather than interpreted, the model's own line breaks are kept, and a link only opens (in your browser, for http and https only) when you click it. Each reply is labeled with the model that wrote it.
- **No cut-off answers.** Max tokens can be set to **No limit**, which is the default for GGUF models: the model writes until it is finished or the context window is full. The two kinds of model remember their own setting (Trance models keep a modest default of 160; "No limit" for a Trance model means up to half its context window). If a reply is still cut short, its note says whether the token limit or a full context window ended it.
- **Think.** A **Think** toggle at the bottom left of the message box turns on a model's thinking mode. It is enabled only for GGUF models whose chat template has one (read from the file), and is greyed out with an explanation for the rest. The model's reasoning appears in a fold above its reply, formatted with the same markdown (lists, bold, math, callouts and tables) in a quieter, smaller style: open while it is thinking, then collapsed as "Thought for N s" until you click it. Reasoning is never sent back to the model on later turns.
- **Speed and size.** Under each reply: tokens per second, then the number of tokens, then the response time (for example `42.7 tok/s \u00b7 612 tokens \u00b7 14.3s`). For GGUF models the speed is the server's own generation speed, which leaves out the time spent reading your prompt; for Trance models it is tokens divided by time.
- **Scrolling.** While you are at the end of the conversation the view follows the reply as it is written. Scroll up to read something earlier and it stays where you put it; scroll back to the end and it follows again. Sending a message, or opening a chat, jumps to the end.
- **Menus and lists.** The chat menus and every drop-down list have rounded corners and hover effects, and the ⋯ button on a chat highlights when the pointer is over it. The rounded corners need a compositing window manager; without one they may look square.
- **Loading and Eject.** Picking a model loads it right away, replacing whichever model was loaded: only one model is in memory at a time. A model stays loaded between messages for speed. The Eject button beside the model button unloads it and frees its memory; it is also released when you start training or close the app. While a model loads the button reads Cancel. Opening another chat does not change the loaded model, and sending without one explains how to load one. The status card shows which model is loaded. In Settings, **Default model** chooses a model that loads automatically when the app starts (None starts empty).
- The sidebar brand is two solid colors (the palette's primary and secondary accents). Chat headings switch between New Chat, Chat and Temporary Chat according to the active conversation. The model button and Eject sit in that same top row; at the top right, the ⋯ button opens the chat menu (Rename, Export, Delete) once a chat has messages, and temporary chats also show **Save Chat** to its left.
- With no model loaded the model button reads "No Model Loaded, Pick one to use". Clicking it opens a model picker over the window: a searchable list with the loaded model first, plus Browse…, Refresh list and Eject. Click a model (or press Enter) to load it; Escape or a click outside closes it. Model architecture and step details appear in the button's tooltip. The model button and Eject sit centered in the top row in a fixed-width group, so it never changes size when a model loads. The window is at least 1240 px wide so the whole top row fits. A new empty chat shows a large, randomly chosen greeting (“Welcome”, “What’s up next?” and so on) and one row of rotating prompt suggestions above the message box; clicking a suggestion fills the box so you can edit it before sending. The suggestions pause rotating while the pointer is over them or you are typing. A temporary chat shows its own not-saved notice instead. Send message and Stop response sit inside the compact message box. It grows for wrapped or multiline text, scrolls for long messages, and shrinks again when cleared. The generation panel includes descriptions for every parameter.
- The sidebar has a Chat / Train / Tools switcher at the top. Choosing a section opens the last page you used in it, and the list below a thin rule follows the section: Chat shows New chat, Temporary chat (a dashed outline button), a search box and the saved conversations (full height, scrolling); Train shows Train, Configs and Recent runs; Tools shows Compare, Documentation and Settings. Search matches titles and message text. Hover or select a chat to see a ⋯ button; it opens the same Rename/Delete menu as right-click. Opening an earlier training run from Recent runs is refused while another run is active. A status card at the bottom shows Engine idle / Ready / Generating… / Training · N% / Last request failed, with the GPU name beneath, and the Train tab shows a ● badge while a run is active.
- Right-click a saved conversation, or use its ⋯ button in the sidebar, for **Rename**, **Export** and **Delete**; the ⋯ button at the top of the chat shows the same menu for the open chat. These actions target the clicked conversation without switching away from a draft or temporary chat. Deleting the active conversation opens a neighboring saved conversation or a fresh unsaved draft if history is empty. Failed history writes preserve the previous conversation/title.
- New chats stay as drafts until the first message is sent; empty drafts do not appear in history. Empty entries left by older releases are filtered on load and removed from the history file when it is next saved. Conversations with messages are preserved.
- Temporary chats keep conversation context in memory and never enter the history file. The empty conversation area displays the centered notice "Chat Not Saved to History. Cleared when you switch chats or close the app." Sending a message replaces the notice with the conversation. Switching to another chat or closing the app discards it. Visiting Train or Compare retains the current temporary chat until you switch chats. **Save Chat** (top right of the chat) keeps it: the chat joins the history and the sidebar, named from its first message unless you renamed it. Export is also available from the ⋯ menu when you only want a copy. Deleting a temporary chat discards it.
- The model list discovers compatible `.bin` models and checkpoints in `models/`, paired with their `.tok` files. Browse… in the picker can add a model elsewhere (and loads it). Refresh list discovers newly trained models.
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
- **Picking a GGUF without a runtime.** While no runtime is installed, GGUF rows in the model picker carry a **Needs runtime** tag. Choosing one opens a popup that offers to download the recommended runtime on the spot (with its size, and a note that this contacts github.com), or to open Settings. After a download the popup closes itself and the model you picked loads automatically. Escape and clicking outside do not close the popup while a download is running; **Cancel download** does. A default model that is a GGUF never opens the popup at launch; it explains in the status bar instead.
- **The runtimes list.** Tools → Settings → Runtimes starts with the **Trance engine** (marked *Built in*): whether its binary is found, where it is, and which GPU it uses once it has started. It is read-only, because there is nothing to download or choose. Below it is the llama.cpp runtime.
- **The runtime.** GGUF needs a `llama-server` program. Tools → Settings → Runtimes lists every runtime the app can use, each with its name, full path and size, an **In use** badge, **Open folder**, and **Use** or **Remove** (downloads) / **Stop using** (a path you chose, which is never deleted). Downloads are kept in `data/gui/runtimes/llama.cpp/<build>-<flavor>/` inside your project folder, and several can be installed side by side. The app looks, in order, for: a path you chose, a runtime it downloaded, a bundled one, then `llama-server` on your PATH. If none is found, **Download** fetches a pinned llama.cpp build (CPU, Vulkan, CUDA 12 or 13, or Metal depending on your system; the recommended one is preselected). Files come from the official `ggml-org/llama.cpp` GitHub releases and are checked against SHA-256 hashes built into the app before they are unpacked. CUDA builds include their CUDA runtime, so they are several hundred MB.
- **Privacy.** The only network use is that download, and only when you click it. Afterwards everything runs on your machine: the server listens on 127.0.0.1 only, on a random port, protected by a random per-launch API key, and it is stopped when you switch engines, start training, or close the app.
- **Behavior notes.** One model is loaded at a time and stays loaded until you Eject it, pick another model, start training, or close the app. Context is capped at 8,192 tokens (or the model's own limit if smaller). Sampling settings map to temperature, top-p, top-k, max tokens and seed; the server applies the model's own chat template. The context hint shows usage after each reply.
- **AppImage note.** The AppImage puts its own library folders (bundled Qt, Python and cuBLAS) on `LD_LIBRARY_PATH`; the llama.cpp server is started without those entries so it loads its own libraries, while anything you had set yourself is kept.
- **Maintenance.** `python tools/pin_llama_runtime.py` re-verifies the pinned hashes; with `--tag bNNNN --write` it moves the app to a newer llama.cpp build.

## Settings and themes

Tools → Settings lets you choose a color theme: Teal (default), Indigo or Amber. A theme changes only the accent colors; it applies immediately, including to open chats, and is saved in `data/gui/settings.json` as `theme`. An unknown saved value falls back to Teal. Palettes live in `gui/theme.py` (`PALETTES`); add an entry there and a card appears automatically on the Settings page.
