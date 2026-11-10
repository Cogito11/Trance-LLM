# Creating a training config

Open **Configs** in the sidebar. Choose a starter preset or load an existing JSON config as a template, give the new config a name, and edit the fields.

## Model tab

Vocabulary, context length, width, layers, attention heads and feed-forward width define the architecture. Heads must divide the model width. Trance caps vocabulary and context at 1,024. Changing architecture generally requires fresh training rather than continuing an existing checkpoint.

## Training tab

Choose batch size, optimizer steps, learning rate, gradient clipping, validation interval, checkpoint interval and seed. A checkpoint interval of zero disables periodic saves; stopping from the GUI still saves an interrupted checkpoint.

The Trance 1.2 architecture preset uses a 50,000-step starting budget. This is a convenient experiment setting, not a guarantee of the best stopping point. Watch validation and compare checkpoint outputs.

## Data and output tab

Choose existing conversation files. Paths can be relative to the selected project or absolute. Multiple input files are comma-separated. Keep validation examples separate. Pick a new output-model path so existing models remain protected.

An optional resume file must contain optimizer state and have a matching architecture. On resume, training_steps means additional updates.

**Save config** writes a new JSON file under the project's configs folder and does not replace an existing config. **Open in Train** selects it for review. Training begins only when you click **Start training**.

The parameter count and weights-only size are estimates of model storage, not total GPU memory requirements. See [GPU memory](article://hardware).
