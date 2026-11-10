# The GUI and background processes

Trance LLM is a local desktop application. Python and PySide6 control the interface; Qt draws the widgets, handles input and paints the loss chart.

## Training

The GUI launches a detached Python supervisor, which launches the compiled **trance train** program. The native CUDA program performs training. The supervisor records progress events and run status in the project's data/gui/runs folder.

Closing the window leaves that training pair running. Reopening the same project reconnects to the run and reads its accumulated logs. Click Stop & save checkpoint when you deliberately want to stop the native trainer.

## Chat

For replies, the GUI launches a separate **trance engine** process. The two programs communicate through JSON-lines messages. Token bytes stream back and are decoded into text. An inference process may keep one model loaded between replies.

Chat generation and comparison are disabled during a GUI-managed training run to reserve GPU resources. Reading documentation and preparing another config remain available; saved configs do not alter the running job's immutable snapshot.

## Portable application

The AppImage bundles the application and runtime libraries. Models, configs, data and histories live in the writable project folder outside it. Use that same project when changing GUI versions so the app can find existing conversations and run records.
