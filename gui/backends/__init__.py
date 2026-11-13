"""Chat backends. Each exposes the same surface as gui.engine.EngineClient:

    generate(model, messages, settings), cancel(), release(), close(), busy
    signals: delta(str), started(dict), finished(dict), failed(str),
             state_changed(str), ready(str), loaded(dict)

so the window can talk to any of them (or to the router that picks one).
"""
