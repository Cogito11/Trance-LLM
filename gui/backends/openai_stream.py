"""Streaming client for OpenAI-compatible /v1/chat/completions endpoints (stdlib only).

Used for the managed llama.cpp server today and for connections to LM Studio,
Ollama or any compatible server later. It runs in its own thread so the UI
never blocks on the network.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit

from PySide6.QtCore import QThread, Signal


def build_payload(model: str, messages: list[dict], settings: dict) -> dict:
    """Map the app's sampling settings onto the request body."""
    payload = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    for source, target in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                           ("max_new_tokens", "max_tokens"), ("seed", "seed")):
        if settings.get(source) is not None:
            payload[target] = settings[source]
    return payload


def _error_text(body: bytes, status: int) -> str:
    try:
        error = json.loads(body).get("error", {})
        message = error.get("message") if isinstance(error, dict) else str(error)
        if message: return f"The server rejected the request ({status}): {message}"
    except (ValueError, AttributeError):
        pass
    return f"The server answered with HTTP {status}."


class ChatStream(QThread):
    """One streamed completion. Emits started, delta* and then exactly one of done or failed."""
    started_stream = Signal()
    delta = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, url: str, payload: dict, headers: dict | None = None, *, timeout: float = 3600, parent=None):
        super().__init__(parent)
        self.url, self.payload, self.timeout = url, payload, timeout
        self.headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **(headers or {})}
        self._cancelled = threading.Event()
        self._conn: http.client.HTTPConnection | None = None

    def cancel(self):
        self._cancelled.set()
        conn = self._conn
        if conn is not None and conn.sock is not None:
            try: conn.sock.shutdown(socket.SHUT_RDWR)   # unblocks a read waiting for the next token
            except OSError: pass

    def run(self):
        parts = urlsplit(self.url)
        factory = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        started, tokens, reason, usage_tokens, prompt_tokens = time.monotonic(), 0, "stop", None, None
        try:
            self._conn = conn = factory(parts.hostname, parts.port, timeout=self.timeout)
            path = parts.path + (f"?{parts.query}" if parts.query else "")
            conn.request("POST", path or "/", body=json.dumps(self.payload), headers=self.headers)
            response = conn.getresponse()
            if response.status != 200:
                self.failed.emit(_error_text(response.read(8192), response.status)); return
            self.started_stream.emit()
            while True:
                line = response.readline()
                if not line: break
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: chunk = json.loads(data)
                except ValueError: continue
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens", usage_tokens)
                    prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        tokens += 1; self.delta.emit(text)
                    if choice.get("finish_reason"): reason = choice["finish_reason"]
            if self._cancelled.is_set(): reason = "cancelled"
            self.done.emit({"reason": "length" if reason == "length" else reason if reason == "cancelled" else "stop",
                            "generated_tokens": usage_tokens if usage_tokens is not None else tokens,
                            "elapsed_seconds": time.monotonic() - started, "prompt_tokens": prompt_tokens})
        except (OSError, http.client.HTTPException) as error:
            if self._cancelled.is_set():
                self.done.emit({"reason": "cancelled", "generated_tokens": tokens, "elapsed_seconds": time.monotonic() - started})
            elif isinstance(error, ConnectionRefusedError):
                self.failed.emit("Could not connect to the server. Is it running?")
            else:
                self.failed.emit(f"The connection to the server failed: {error}")
        finally:
            if self._conn is not None:
                try: self._conn.close()
                except OSError: pass
