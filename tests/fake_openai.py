"""A tiny OpenAI-compatible chat server for tests (streams chunked SSE like llama-server)."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    token = None
    pieces = ["Hello", " from", " the", " fake", " server", "."]
    delay = 0.0
    seen: list = []

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _chunk(self, text: str):
        data = text.encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n"); self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._json(200, {"data": [{"id": "fake-model"}]})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        type(self).seen.append({"headers": dict(self.headers), "body": body})
        if self.token and self.headers.get("Authorization") != f"Bearer {self.token}":
            return self._json(401, {"error": {"message": "Invalid API Key"}})
        if body.get("model") == "boom":
            return self._json(400, {"error": {"message": "context too long"}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        limit = body.get("max_tokens") or len(self.pieces)
        endless = limit >= 1000          # tests use a huge limit to get a stream that must be cancelled
        count = limit if endless else min(limit, len(self.pieces))
        reason = "length" if endless or limit < len(self.pieces) else "stop"
        try:
            self._chunk('data: ' + json.dumps({"choices": [{"delta": {"role": "assistant", "content": None}}]}) + "\n\n")
            for n in range(count):
                time.sleep(self.delay)
                self._chunk('data: ' + json.dumps({"choices": [{"delta": {"content": self.pieces[n % len(self.pieces)]}}]}) + "\n\n")
            self._chunk('data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": reason}]}) + "\n\n")
            self._chunk('data: ' + json.dumps({"choices": [], "usage": {"completion_tokens": count}}) + "\n\n")
            self._chunk("data: [DONE]\n\n"); self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def start(token=None, delay=0.0, port=0):
    """Start a server on a background thread; returns (server, port). Call server.shutdown() when done."""
    handler = type("H", (Handler,), {"token": token, "delay": delay, "seen": []})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]
