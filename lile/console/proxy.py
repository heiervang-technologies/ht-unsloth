"""Tiny stdlib-only proxy: serves demo.html at /, forwards /api/* to lile on 8765.

Run with system Python — no venv, no deps.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
HTML = HERE / "demo.html"
DASHBOARD = HERE / "dashboard.html"
METRICS = HERE / "metrics.html"
UPSTREAM = "http://127.0.0.1:" + os.environ.get("LILE_PORT", "8768")
PORT = int(os.environ.get("LILE_PROXY_PORT", "8766"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[proxy] %s — %s\n" % (self.address_string(), fmt % args))

    def _serve_file(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, method: str):
        upstream_path = self.path[len("/api"):] or "/"
        url = UPSTREAM + upstream_path
        length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(
            url, data=body, method=method,
            headers={"content-type": self.headers.get("content-type", "application/json")},
        )
        try:
            r = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            data = e.read()
            ctype = e.headers.get("content-type", "application/json")
            self.send_response(e.code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data); return
        except Exception as e:
            data = json.dumps({"error": f"proxy upstream failure: {e}"}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data); return

        ctype = r.headers.get("content-type", "application/json")
        if ctype.startswith("text/event-stream"):
            # Streaming pass-through — no content-length, flush every chunk
            # so the browser's EventSource sees tokens in real time.
            self.send_response(r.status)
            self.send_header("content-type", ctype)
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.send_header("x-accel-buffering", "no")
            self.end_headers()
            try:
                while True:
                    chunk = r.read1(4096) if hasattr(r, "read1") else r.read(256)
                    if not chunk: break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                r.close()
            return

        status = r.status
        data = r.read()
        r.close()
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(HTML); return
        if self.path == "/dashboard" or self.path == "/dashboard.html":
            self._serve_file(DASHBOARD); return
        if self.path == "/metrics" or self.path == "/metrics.html":
            self._serve_file(METRICS); return
        if self.path.startswith("/api/") or self.path == "/api":
            self._proxy("GET"); return
        self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/api/"): self._proxy("POST"); return
        self.send_error(404)


def main():
    print(f"[proxy] serving {HTML} at http://127.0.0.1:{PORT}/")
    print(f"[proxy] forwarding /api/* -> {UPSTREAM}/*")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
