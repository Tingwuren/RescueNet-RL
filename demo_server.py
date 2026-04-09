"""Standalone demo server using only the Python standard library.

Serves the mahimahi simulation API so the frontend can be previewed
without installing FastAPI or other project dependencies.

Usage:  python demo_server.py
Then:   cd frontend && npm run dev
"""

import http.server
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.mahimahi_manager import MahimahiManager

manager = MahimahiManager()

PORT = 8000


class DemoHandler(http.server.BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/api/health":
            self._json_response({"status": "ok"})
        elif path == "/api/mahimahi/status":
            self._json_response({
                "mahimahi_available": manager.mahimahi_available,
                "traces_dir": str(manager.traces_dir),
            })
        elif path == "/api/mahimahi/traces":
            self._json_response({"traces": manager.list_traces()})
        elif path.startswith("/api/mahimahi/traces/"):
            name = path.split("/")[-1]
            try:
                data = manager.analyze_trace(name, duration_s=60, window_ms=500)
                self._json_response(data)
            except FileNotFoundError:
                self._json_response({"detail": "Trace '%s' not found" % name}, 404)
        elif path == "/api/scenarios":
            self._json_response({"scenarios": []})
        else:
            self._json_response({"detail": "Not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        if path == "/api/mahimahi/simulate":
            try:
                result = manager.simulate(
                    trace_name=body.get("trace_name", ""),
                    duration_s=body.get("duration_s", 60),
                    rtt_ms=body.get("rtt_ms", 80),
                    buffer_packets=body.get("buffer_packets", 100),
                    window_ms=body.get("window_ms", 500),
                )
                self._json_response(result)
            except FileNotFoundError:
                self._json_response({"detail": "Trace not found"}, 404)
            except Exception as e:
                self._json_response({"detail": str(e)}, 500)
        else:
            self._json_response({"detail": "Not found"}, 404)

    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def log_message(self, fmt, *args):
        sys.stderr.write("[API] %s\n" % args[0])
        sys.stderr.flush()


if __name__ == "__main__":
    sys.stderr.write("Demo API server on http://localhost:%d\n" % PORT)
    sys.stderr.write("Traces: %s\n" % [t["name"] for t in manager.list_traces()])
    sys.stderr.flush()
    server = http.server.HTTPServer(("0.0.0.0", PORT), DemoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
