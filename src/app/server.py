"""Local HTTP server exposing the supervisor interface and its JSON API.

Built only on the Python standard library (`http.server`): no web framework is
added. It serves three static assets and four JSON endpoints, all backed by
`src.app.service.AppService`:

  GET  /                 the single-page interface (index.html)
  GET  /app.css          stylesheet
  GET  /app.js           interface script
  GET  /api/config       classes, modes, bundled clips, policy defaults
  POST /api/preferences  validate + apply a policy set (resets the session)
  POST /api/classify     run the real predictor on one allow-listed clip
  POST /api/step         feed one detection (or none) through the supervisor
  POST /api/quick-action confirm/reject the pending quick action

The server binds to 127.0.0.1 by default and is intended for local, single-user
use. Requests and responses are ASCII-encoded JSON; AppError becomes HTTP 400
with an explicit message.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.app.service import AppError, AppService, SampleCatalog

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_FILES: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 1_000_000


def _make_handler(service: AppService) -> type[BaseHTTPRequestHandler]:
    """Build the request handler bound to one AppService instance."""

    post_routes = {
        "/api/preferences": service.apply_preferences,
        "/api/classify": service.classify,
        "/api/step": service.step,
        "/api/quick-action": service.quick_action,
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "EdgeAI-Acoustic-Supervisor/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the default per-request stderr logging."""
            return

        def _send_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, name: str) -> None:
            path = WEB_DIR / name
            if not path.is_file():
                self._send_json(404, {"error": f"missing asset: {name}"})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", STATIC_FILES[name])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            if length > MAX_BODY_BYTES:
                raise AppError("request body too large")
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                raise AppError(f"could not parse JSON body: {exc}") from exc
            if not isinstance(data, dict):
                raise AppError("JSON body must be an object")
            return data

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            try:
                if path in ("/", "/index.html"):
                    self._send_static("index.html")
                elif path == "/app.css":
                    self._send_static("app.css")
                elif path == "/app.js":
                    self._send_static("app.js")
                elif path == "/api/config":
                    self._send_json(200, service.config())
                else:
                    self._send_json(404, {"error": f"not found: {path}"})
            except AppError as exc:
                self._send_json(400, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            handler = post_routes.get(path)
            if handler is None:
                self._send_json(404, {"error": f"not found: {path}"})
                return
            try:
                payload = self._read_json()
                result = handler(payload)
            except AppError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                self._send_json(
                    500, {"error": f"internal error: {type(exc).__name__}: {exc}"}
                )
            else:
                self._send_json(200, result)

    return Handler


def create_server(
    service: AppService,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Create (but do not start) a threaded HTTP server for the interface."""
    httpd = ThreadingHTTPServer((host, port), _make_handler(service))
    httpd.daemon_threads = True
    return httpd


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the service and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port")
    args = parser.parse_args(argv)

    catalog = SampleCatalog()
    service = AppService(catalog)
    httpd = create_server(service, args.host, args.port)
    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"Edge AI Acoustic Supervisor interface: http://{host}:{port}")
    print(f"bundled validation clips: {len(catalog.clips)}")
    print("Local use only. SIMULATED: no live mic, no earbud control, and a "
          "confirmed quick action executes nothing.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped.")
    finally:
        httpd.server_close()
    return 0
