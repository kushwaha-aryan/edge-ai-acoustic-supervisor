"""HTTP tests for the local server (src/app/server.py).

Starts the real threaded server on an ephemeral port with a temporary catalog
and a fake predictor, and drives it over HTTP. No TensorFlow, audio, or external
network is involved.

Run with:  .venv\\Scripts\\python.exe -m unittest tests.test_app_server
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest

from src.app.server import create_server
from src.app.service import AppService
from tests.test_app_service import FakePredictor, _make_catalog, _valid_payload


class ServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.catalog = _make_catalog(self._tmp.name)
        self.service = AppService(
            self.catalog, predictor_factory=lambda: FakePredictor()
        )
        self.httpd = create_server(self.service, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def _request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {}
        payload = None
        if body is not None:
            payload = json.dumps(body)
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def _request_raw(self, method, path, raw):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            method, path, body=raw, headers={"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def _clip_id(self, label):
        for clip in self.catalog.clips:
            if clip.label == label:
                return clip.clip_id
        raise AssertionError(f"no clip for {label}")

    def test_index_served(self):
        status, body = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Edge AI Acoustic Supervisor", body)

    def test_config_endpoint(self):
        status, body = self._request("GET", "/api/config")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["classes"])
        self.assertTrue(data["modes"])

    def test_unknown_get_is_404(self):
        status, _ = self._request("GET", "/nope")
        self.assertEqual(status, 404)

    def test_unknown_post_is_404(self):
        status, _ = self._request("POST", "/api/nope", {})
        self.assertEqual(status, 404)

    def test_apply_preferences_ok(self):
        status, body = self._request("POST", "/api/preferences", _valid_payload())
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["state"]["enabled_classes"], ["Siren"])

    def test_apply_preferences_bad_class_is_400(self):
        status, body = self._request(
            "POST", "/api/preferences", {"policies": {"Nope": {"enabled": True}}}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_malformed_body_is_400(self):
        status, body = self._request_raw("POST", "/api/preferences", "not json")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_classify_ok(self):
        status, body = self._request(
            "POST", "/api/classify", {"clip_id": self._clip_id("Siren")}
        )
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["prediction"]["label"], "Siren")

    def test_classify_unknown_clip_is_400(self):
        status, body = self._request(
            "POST", "/api/classify", {"clip_id": "not-real"}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_step_before_detection_is_400(self):
        status, _ = self._request("POST", "/api/step", {})
        self.assertEqual(status, 400)

    def test_full_http_flow(self):
        self._request("POST", "/api/preferences", _valid_payload())
        self._request("POST", "/api/classify", {"clip_id": self._clip_id("Siren")})
        status, body = self._request("POST", "/api/step", {})
        self.assertEqual(status, 200)
        decision = json.loads(body)["decision"]
        self.assertTrue(decision["verified"])
        self.assertEqual(decision["audio_mode"], "Safety Transparency Mode")

    def test_quick_action_without_pending_is_400(self):
        status, _ = self._request(
            "POST", "/api/quick-action", {"action": "confirm"}
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
