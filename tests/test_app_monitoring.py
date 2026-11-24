"""Tests for dashboard timing/telemetry without TensorFlow or network.

Run with:  .venv\\Scripts\\python.exe -m unittest tests.test_app_monitoring
"""

from __future__ import annotations

import base64
import json
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection

from src.app.monitor import DashboardMonitor
from src.app.server import create_server
from src.app.service import AppError, AppService
from src.timing import TimingSink, activate, deactivate, stage
from tests.test_app_service import FakePredictor, _make_catalog, _valid_payload


class TimingTests(unittest.TestCase):
    def test_stage_uses_monotonic_clock_and_callback(self):
        sink = TimingSink(on_stage=lambda name: seen.append(name))
        seen: list[str] = []
        token = activate(sink)
        try:
            with stage("reception"):
                time.sleep(0.01)
        finally:
            deactivate(token)
        self.assertEqual(seen, ["reception"])
        self.assertGreaterEqual(sink.snapshot()["reception"], 0.005)

    def test_sinks_are_context_local(self):
        first = TimingSink()
        second = TimingSink()

        def in_other_thread():
            activate(second)
            with stage("b"):
                pass

        token = activate(first)
        try:
            with stage("a"):
                pass
            nested = threading.Thread(target=in_other_thread)
            nested.start()
            nested.join()
        finally:
            deactivate(token)
        self.assertIn("a", first.snapshot())
        self.assertNotIn("b", first.snapshot())
        self.assertIn("b", second.snapshot())
        self.assertNotIn("a", second.snapshot())


class MonitorTests(unittest.TestCase):
    def test_activity_and_history_are_per_request(self):
        monitor = DashboardMonitor()
        monitor.begin_init(TimingSink())
        monitor.finish_init(0.25, False)
        monitor.begin_request("one", "a.wav", "uploaded")
        monitor.begin_request("two", "b.wav", "validation_clip")
        monitor.phase("one", "embedding")
        monitor.finish_request("one", {"total_ms": 12.0, "file_name": "a.wav"})
        monitor.finish_request("two", {"total_ms": 20.0, "file_name": "b.wav"})
        self.assertEqual(monitor.activity("one")["track"]["status"], "completed")
        self.assertEqual(monitor.activity("two")["track"]["file_name"], "b.wav")
        stats = monitor.history()["stats"]
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["avg_ms"], 16.0)
        self.assertEqual(stats["min_ms"], 12.0)
        self.assertEqual(stats["max_ms"], 20.0)
        self.assertEqual(stats["init_ms"], 250.0)

    def test_failed_request_is_recorded(self):
        monitor = DashboardMonitor()
        monitor.begin_request("bad", "bad.wav", "uploaded")
        monitor.fail_request("bad", "AppError: invalid WAV file")
        self.assertEqual(monitor.activity("bad")["track"]["status"], "failed")
        self.assertEqual(monitor.history()["stats"]["count"], 0)


class ServiceMonitoringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.catalog = _make_catalog(self._tmp.name)
        self.service = AppService(
            self.catalog, predictor_factory=lambda: FakePredictor()
        )

    def test_classify_reports_measured_timing(self):
        clip = self.catalog.clips[0]
        result = self.service.classify({
            "clip_id": clip.clip_id, "request_id": "req-ok",
        })
        timing = result["timing"]
        self.assertEqual(timing["timer"], "time.perf_counter")
        self.assertGreater(timing["total_ms"], 0)
        self.assertIn("reception", timing["stages_ms"])
        self.assertIn("aggregation", timing["stages_ms"])
        self.assertIn("other_overhead", timing["stages_ms"])
        self.assertEqual(result["request_id"], "req-ok")
        history = self.service.monitor.history()
        self.assertEqual(history["stats"]["count"], 1)
        self.assertEqual(history["history"][0]["file_name"], clip.file_name)

    def test_failed_upload_marks_activity_failed(self):
        blob = base64.b64encode(b"not a wav").decode("ascii")
        with self.assertRaises(AppError):
            self.service.classify_upload({
                "file_name": "bad.wav", "data": blob, "request_id": "req-bad",
            })
        track = self.service.monitor.activity("req-bad")["track"]
        self.assertEqual(track["status"], "failed")
        self.assertIn("invalid WAV file", track["error"])


class ServerMonitoringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        catalog = _make_catalog(self._tmp.name)
        self.service = AppService(
            catalog, predictor_factory=lambda: FakePredictor()
        )
        self.service.monitor.begin_init(TimingSink())
        self.service.monitor.finish_init(0.5, False)
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
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, body=payload,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def test_model_activity_history_endpoints(self):
        status, body = self._request("GET", "/api/model")
        self.assertEqual(status, 200)
        info = json.loads(body)
        self.assertEqual(info["status"], "ready")
        self.assertEqual(info["total_ms"], 500.0)

        # An already ready model must not pretend to reload.
        status, body = self._request("POST", "/api/model/init", {})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["attempts"], 1)

        status, body = self._request("GET", "/api/activity?request_id=none")
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body)["track"])

        status, body = self._request("GET", "/api/history")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["stats"]["init_ms"], 500.0)

    def test_classify_response_has_timing_and_history_endpoint(self):
        clip = self.service.catalog.clips[0]
        status, body = self._request("POST", "/api/classify", {
            "clip_id": clip.clip_id, "request_id": "req-http",
        })
        self.assertEqual(status, 200)
        self.assertIn("timing", json.loads(body))
        status, body = self._request("GET", "/api/history")
        self.assertEqual(json.loads(body)["stats"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
