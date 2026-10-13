"""Integration tests for the application service (src/app/service.py).

Covers the HTTP-facing service without TensorFlow, audio, or the network: the
clip allow-list, preference validation, and the classify/step/quick-action flow
all run against a temporary catalog and an injected fake predictor.

Run with:  .venv\\Scripts\\python.exe -m unittest tests.test_app_service
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src import config
from src.app.service import (
    AppError,
    AppService,
    SampleCatalog,
    build_preferences,
)
from src.routing.supervisor import VerificationState


class FakePrediction:
    def __init__(self, label, confidence, top_k):
        self.label = label
        self.confidence = confidence
        self._top_k = list(top_k)

    def top_k(self, k=3):
        return self._top_k[:k]


class FakePredictor:
    def __init__(self, label="Siren", confidence=0.92):
        self.label = label
        self.confidence = confidence
        self.calls = []

    def predict_file(self, path):
        self.calls.append(str(path))
        return FakePrediction(
            self.label,
            self.confidence,
            [(self.label, self.confidence), ("Alarm", 0.05), ("Vehicle_Horn", 0.03)],
        )


def _write_csv(path, header, rows):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(cell) for cell in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_catalog(directory, per_class=1):
    split_csv = Path(directory) / "splits.csv"
    ledger_csv = Path(directory) / "ledger.csv"
    _write_csv(
        split_csv,
        ["clip_id", "split"],
        [("Siren#0", "val"), ("Alarm#0", "val"), ("Siren#train", "train")],
    )
    _write_csv(
        ledger_csv,
        [
            "clip_id",
            "status",
            "project_class",
            "source_dataset",
            "source_file",
            "local_path",
            "duration_s",
        ],
        [
            ("Siren#0", "kept", "Siren", "demo", "siren0.wav", "audio/siren0.wav", "3.5"),
            ("Alarm#0", "kept", "Alarm", "demo", "alarm0.wav", "audio/alarm0.wav", "2.0"),
            ("Siren#train", "kept", "Siren", "demo", "t.wav", "audio/t.wav", "1.0"),
            ("Dropped#0", "dropped", "Siren", "demo", "d.wav", "audio/d.wav", "1.0"),
        ],
    )
    return SampleCatalog(split_csv=split_csv, ledger_csv=ledger_csv, per_class=per_class)


def _valid_payload(**overrides):
    policy = {
        "enabled": True,
        "target_mode": config.MODE_SAFETY,
        "confidence_threshold": 0.5,
        "confirm_windows": 1,
        "release_seconds": 2.0,
        "cooldown_seconds": 0.0,
        "high_priority": True,
        "critical": False,
        "notify": True,
    }
    policy.update(overrides.pop("policy", {}))
    payload = {
        "profile_name": "Test",
        "preferred_mode": config.MODE_MAX_ANC,
        "policies": {"Siren": policy},
    }
    payload.update(overrides)
    return payload


class CatalogTests(unittest.TestCase):
    def test_only_validation_kept_clips_are_offered(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _make_catalog(tmp)
            ids = [clip.clip_id for clip in catalog.clips]
            self.assertIn("Siren#0", ids)
            self.assertIn("Alarm#0", ids)
            self.assertNotIn("Siren#train", ids)
            self.assertNotIn("Dropped#0", ids)

    def test_clip_to_dict_hides_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _make_catalog(tmp)
            view = catalog.clips[0].to_dict()
            self.assertNotIn("local_path", view)
            self.assertIn("clip_id", view)

    def test_unknown_clip_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _make_catalog(tmp)
            with self.assertRaises(AppError):
                catalog.get("Siren#train")

    def test_missing_input_files_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AppError):
                SampleCatalog(
                    split_csv=Path(tmp) / "nope.csv",
                    ledger_csv=Path(tmp) / "nope2.csv",
                )


class BuildPreferencesTests(unittest.TestCase):
    def test_valid_payload_builds(self):
        prefs = build_preferences(_valid_payload())
        self.assertEqual(prefs.preferred_mode, config.MODE_MAX_ANC)
        self.assertEqual(list(prefs.enabled_classes()), ["Siren"])
        policy = prefs.policy_for("Siren")
        self.assertEqual(policy.target_mode, config.MODE_SAFETY)
        self.assertTrue(policy.high_priority)

    def test_unknown_class_raises(self):
        with self.assertRaises(AppError):
            build_preferences({"policies": {"Nope": {"enabled": True}}})

    def test_unknown_field_raises(self):
        with self.assertRaises(AppError):
            build_preferences({"policies": {"Siren": {"bogus": 1}}})

    def test_wrong_type_raises(self):
        with self.assertRaises(AppError):
            build_preferences({"policies": {"Siren": {"enabled": "yes"}}})

    def test_non_mapping_policies_raises(self):
        with self.assertRaises(AppError):
            build_preferences({"policies": [1, 2, 3]})

    def test_absent_class_is_disabled(self):
        prefs = build_preferences({"policies": {}})
        self.assertEqual(list(prefs.enabled_classes()), [])


class AppServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.catalog = _make_catalog(self._tmp.name)
        self.predictor = FakePredictor()
        self.service = AppService(self.catalog, predictor_factory=lambda: self.predictor)

    def _first_clip(self, label):
        for clip in self.catalog.clips:
            if clip.label == label:
                return clip
        raise AssertionError(f"no clip for {label}")

    def test_config_exposes_interfaces(self):
        info = self.service.config()
        self.assertEqual(info["classes"], list(config.TRIGGER_CLASS_NAMES))
        self.assertEqual(info["modes"], list(config.AUDIO_MODES))
        self.assertTrue(info["clips"])
        self.assertIn("preferred_mode_default", info)
        self.assertIn("policy_defaults", info)

    def test_classify_returns_prediction_and_state(self):
        clip = self._first_clip("Siren")
        result = self.service.classify({"clip_id": clip.clip_id})
        self.assertEqual(result["prediction"]["label"], "Siren")
        self.assertAlmostEqual(result["prediction"]["confidence"], 0.92)
        self.assertEqual(result["prediction"]["top_k"][0][0], "Siren")
        self.assertNotIn("local_path", result["prediction"]["clip"])
        self.assertEqual(self.predictor.calls, [clip.local_path])

    def test_classify_unknown_clip_raises(self):
        with self.assertRaises(AppError):
            self.service.classify({"clip_id": "not-real"})

    def test_classify_missing_clip_id_raises(self):
        with self.assertRaises(AppError):
            self.service.classify({})

    def test_step_before_any_detection_raises(self):
        with self.assertRaises(AppError):
            self.service.step({})

    def test_full_trigger_and_restore_flow(self):
        self.service.apply_preferences(_valid_payload())
        clip = self._first_clip("Siren")
        self.service.classify({"clip_id": clip.clip_id})

        feed = self.service.step({})
        decision = feed["decision"]
        self.assertTrue(decision["verified"])
        self.assertEqual(decision["verification_state"], VerificationState.VERIFIED.value)
        self.assertEqual(feed["state"]["current_mode"], config.MODE_SAFETY)

        release = self.service.step({"label": None, "advance_seconds": 2.0})
        self.assertEqual(
            release["decision"]["verification_state"], VerificationState.RELEASED.value
        )
        self.assertTrue(release["decision"]["restored"])
        self.assertEqual(release["state"]["current_mode"], config.MODE_MAX_ANC)

    def test_advance_seconds_must_be_non_negative(self):
        with self.assertRaises(AppError):
            self.service.step({"advance_seconds": -1})

    def test_quick_action_requires_pending(self):
        with self.assertRaises(AppError):
            self.service.quick_action({"action": "confirm"})

    def test_quick_action_rejects_bad_action(self):
        with self.assertRaises(AppError):
            self.service.quick_action({"action": "maybe"})

    def test_quick_action_flow_executes_nothing(self):
        payload = _valid_payload(
            policy={"critical": True, "quick_action_label": "Call help"}
        )
        self.service.apply_preferences(payload)
        clip = self._first_clip("Siren")
        self.service.classify({"clip_id": clip.clip_id})
        feed = self.service.step({})
        self.assertTrue(feed["decision"]["quick_action_pending"])
        self.assertIsNotNone(feed["state"]["pending_quick_action"])

        result = self.service.quick_action({"action": "confirm"})
        self.assertFalse(result["result"]["executed"])
        self.assertTrue(result["result"]["confirmed"])
        self.assertIsNone(result["state"]["pending_quick_action"])


if __name__ == "__main__":
    unittest.main()
