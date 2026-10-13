"""Unit tests for the Phase 5 preference layer (src/routing/preferences.py).

Run with:  .venv\\Scripts\\python.exe -m unittest tests.test_routing_preferences
No model, audio, or dataset is loaded.
"""

from __future__ import annotations

import unittest

from src import config
from src.routing import preferences as prefs


class DisabledPreferencesTests(unittest.TestCase):
    def test_covers_full_taxonomy_and_is_valid(self):
        prefs_set = prefs.disabled_preferences()
        self.assertEqual(len(prefs_set.policies), config.NUM_TRIGGER_CLASSES)
        self.assertEqual(sorted(prefs_set.policies), sorted(config.TRIGGER_CLASS_NAMES))
        self.assertEqual(prefs_set.validate(), [])

    def test_exports_nothing_enabled(self):
        prefs_set = prefs.disabled_preferences()
        self.assertEqual(prefs_set.enabled_classes(), ())
        for name in config.TRIGGER_CLASS_NAMES:
            entry = prefs_set.policy_for(name)
            self.assertIsNotNone(entry)
            self.assertFalse(entry.enabled)
            self.assertIsNone(entry.target_mode)

    def test_policy_for_unknown_class_is_none(self):
        prefs_set = prefs.disabled_preferences()
        self.assertIsNone(prefs_set.policy_for("Chainsaw"))


class TriggerPolicyValidationTests(unittest.TestCase):
    def test_unknown_class_is_flagged(self):
        problems = prefs.TriggerPolicy(class_name="Chainsaw").validate()
        self.assertTrue(any("unknown trigger class" in p for p in problems))

    def test_invalid_target_mode_is_flagged(self):
        entry = prefs.TriggerPolicy(class_name="Siren", target_mode="Loud Mode")
        self.assertTrue(any("target_mode" in p for p in entry.validate()))

    def test_none_target_mode_is_allowed(self):
        entry = prefs.TriggerPolicy(class_name="Siren", target_mode=None)
        self.assertEqual(entry.validate(), [])

    def test_threshold_bounds_are_enforced(self):
        for bad in (0.0, -0.1, 1.5, "high"):
            entry = prefs.TriggerPolicy(
                class_name="Siren", confidence_threshold=bad  # type: ignore[arg-type]
            )
            self.assertTrue(
                any("confidence_threshold" in p for p in entry.validate()),
                msg=f"expected rejection of threshold {bad!r}",
            )

    def test_confirm_windows_must_be_positive_int(self):
        for bad in (0, -1, 2.5, True):
            entry = prefs.TriggerPolicy(
                class_name="Siren", confirm_windows=bad  # type: ignore[arg-type]
            )
            self.assertTrue(
                any("confirm_windows" in p for p in entry.validate()),
                msg=f"expected rejection of confirm_windows {bad!r}",
            )

    def test_negative_durations_are_flagged(self):
        entry = prefs.TriggerPolicy(
            class_name="Siren", release_seconds=-1.0, cooldown_seconds=-2.0
        )
        problems = entry.validate()
        self.assertTrue(any("release_seconds" in p for p in problems))
        self.assertTrue(any("cooldown_seconds" in p for p in problems))

    def test_priority_flags_require_enabled(self):
        entry = prefs.TriggerPolicy(class_name="Siren", high_priority=True)
        self.assertTrue(any("only to an enabled entry" in p for p in entry.validate()))

    def test_critical_requires_label(self):
        entry = prefs.TriggerPolicy(
            class_name="Glass_Breaking", enabled=True, critical=True
        )
        self.assertTrue(any("quick_action_label" in p for p in entry.validate()))

    def test_label_requires_critical(self):
        entry = prefs.TriggerPolicy(
            class_name="Siren", enabled=True, quick_action_label="Call 112"
        )
        self.assertTrue(any("only valid when" in p for p in entry.validate()))


class UserPreferencesTests(unittest.TestCase):
    def test_build_raises_on_any_problem(self):
        bad = prefs.TriggerPolicy(class_name="Nope", enabled=True)
        with self.assertRaises(prefs.PreferenceError):
            prefs.UserPreferences.build([bad])

    def test_build_accepts_valid_set(self):
        good = prefs.TriggerPolicy(
            class_name="Siren", enabled=True, target_mode=config.MODE_SAFETY
        )
        prefs_set = prefs.UserPreferences.build([good], profile_name="Commute")
        self.assertEqual(prefs_set.profile_name, "Commute")
        self.assertEqual(prefs_set.enabled_classes(), ("Siren",))

    def test_enabled_classes_follow_taxonomy_order(self):
        policies = [
            prefs.TriggerPolicy(
                class_name=name, enabled=True, target_mode=config.MODE_MAX_ANC
            )
            for name in ("Vehicle_Horn", "Aircraft", "Siren")
        ]
        prefs_set = prefs.UserPreferences.build(policies)
        ordered = tuple(
            n for n in config.TRIGGER_CLASS_NAMES
            if n in {"Vehicle_Horn", "Aircraft", "Siren"}
        )
        self.assertEqual(prefs_set.enabled_classes(), ordered)

    def test_invalid_preferred_mode_is_flagged(self):
        prefs_set = prefs.UserPreferences(policies={}, preferred_mode="Lounge Mode")
        self.assertTrue(any("preferred_mode" in p for p in prefs_set.validate()))

    def test_build_raises_on_invalid_preferred_mode(self):
        with self.assertRaises(prefs.PreferenceError):
            prefs.UserPreferences.build([], preferred_mode="Lounge Mode")

    def test_policies_mapping_is_immutable(self):
        prefs_set = prefs.disabled_preferences()
        with self.assertRaises(TypeError):
            prefs_set.policies["Siren"] = prefs.TriggerPolicy(  # type: ignore[index]
                class_name="Siren", enabled=True
            )

    def test_key_class_name_mismatch_is_flagged(self):
        prefs_set = prefs.UserPreferences(
            policies={"Siren": prefs.TriggerPolicy(class_name="Alarm")}
        )
        self.assertTrue(any("does not match" in p for p in prefs_set.validate()))

    def test_from_class_modes_enables_only_named_classes(self):
        prefs_set = prefs.UserPreferences.from_class_modes(
            {"Siren": config.MODE_SAFETY, "Baby_Crying": config.MODE_TRANSPARENCY},
            profile_name="Night",
            preferred_mode=config.MODE_MAX_ANC,
        )
        self.assertEqual(prefs_set.enabled_classes(), ("Baby_Crying", "Siren"))
        self.assertEqual(
            prefs_set.policy_for("Siren").target_mode, config.MODE_SAFETY
        )
        self.assertFalse(prefs_set.policy_for("Alarm").enabled)
        self.assertEqual(prefs_set.validate(), [])

    def test_from_class_modes_rejects_unknown_class(self):
        with self.assertRaises(prefs.PreferenceError):
            prefs.UserPreferences.from_class_modes(
                {"Chainsaw": config.MODE_MAX_ANC}
            )


if __name__ == "__main__":
    unittest.main()
