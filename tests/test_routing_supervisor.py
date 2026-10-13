"""Unit tests for the Phase 5 supervisor (src/routing/supervisor.py).

Uses synthetic predictions only (no audio, model, or dataset). Time is passed
explicitly so release/cooldown behavior is deterministic and no sleep is used.

Run with:  .venv\\Scripts\\python.exe -m unittest tests.test_routing_supervisor
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from src import config
from src.routing import supervisor as sup


@dataclass
class FakePrediction:
    label: str
    confidence: float


def siren_policy(**overrides):
    base = dict(
        class_name="Siren",
        enabled=True,
        target_mode=config.MODE_SAFETY,
        confidence_threshold=0.80,
        confirm_windows=1,
        release_seconds=5.0,
        cooldown_seconds=config.RESPONSE_COOLDOWN_SECONDS,
    )
    base.update(overrides)
    from src.routing.preferences import TriggerPolicy

    return TriggerPolicy(**base)


def make_supervisor(*policies, preferred_mode=config.MODE_MAX_ANC, profile="Test"):
    from src.routing.preferences import UserPreferences

    prefs = UserPreferences.build(
        list(policies), profile_name=profile, preferred_mode=preferred_mode
    )
    return sup.Supervisor(prefs)


class GateTests(unittest.TestCase):
    def test_disabled_class_produces_no_action(self):
        from src.routing.preferences import TriggerPolicy

        disabled = TriggerPolicy(class_name="Siren")
        engine = make_supervisor(disabled)
        decision = engine.update(FakePrediction("Siren", 0.99), now=0.0)
        self.assertFalse(decision.trigger_enabled)
        self.assertFalse(decision.verified)
        self.assertEqual(decision.verification_state, sup.VerificationState.IDLE)
        self.assertIsNone(decision.notification)
        self.assertEqual(decision.audio_mode, config.MODE_MAX_ANC)

    def test_unknown_class_produces_no_action(self):
        engine = make_supervisor(siren_policy())
        decision = engine.update(FakePrediction("Chainsaw", 0.99), now=0.0)
        self.assertFalse(decision.trigger_enabled)
        self.assertFalse(decision.verified)
        self.assertTrue(any("not a supported trigger class" in r for r in decision.reasons))

    def test_low_confidence_produces_no_action(self):
        engine = make_supervisor(siren_policy(confidence_threshold=0.9))
        decision = engine.update(FakePrediction("Siren", 0.5), now=0.0)
        self.assertFalse(decision.verified)
        self.assertTrue(any("below threshold" in r for r in decision.reasons))


class PersistenceTests(unittest.TestCase):
    def test_confirm_windows_three_fires_on_third_window(self):
        engine = make_supervisor(siren_policy(confirm_windows=3))
        first = engine.update(FakePrediction("Siren", 0.9), now=0.0)
        second = engine.update(FakePrediction("Siren", 0.9), now=1.0)
        third = engine.update(FakePrediction("Siren", 0.9), now=2.0)
        self.assertEqual(first.verification_state, sup.VerificationState.PENDING)
        self.assertEqual(second.verification_state, sup.VerificationState.PENDING)
        self.assertTrue(third.verified)
        self.assertEqual(third.verification_state, sup.VerificationState.VERIFIED)

    def test_confirm_windows_one_fires_immediately(self):
        engine = make_supervisor(siren_policy(confirm_windows=1))
        decision = engine.update(FakePrediction("Siren", 0.9), now=0.0)
        self.assertTrue(decision.verified)

    def test_interruption_resets_persistence(self):
        engine = make_supervisor(siren_policy(confirm_windows=3))
        engine.update(FakePrediction("Siren", 0.9), now=0.0)
        engine.update(FakePrediction("Siren", 0.9), now=1.0)
        engine.update(FakePrediction("Dog_Bark", 0.1), now=2.0)
        decision = engine.update(FakePrediction("Siren", 0.9), now=3.0)
        self.assertEqual(decision.verification_state, sup.VerificationState.PENDING)


class ResponseTypeTests(unittest.TestCase):
    def test_mode_response_depends_on_preference(self):
        engine = make_supervisor(siren_policy(target_mode=config.MODE_MAX_ANC))
        decision = engine.update(FakePrediction("Siren", 0.95), now=0.0)
        self.assertEqual(decision.audio_mode, config.MODE_MAX_ANC)

    def test_other_preference_same_class_different_mode(self):
        engine = make_supervisor(siren_policy(target_mode=config.MODE_STANDBY))
        decision = engine.update(FakePrediction("Siren", 0.95), now=0.0)
        self.assertEqual(decision.audio_mode, config.MODE_STANDBY)

    def test_none_target_mode_keeps_mode_but_still_notifies(self):
        engine = make_supervisor(siren_policy(target_mode=None))
        decision = engine.update(FakePrediction("Siren", 0.95), now=0.0)
        self.assertEqual(decision.audio_mode, config.MODE_MAX_ANC)
        self.assertEqual(decision.notification, "Siren detected")

    def test_notification_wording_includes_mode(self):
        engine = make_supervisor(
            siren_policy(class_name="Baby_Crying", target_mode=config.MODE_TRANSPARENCY)
        )
        decision = engine.update(FakePrediction("Baby_Crying", 0.95), now=0.0)
        self.assertEqual(
            decision.notification,
            "Baby Crying detected - Transparency Mode enabled",
        )

    def test_high_priority_emits_warning(self):
        engine = make_supervisor(siren_policy(high_priority=True))
        decision = engine.update(FakePrediction("Siren", 0.95), now=0.0)
        self.assertEqual(decision.warning, "[WARNING] Siren detected")


class QuickActionTests(unittest.TestCase):
    def test_critical_presents_but_never_executes(self):
        policy = siren_policy(critical=True, quick_action_label="Call 112")
        engine = make_supervisor(policy)
        decision = engine.update(FakePrediction("Siren", 0.95), now=0.0)
        self.assertEqual(decision.quick_action, "Call 112")
        self.assertTrue(decision.quick_action_pending)
        result = engine.confirm_quick_action(now=1.0)
        self.assertTrue(result.confirmed)
        self.assertFalse(result.executed)
        self.assertIsNone(engine.pending_quick_action)

    def test_reject_clears_pending(self):
        policy = siren_policy(critical=True, quick_action_label="Call 112")
        engine = make_supervisor(policy)
        engine.update(FakePrediction("Siren", 0.95), now=0.0)
        result = engine.reject_quick_action(now=1.0)
        self.assertFalse(result.confirmed)
        self.assertFalse(result.executed)
        self.assertIsNone(engine.pending_quick_action)

    def test_confirm_without_pending_raises(self):
        engine = make_supervisor(siren_policy())
        with self.assertRaises(sup.NoPendingQuickActionError):
            engine.confirm_quick_action(now=0.0)


class ReleaseAndCooldownTests(unittest.TestCase):
    def test_release_restores_preferred_mode_after_period(self):
        engine = make_supervisor(
            siren_policy(confirm_windows=1, release_seconds=5.0),
            preferred_mode=config.MODE_MAX_ANC,
        )
        engine.update(FakePrediction("Siren", 0.95), now=0.0)
        mid = engine.update(FakePrediction("Silence", 0.0), now=3.0)
        self.assertFalse(mid.restored)
        self.assertTrue(mid.release_timer_active)
        self.assertEqual(mid.audio_mode, config.MODE_SAFETY)
        end = engine.update(FakePrediction("Silence", 0.0), now=5.0)
        self.assertTrue(end.restored)
        self.assertEqual(end.verification_state, sup.VerificationState.RELEASED)
        self.assertEqual(end.audio_mode, config.MODE_MAX_ANC)

    def test_duplicate_alert_suppressed_by_cooldown(self):
        engine = make_supervisor(
            siren_policy(confirm_windows=1, release_seconds=1.0, cooldown_seconds=10.0)
        )
        first = engine.update(FakePrediction("Siren", 0.95), now=0.0)
        self.assertIsNotNone(first.notification)
        engine.update(FakePrediction("Silence", 0.0), now=2.0)
        second = engine.update(FakePrediction("Siren", 0.95), now=3.0)
        self.assertIsNone(second.notification)
        self.assertTrue(any("suppressed by cooldown" in r for r in second.reasons))

    def test_cooldown_expiry_allows_new_alert(self):
        engine = make_supervisor(
            siren_policy(confirm_windows=1, release_seconds=1.0, cooldown_seconds=10.0)
        )
        engine.update(FakePrediction("Siren", 0.95), now=0.0)
        engine.update(FakePrediction("Silence", 0.0), now=2.0)
        later = engine.update(FakePrediction("Siren", 0.95), now=11.0)
        self.assertIsNotNone(later.notification)


if __name__ == "__main__":
    unittest.main()
