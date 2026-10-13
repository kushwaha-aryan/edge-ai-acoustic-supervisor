"""Supervisor decision engine: stages 3 (verification) and 4 (Context Response).

The decision layer of the Edge AI Acoustic Supervisor. It consumes a
prediction result plus validated `UserPreferences` and returns an explicit
`SupervisorDecision`. It is deliberately separated from perception: it never
classifies audio, never loads or retrains a model, never opens the test split,
and never duplicates the inference pipeline (blueprint sections 2.1-2.2).

The prediction input is duck-typed - any object exposing `.label` and
`.confidence` (the contract of `src.inference.yamnet_mlp.Prediction`) is
accepted, so tests can feed synthetic predictions without importing TensorFlow
or the inference module.

Implemented response types (blueprint section 2A):
  A  audio-mode response   - the policy-mapped mode, or no change for "none"
  B  notification          - wording per section 2A.4
  C  warning / priority    - for entries flagged high_priority
  D  user-confirmed quick action - PRESENTED only; never executed
  E  state management      - release timer + preferred-mode restoration,
                             cooldown / duplicate-alert suppression, log

Hard boundary (blueprint section 2A.3): the quick action is presentation-only.
`confirm_quick_action` returns a result whose `executed` field is always False;
nothing is called, sent, or dispatched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

from src import config
from src.routing.preferences import UserPreferences

__all__ = [
    "PredictionLike",
    "VerificationState",
    "SupervisorDecision",
    "QuickActionResult",
    "NoPendingQuickActionError",
    "Supervisor",
]


@runtime_checkable
class PredictionLike(Protocol):
    """Minimal prediction contract consumed by the decision layer."""

    label: str
    confidence: float


class NoPendingQuickActionError(RuntimeError):
    """Raised when a quick action is confirmed/rejected but none is pending."""


class VerificationState(str, Enum):
    """Decision-stage state (blueprint section 2A.5).

    IDLE is the refinement of the blueprint's three states for the case where
    no candidate trigger is being tracked (nothing detected, gate failed, or
    below threshold); PENDING / VERIFIED / RELEASED match the blueprint.
    """

    IDLE = "idle"
    PENDING = "pending"
    VERIFIED = "verified"
    RELEASED = "released"


@dataclass(frozen=True)
class QuickActionResult:
    """Outcome of a user confirmation or rejection of the presented action.

    `executed` is always False: the supervisor surfaces an action, it never
    runs one (blueprint section 2A.3). A host UI may act only by its own,
    separate, explicit means.
    """

    class_name: str
    label: str
    confirmed: bool
    executed: bool
    note: str


@dataclass(frozen=True)
class SupervisorDecision:
    """One explicit decision for a single prediction, ready for a frontend."""

    detected_class: str | None
    detected_confidence: float
    trigger_enabled: bool
    verification_state: VerificationState
    verified: bool
    policy_target_mode: str | None
    audio_mode: str | None
    preferred_mode: str | None
    notification: str | None
    warning: str | None
    quick_action: str | None
    quick_action_pending: bool
    release_timer_active: bool
    release_remaining_seconds: float | None
    cooldown_active: bool
    restored: bool
    reasons: tuple[str, ...]
    log: tuple[str, ...]


def _humanize(class_name: str) -> str:
    return class_name.replace("_", " ")


class Supervisor:
    """Stateful decision engine over a stream of predictions.

    Time is supplied explicitly through `now` (seconds, monotonic) so behavior
    is deterministic and testable; when omitted a monotonic clock is used. Each
    call to `update` consumes exactly one prediction and returns one decision.
    """

    def __init__(
        self,
        preferences: UserPreferences,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._prefs = preferences
        self._clock = clock
        self._pending_class: str | None = None
        self._pending_count: int = 0
        self._active_class: str | None = None
        self._active_release_seconds: float = config.TRIGGER_RELEASE_SECONDS
        self._current_mode: str | None = preferences.preferred_mode
        self._last_seen: float | None = None
        self._cooldown_until: dict[str, float] = {}
        self._qa_label: str | None = None
        self._qa_class: str | None = None
        self._log: list[tuple[float, str]] = []

    @property
    def preferences(self) -> UserPreferences:
        return self._prefs

    @property
    def current_mode(self) -> str | None:
        return self._current_mode

    @property
    def pending_quick_action(self) -> str | None:
        """The quick-action label currently surfaced, or None."""
        return self._qa_label

    @property
    def log(self) -> tuple[tuple[float, str], ...]:
        return tuple(self._log)

    def update(
        self,
        prediction: PredictionLike,
        now: float | None = None,
    ) -> SupervisorDecision:
        """Consume one prediction and return the resulting decision."""
        moment = float(now) if now is not None else float(self._clock())
        detected = getattr(prediction, "label", None)
        confidence = float(getattr(prediction, "confidence", 0.0))

        reasons: list[str] = []
        added: list[str] = []
        restored = False

        policy = (
            self._prefs.policy_for(detected) if isinstance(detected, str) else None
        )
        known = isinstance(detected, str) and detected in config.TRIGGER_CLASS_NAMES
        trigger_enabled = bool(policy is not None and policy.enabled)

        if not known:
            reasons.append(
                f"detected class {detected!r} is not a supported trigger class - "
                f"no action"
            )
            candidate = False
        elif policy is None or not policy.enabled:
            reasons.append(
                f"detected class '{detected}' is not enabled in profile "
                f"'{self._prefs.profile_name}' - no action"
            )
            candidate = False
        elif confidence < float(policy.confidence_threshold):
            reasons.append(
                f"confidence {confidence:.2f} below threshold "
                f"{float(policy.confidence_threshold):.2f} for '{detected}' - "
                f"no action"
            )
            candidate = False
        else:
            candidate = True

        if candidate:
            if self._pending_class == detected:
                self._pending_count += 1
            else:
                self._pending_class = detected
                self._pending_count = 1
        else:
            self._pending_class = None
            self._pending_count = 0

        verified_now = bool(candidate) and self._pending_count >= int(
            policy.confirm_windows  # type: ignore[union-attr]
        )

        fire_edge = False
        notification: str | None = None
        warning: str | None = None
        quick_action: str | None = None

        if verified_now:
            fire_edge = self._active_class != detected
            self._active_class = detected  # type: ignore[assignment]
            self._active_release_seconds = float(policy.release_seconds)  # type: ignore[union-attr]
            self._last_seen = moment
            if policy.target_mode is not None:  # type: ignore[union-attr]
                self._current_mode = policy.target_mode  # type: ignore[union-attr]

        if fire_edge:
            in_cooldown = moment < self._cooldown_until.get(detected, float("-inf"))
            if in_cooldown:
                reasons.append(
                    f"response for '{detected}' suppressed by cooldown "
                    f"(duplicate alert)"
                )
                added.append(f"response suppressed by cooldown for '{detected}'")
            else:
                if policy.notify:  # type: ignore[union-attr]
                    if policy.target_mode is not None:  # type: ignore[union-attr]
                        notification = (
                            f"{_humanize(detected)} detected - "  # type: ignore[arg-type]
                            f"{policy.target_mode} enabled"  # type: ignore[union-attr]
                        )
                    else:
                        notification = f"{_humanize(detected)} detected"  # type: ignore[arg-type]
                    added.append(f"notification: {notification}")
                if policy.high_priority:  # type: ignore[union-attr]
                    warning = f"[WARNING] {_humanize(detected)} detected"  # type: ignore[arg-type]
                    added.append(f"warning: {warning}")
                if policy.critical:  # type: ignore[union-attr]
                    self._qa_label = policy.quick_action_label  # type: ignore[union-attr]
                    self._qa_class = detected  # type: ignore[assignment]
                    quick_action = self._qa_label
                    added.append(
                        f"quick action presented (awaiting user confirmation): "
                        f"{quick_action}"
                    )
                self._cooldown_until[detected] = (  # type: ignore[index]
                    moment + float(policy.cooldown_seconds)  # type: ignore[union-attr]
                )

        if self._active_class is not None and self._pending_class != self._active_class:
            elapsed = moment - self._last_seen  # type: ignore[operator]
            if elapsed >= self._active_release_seconds:
                self._current_mode = self._prefs.preferred_mode
                added.append(
                    f"trigger '{self._active_class}' absent for "
                    f"{self._active_release_seconds:.1f}s - preferred mode restored"
                )
                self._active_class = None
                self._qa_label = None
                self._qa_class = None
                restored = True

        if verified_now:
            state = VerificationState.VERIFIED
        elif restored:
            state = VerificationState.RELEASED
        elif candidate:
            state = VerificationState.PENDING
            added.append(
                f"awaiting persistence {self._pending_count}/"
                f"{int(policy.confirm_windows)} for '{detected}'"  # type: ignore[union-attr]
            )
        else:
            state = VerificationState.IDLE

        if self._active_class is not None:
            release_active = not verified_now
            elapsed_since_seen = moment - self._last_seen  # type: ignore[operator]
            release_remaining = max(
                0.0, self._active_release_seconds - elapsed_since_seen
            )
        else:
            release_active = False
            release_remaining = None

        cooldown_active = any(
            until > moment for until in self._cooldown_until.values()
        )

        self._log.extend((moment, message) for message in added)

        return SupervisorDecision(
            detected_class=detected if isinstance(detected, str) else None,
            detected_confidence=confidence,
            trigger_enabled=trigger_enabled,
            verification_state=state,
            verified=verified_now,
            policy_target_mode=(
                policy.target_mode if policy is not None else None  # type: ignore[union-attr]
            ),
            audio_mode=self._current_mode,
            preferred_mode=self._prefs.preferred_mode,
            notification=notification,
            warning=warning,
            quick_action=quick_action,
            quick_action_pending=self._qa_label is not None,
            release_timer_active=release_active,
            release_remaining_seconds=release_remaining,
            cooldown_active=cooldown_active,
            restored=restored,
            reasons=tuple(reasons),
            log=tuple(f"t={moment:.2f} {m}" for m in added),
        )

    def confirm_quick_action(self, now: float | None = None) -> QuickActionResult:
        """Confirm the surfaced quick action: presentation-only, nothing runs."""
        if self._qa_label is None or self._qa_class is None:
            raise NoPendingQuickActionError(
                "no user-confirmed quick action is currently pending"
            )
        moment = float(now) if now is not None else float(self._clock())
        label = self._qa_label
        class_name = self._qa_class
        self._qa_label = None
        self._qa_class = None
        self._log.append(
            (
                moment,
                f"quick action confirmed by user for '{class_name}': {label} "
                f"(presentation-only; the supervisor executed nothing)",
            )
        )
        return QuickActionResult(
            class_name=class_name,
            label=label,
            confirmed=True,
            executed=False,
            note=(
                "presentation-only UI shortcut prototype; no action was executed "
                "and no emergency capability is claimed"
            ),
        )

    def reject_quick_action(self, now: float | None = None) -> QuickActionResult:
        """Dismiss the surfaced quick action without any effect."""
        if self._qa_label is None or self._qa_class is None:
            raise NoPendingQuickActionError(
                "no user-confirmed quick action is currently pending"
            )
        moment = float(now) if now is not None else float(self._clock())
        label = self._qa_label
        class_name = self._qa_class
        self._qa_label = None
        self._qa_class = None
        self._log.append(
            (moment, f"quick action dismissed by user for '{class_name}': {label}")
        )
        return QuickActionResult(
            class_name=class_name,
            label=label,
            confirmed=False,
            executed=False,
            note="dismissed; no action was executed",
        )
