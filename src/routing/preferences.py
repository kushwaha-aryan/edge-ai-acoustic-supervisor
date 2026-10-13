"""User trigger-policy preferences for the supervisor decision layer.

Stage 2 (Context / policy) of the four-stage architecture (blueprint section
2.1): a validated, in-memory representation of the user's per-class trigger
policy. This module holds no model state and performs no inference - it only
describes which detected classes should cause a response and how that response
is shaped (target audio mode, confidence threshold, persistence, notification,
warning, and the optional user-confirmed quick action).

Everything here is configuration consumed by `src/routing/supervisor.py`; the
classifier in `src/inference/yamnet_mlp.py` is never imported, loaded, or
retrained by this layer (blueprint sections 2.1-2.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from src import config

__all__ = [
    "PreferenceError",
    "TriggerPolicy",
    "UserPreferences",
    "disabled_preferences",
]

NO_TARGET_MODE: None = None


class PreferenceError(ValueError):
    """Raised when a preference set violates a structural/blueprint rule."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class TriggerPolicy:
    """One per-class policy entry (blueprint section 2.1, stage 2 + per-entry rules).

    `enabled` is the context/policy gate. When an entry is disabled a correct
    detection must produce no response at all (blueprint section 1.4).
    `target_mode` is `None` for the explicit "no audio-mode change" case
    (blueprint section 3); otherwise it must be one of `config.AUDIO_MODES`.
    `confirm_windows` is the per-entry persistence rule, so a safety-relevant
    trigger can require a single high-confidence window (blueprint section 7
    Phase 5). `high_priority` requests a warning (response type C) and
    `critical` requests a user-confirmed quick action (response type D), which
    is presentation-only and disabled by default (blueprint section 2A.3).
    """

    class_name: str
    enabled: bool = False
    target_mode: str | None = NO_TARGET_MODE
    confidence_threshold: float = config.TRIGGER_MIN_CONFIDENCE
    confirm_windows: int = config.TRIGGER_CONFIRM_WINDOWS
    release_seconds: float = config.TRIGGER_RELEASE_SECONDS
    cooldown_seconds: float = config.RESPONSE_COOLDOWN_SECONDS
    high_priority: bool = False
    critical: bool = False
    quick_action_label: str | None = None
    notify: bool = True

    def validate(self) -> list[str]:
        """Return every problem with this entry; an empty list means valid."""
        problems: list[str] = []
        if self.class_name not in config.TRIGGER_CLASS_NAMES:
            problems.append(
                f"unknown trigger class '{self.class_name}' (not in the ratified "
                f"{config.NUM_TRIGGER_CLASSES}-class taxonomy)"
            )
        if self.target_mode is not None and self.target_mode not in config.AUDIO_MODES:
            problems.append(
                f"{self.class_name}: target_mode '{self.target_mode}' is not one of "
                f"{config.AUDIO_MODES} (use None for the explicit no-action case)"
            )
        if not _is_number(self.confidence_threshold) or not (
            0.0 < float(self.confidence_threshold) <= 1.0
        ):
            problems.append(
                f"{self.class_name}: confidence_threshold must be in (0.0, 1.0], "
                f"got {self.confidence_threshold!r}"
            )
        if not isinstance(self.confirm_windows, int) or isinstance(
            self.confirm_windows, bool
        ) or self.confirm_windows < 1:
            problems.append(
                f"{self.class_name}: confirm_windows must be an integer >= 1, "
                f"got {self.confirm_windows!r}"
            )
        if not _is_number(self.release_seconds) or float(self.release_seconds) < 0.0:
            problems.append(
                f"{self.class_name}: release_seconds must be >= 0, "
                f"got {self.release_seconds!r}"
            )
        if not _is_number(self.cooldown_seconds) or float(self.cooldown_seconds) < 0.0:
            problems.append(
                f"{self.class_name}: cooldown_seconds must be >= 0, "
                f"got {self.cooldown_seconds!r}"
            )
        if (self.high_priority or self.critical) and not self.enabled:
            problems.append(
                f"{self.class_name}: high_priority/critical apply only to an "
                f"enabled entry; this entry is disabled"
            )
        if self.critical and not (self.quick_action_label or "").strip():
            problems.append(
                f"{self.class_name}: critical entry needs a non-empty "
                f"quick_action_label (the action surfaced for user confirmation)"
            )
        if self.quick_action_label and not self.critical:
            problems.append(
                f"{self.class_name}: quick_action_label is only valid when "
                f"critical is True (quick action is disabled by default)"
            )
        return problems


@dataclass(frozen=True)
class UserPreferences:
    """A validated set of per-class policies plus the preferred (default) mode.

    `preferred_mode` is the user-default audio mode restored after a trigger is
    absent for its release period (blueprint section 2A.2). A class with no
    entry is treated as disabled - a missing preference is never treated as an
    enabled trigger.
    """

    policies: Mapping[str, TriggerPolicy]
    profile_name: str = "Custom"
    preferred_mode: str | None = NO_TARGET_MODE

    def __post_init__(self) -> None:
        object.__setattr__(self, "policies", MappingProxyType(dict(self.policies)))

    def policy_for(self, class_name: str) -> TriggerPolicy | None:
        """Return the policy entry for a class, or None when it has none."""
        return self.policies.get(class_name)

    def enabled_classes(self) -> tuple[str, ...]:
        """Return the enabled trigger classes, in taxonomy order."""
        return tuple(
            name for name in config.TRIGGER_CLASS_NAMES
            if (entry := self.policies.get(name)) is not None and entry.enabled
        )

    def validate(self) -> list[str]:
        """Return every problem with this preference set; empty means valid."""
        problems: list[str] = []
        if self.preferred_mode is not None and self.preferred_mode not in config.AUDIO_MODES:
            problems.append(
                f"preferred_mode '{self.preferred_mode}' is not one of "
                f"{config.AUDIO_MODES}"
            )
        for key, entry in self.policies.items():
            if key != entry.class_name:
                problems.append(
                    f"policy key '{key}' does not match its class_name "
                    f"'{entry.class_name}'"
                )
            problems.extend(entry.validate())
        return problems

    @classmethod
    def build(
        cls,
        policies: Iterable[TriggerPolicy] | Mapping[str, TriggerPolicy],
        *,
        profile_name: str = "Custom",
        preferred_mode: str | None = NO_TARGET_MODE,
    ) -> "UserPreferences":
        """Construct and validate a preference set, raising on any problem."""
        if isinstance(policies, Mapping):
            mapping = dict(policies)
        else:
            mapping = {entry.class_name: entry for entry in policies}
        prefs = cls(
            policies=mapping,
            profile_name=profile_name,
            preferred_mode=preferred_mode,
        )
        problems = prefs.validate()
        if problems:
            raise PreferenceError(
                f"invalid preferences ({len(problems)} problem(s)): "
                + "; ".join(problems)
            )
        return prefs

    @classmethod
    def from_class_modes(
        cls,
        class_modes: Mapping[str, str | None],
        *,
        profile_name: str = "Custom",
        preferred_mode: str | None = NO_TARGET_MODE,
        defaults: TriggerPolicy | None = None,
    ) -> "UserPreferences":
        """Build an all-disabled table, enabling exactly the given class -> mode map.

        A convenience for expressing "these classes are enabled, each mapped to
        this audio mode"; every other supported class stays disabled. Per-class
        extras (threshold, persistence, warning, quick action) come from
        `defaults` when supplied.
        """
        unknown = sorted(set(class_modes) - set(config.TRIGGER_CLASS_NAMES))
        if unknown:
            raise PreferenceError(
                f"unknown trigger class(es) in class_modes: {unknown}"
            )
        base = defaults or TriggerPolicy(class_name="")
        policies: dict[str, TriggerPolicy] = {}
        for name in config.TRIGGER_CLASS_NAMES:
            if name in class_modes:
                policies[name] = TriggerPolicy(
                    class_name=name,
                    enabled=True,
                    target_mode=class_modes[name],
                    confidence_threshold=base.confidence_threshold,
                    confirm_windows=base.confirm_windows,
                    release_seconds=base.release_seconds,
                    cooldown_seconds=base.cooldown_seconds,
                    high_priority=base.high_priority,
                    critical=base.critical,
                    quick_action_label=base.quick_action_label,
                    notify=base.notify,
                )
            else:
                policies[name] = TriggerPolicy(class_name=name)
        return cls.build(
            policies,
            profile_name=profile_name,
            preferred_mode=preferred_mode,
        )


def disabled_preferences(
    profile_name: str = "Custom",
    preferred_mode: str | None = NO_TARGET_MODE,
) -> UserPreferences:
    """Return a safe base preference set: every supported class disabled.

    Nothing is enabled and no policy is invented; the user (or a later phase)
    turns specific classes on. A disabled set makes every detection a no-action
    case (blueprint section 1.4).
    """
    policies: Sequence[TriggerPolicy] = [
        TriggerPolicy(class_name=name) for name in config.TRIGGER_CLASS_NAMES
    ]
    return UserPreferences.build(
        policies,
        profile_name=profile_name,
        preferred_mode=preferred_mode,
    )
