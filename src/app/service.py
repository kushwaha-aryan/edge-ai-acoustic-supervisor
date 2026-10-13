"""Application service tying the UI to the real inference and decision layers.

Responsibilities:
  - `SampleCatalog` resolves a small, deterministic, allow-listed set of
    existing VALIDATION clips from the tracked inputs
    (data/splits/clip_splits.csv filtered to split == val, joined on clip_id
    with data/provenance/phase2a_clip_ledger.csv filtered to status == kept).
    Only clip_ids the catalog knows are ever classified, so no client-supplied
    filesystem path reaches the predictor.
  - `build_preferences` turns a JSON-shaped policy payload into a validated
    `UserPreferences`, reusing `src.routing.preferences` (no rule duplication).
  - `SupervisorSession` owns one `Supervisor` plus an explicit virtual clock so
    the release timer and cooldown can be advanced deterministically from the
    UI (the engine consumes every timestamp through `now`).
  - `AppService` is the single entry point the HTTP layer calls.

The model is loaded lazily and can be injected, so tests never import
TensorFlow or touch audio. The decision rules live only in
`src/routing/supervisor.py`; this module never re-implements them.
"""

from __future__ import annotations

import csv
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src import config
from src.routing.preferences import (
    PreferenceError,
    TriggerPolicy,
    UserPreferences,
    disabled_preferences,
)
from src.routing.supervisor import (
    NoPendingQuickActionError,
    Supervisor,
    SupervisorDecision,
)

__all__ = [
    "AppError",
    "SampleClip",
    "SampleCatalog",
    "SupervisorSession",
    "AppService",
    "build_preferences",
    "decision_to_dict",
]

SPLIT_CSV = config.PROJECT_ROOT / "data" / "splits" / "clip_splits.csv"
LEDGER_CSV = config.PROJECT_ROOT / "data" / "provenance" / "phase2a_clip_ledger.csv"
VALIDATION_SPLIT = "val"
KEPT_STATUS = "kept"
DEFAULT_CLIPS_PER_CLASS = 2
DEFAULT_PROFILE_NAME = "Baseline (all disabled)"
_UNSET: Any = object()


class AppError(ValueError):
    """A user-facing input or configuration problem (mapped to HTTP 400)."""


class PredictorLike(Protocol):
    """The only predictor behavior this layer needs."""

    def predict_file(self, path: str | Path) -> Any:
        """Classify one audio file and return a Prediction-like object."""


@dataclass(frozen=True)
class SampleClip:
    """One allow-listed validation clip offered to the interface."""

    clip_id: str
    label: str
    source_dataset: str
    file_name: str
    local_path: str
    duration_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        """Client-facing view; the filesystem path is intentionally omitted."""
        return {
            "clip_id": self.clip_id,
            "label": self.label,
            "source_dataset": self.source_dataset,
            "file_name": self.file_name,
            "duration_seconds": self.duration_seconds,
        }


def _resolve_path(local_path: str) -> Path:
    """Resolve a ledger local_path (either slash style) against the repo root."""
    path = Path(local_path.replace("\\", "/"))
    if path.is_absolute():
        return path
    return config.PROJECT_ROOT / path


def _duration_of(row: Mapping[str, str]) -> float | None:
    for key in ("processed_duration_s", "duration_s"):
        raw = (row.get(key) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return None


class SampleCatalog:
    """A deterministic allow-list of existing validation clips.

    Builds from the same one-to-one val<->kept join the demo scripts use, keeps
    up to `per_class` clips per supported class in taxonomy order, and exposes
    lookups by clip_id only.
    """

    def __init__(
        self,
        split_csv: str | Path | None = None,
        ledger_csv: str | Path | None = None,
        per_class: int = DEFAULT_CLIPS_PER_CLASS,
    ) -> None:
        self._split_csv = Path(split_csv) if split_csv else SPLIT_CSV
        self._ledger_csv = Path(ledger_csv) if ledger_csv else LEDGER_CSV
        self._per_class = int(per_class)
        clips = self._build()
        self._clips: tuple[SampleClip, ...] = tuple(clips)
        self._by_id: dict[str, SampleClip] = {c.clip_id: c for c in self._clips}

    @property
    def clips(self) -> tuple[SampleClip, ...]:
        return self._clips

    def get(self, clip_id: str) -> SampleClip:
        """Return the allowed clip with this id, or raise AppError."""
        clip = self._by_id.get(clip_id)
        if clip is None:
            raise AppError(f"unknown clip_id {clip_id!r} (not in the bundled set)")
        return clip

    def _build(self) -> list[SampleClip]:
        kept = self._kept_rows()
        val_ids = self._validation_ids()
        buckets: dict[str, list[SampleClip]] = {}
        for clip_id in val_ids:
            row = kept.get(clip_id)
            if row is None:
                continue
            label = row.get("project_class", "-")
            bucket = buckets.setdefault(label, [])
            if len(bucket) < self._per_class:
                bucket.append(
                    SampleClip(
                        clip_id=clip_id,
                        label=label,
                        source_dataset=row.get("source_dataset", "-"),
                        file_name=row.get("source_file", Path(clip_id).name),
                        local_path=str(
                            _resolve_path(row.get("local_path") or clip_id)
                        ),
                        duration_seconds=_duration_of(row),
                    )
                )
        ordered: list[SampleClip] = []
        for name in config.TRIGGER_CLASS_NAMES:
            ordered.extend(buckets.get(name, []))
        return ordered

    def _kept_rows(self) -> dict[str, dict[str, str]]:
        if not self._ledger_csv.is_file():
            raise AppError(f"clip ledger not found: {self._ledger_csv}")
        kept: dict[str, dict[str, str]] = {}
        with self._ledger_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == KEPT_STATUS:
                    clip_id = row["clip_id"]
                    if clip_id in kept:
                        raise AppError(f"ledger has duplicate kept clip_id: {clip_id}")
                    kept[clip_id] = row
        return kept

    def _validation_ids(self) -> list[str]:
        if not self._split_csv.is_file():
            raise AppError(f"split file not found: {self._split_csv}")
        ids: list[str] = []
        seen: set[str] = set()
        with self._split_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != VALIDATION_SPLIT:
                    continue
                clip_id = row["clip_id"]
                if clip_id in seen:
                    raise AppError(f"split file has duplicate val clip_id: {clip_id}")
                seen.add(clip_id)
                ids.append(clip_id)
        return ids


def _as_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise AppError(f"{field} must be a boolean, got {value!r}")


def _as_float(value: object, field: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise AppError(f"{field} must be a number, got {value!r}")


def _as_int(value: object, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    raise AppError(f"{field} must be an integer, got {value!r}")


def _as_str_or_none(value: object, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    raise AppError(f"{field} must be a string or null, got {value!r}")


_POLICY_FIELDS = {
    "enabled",
    "target_mode",
    "confidence_threshold",
    "confirm_windows",
    "release_seconds",
    "cooldown_seconds",
    "high_priority",
    "critical",
    "quick_action_label",
    "notify",
}


def _policy_from_mapping(name: str, entry: Mapping[str, object]) -> TriggerPolicy:
    unknown = sorted(set(entry) - _POLICY_FIELDS)
    if unknown:
        raise AppError(f"policy '{name}': unknown field(s): {', '.join(unknown)}")
    kwargs: dict[str, object] = {}
    if "enabled" in entry:
        kwargs["enabled"] = _as_bool(entry["enabled"], f"{name}.enabled")
    if "target_mode" in entry:
        kwargs["target_mode"] = _as_str_or_none(
            entry["target_mode"], f"{name}.target_mode"
        )
    if "confidence_threshold" in entry:
        kwargs["confidence_threshold"] = _as_float(
            entry["confidence_threshold"], f"{name}.confidence_threshold"
        )
    if "confirm_windows" in entry:
        kwargs["confirm_windows"] = _as_int(
            entry["confirm_windows"], f"{name}.confirm_windows"
        )
    if "release_seconds" in entry:
        kwargs["release_seconds"] = _as_float(
            entry["release_seconds"], f"{name}.release_seconds"
        )
    if "cooldown_seconds" in entry:
        kwargs["cooldown_seconds"] = _as_float(
            entry["cooldown_seconds"], f"{name}.cooldown_seconds"
        )
    if "high_priority" in entry:
        kwargs["high_priority"] = _as_bool(
            entry["high_priority"], f"{name}.high_priority"
        )
    if "critical" in entry:
        kwargs["critical"] = _as_bool(entry["critical"], f"{name}.critical")
    if "quick_action_label" in entry:
        kwargs["quick_action_label"] = _as_str_or_none(
            entry["quick_action_label"], f"{name}.quick_action_label"
        )
    if "notify" in entry:
        kwargs["notify"] = _as_bool(entry["notify"], f"{name}.notify")
    return TriggerPolicy(class_name=name, **kwargs)


def build_preferences(payload: Mapping[str, object]) -> UserPreferences:
    """Validate a JSON policy payload into a UserPreferences (or raise AppError).

    Supported keys: `profile_name` (str), `preferred_mode` (str or null), and
    `policies` (class name -> policy object). A class absent from `policies` is
    disabled. Structural/blueprint rule violations become a single AppError
    carrying the explicit problems from `src.routing.preferences`.
    """
    if not isinstance(payload, Mapping):
        raise AppError("preferences payload must be an object")
    profile_name = payload.get("profile_name") or "Custom"
    if not isinstance(profile_name, str):
        raise AppError("profile_name must be a string")
    preferred_mode = _as_str_or_none(payload.get("preferred_mode"), "preferred_mode")

    policies_raw = payload.get("policies", {})
    if not isinstance(policies_raw, Mapping):
        raise AppError("'policies' must be an object mapping class names to policies")
    unknown = sorted(set(policies_raw) - set(config.TRIGGER_CLASS_NAMES))
    if unknown:
        raise AppError(f"unknown trigger class(es): {', '.join(unknown)}")

    policies: list[TriggerPolicy] = []
    for name in config.TRIGGER_CLASS_NAMES:
        entry = policies_raw.get(name)
        if entry is None:
            policies.append(TriggerPolicy(class_name=name))
            continue
        if not isinstance(entry, Mapping):
            raise AppError(f"policy for '{name}' must be an object")
        policies.append(_policy_from_mapping(name, entry))

    try:
        return UserPreferences.build(
            policies, profile_name=profile_name, preferred_mode=preferred_mode
        )
    except PreferenceError as exc:
        raise AppError(str(exc)) from exc


@dataclass(frozen=True)
class _Detection:
    label: str | None
    confidence: float


class SupervisorSession:
    """One supervisor plus an explicit virtual clock advanced by the UI."""

    def __init__(self, preferences: UserPreferences) -> None:
        self._preferences = preferences
        self._supervisor = Supervisor(preferences)
        self._now = 0.0
        self._last_label: str | None = None
        self._last_confidence: float | None = None
        self._last_decision: SupervisorDecision | None = None

    @property
    def preferences(self) -> UserPreferences:
        return self._preferences

    def set_prediction(self, label: str, confidence: float) -> None:
        self._last_label = label
        self._last_confidence = confidence

    def step(
        self,
        *,
        label: object = _UNSET,
        confidence: float | None = None,
        advance_seconds: float = 0.0,
    ) -> SupervisorDecision:
        """Feed one detection (or none) and return the resulting decision."""
        if advance_seconds < 0:
            raise AppError("advance_seconds must be >= 0")
        if label is _UNSET:
            if self._last_label is None:
                raise AppError("no detection yet - analyze a clip first")
            detected: str | None = self._last_label
            conf = float(self._last_confidence or 0.0)
        elif label is None:
            detected = None
            conf = 0.0
        else:
            detected = str(label)
            conf = (
                float(confidence)
                if confidence is not None
                else float(self._last_confidence or 1.0)
            )
        self._now += float(advance_seconds)
        decision = self._supervisor.update(
            _Detection(detected, conf), now=self._now
        )
        self._last_decision = decision
        return decision

    def quick_action(self, action: str) -> object:
        """Confirm or reject the pending quick action (presentation-only)."""
        try:
            if action == "confirm":
                return self._supervisor.confirm_quick_action(now=self._now)
            if action == "reject":
                return self._supervisor.reject_quick_action(now=self._now)
        except NoPendingQuickActionError as exc:
            raise AppError(str(exc)) from exc
        raise AppError("action must be 'confirm' or 'reject'")

    def snapshot(self) -> dict[str, object]:
        return {
            "now": self._now,
            "profile_name": self._preferences.profile_name,
            "preferred_mode": self._preferences.preferred_mode,
            "current_mode": self._supervisor.current_mode,
            "pending_quick_action": self._supervisor.pending_quick_action,
            "enabled_classes": list(self._preferences.enabled_classes()),
            "decision": (
                decision_to_dict(self._last_decision)
                if self._last_decision is not None
                else None
            ),
        }


def decision_to_dict(decision: SupervisorDecision) -> dict[str, object]:
    """Serialize a SupervisorDecision for the JSON API."""
    return {
        "detected_class": decision.detected_class,
        "detected_confidence": decision.detected_confidence,
        "trigger_enabled": decision.trigger_enabled,
        "verification_state": decision.verification_state.value,
        "verified": decision.verified,
        "policy_target_mode": decision.policy_target_mode,
        "audio_mode": decision.audio_mode,
        "preferred_mode": decision.preferred_mode,
        "notification": decision.notification,
        "warning": decision.warning,
        "quick_action": decision.quick_action,
        "quick_action_pending": decision.quick_action_pending,
        "release_timer_active": decision.release_timer_active,
        "release_remaining_seconds": decision.release_remaining_seconds,
        "cooldown_active": decision.cooldown_active,
        "restored": decision.restored,
        "reasons": list(decision.reasons),
        "log": list(decision.log),
    }


def _load_default_predictor() -> PredictorLike:
    from src.inference import yamnet_mlp

    return yamnet_mlp.Predictor()


class AppService:
    """The application's single entry point for the HTTP layer."""

    def __init__(
        self,
        catalog: SampleCatalog,
        predictor_factory: Callable[[], PredictorLike] | None = None,
    ) -> None:
        self._catalog = catalog
        self._predictor_factory = predictor_factory or _load_default_predictor
        self._predictor: PredictorLike | None = None
        self._lock = threading.Lock()
        self._session = SupervisorSession(
            disabled_preferences(
                profile_name=DEFAULT_PROFILE_NAME,
                preferred_mode=config.MODE_MAX_ANC,
            )
        )

    @property
    def catalog(self) -> SampleCatalog:
        return self._catalog

    def _get_predictor(self) -> PredictorLike:
        if self._predictor is None:
            self._predictor = self._predictor_factory()
        return self._predictor

    def config(self) -> dict[str, object]:
        """Everything the interface needs to render its controls."""
        clips = [clip.to_dict() for clip in self._catalog.clips]
        by_class: dict[str, int] = {}
        for clip in self._catalog.clips:
            by_class[clip.label] = by_class.get(clip.label, 0) + 1
        return {
            "classes": list(config.TRIGGER_CLASS_NAMES),
            "modes": list(config.AUDIO_MODES),
            "no_action_mode_label": "No change (leave mode as is)",
            "clips": clips,
            "clip_counts": by_class,
            "preferred_mode_default": config.MODE_MAX_ANC,
            "policy_defaults": {
                "confidence_threshold": config.TRIGGER_MIN_CONFIDENCE,
                "confirm_windows": config.TRIGGER_CONFIRM_WINDOWS,
                "release_seconds": config.TRIGGER_RELEASE_SECONDS,
                "cooldown_seconds": config.RESPONSE_COOLDOWN_SECONDS,
            },
            "notes": [
                "SIMULATED interface: audio-mode changes are software state only "
                "and no earbud hardware is controlled.",
                "No live-microphone capture: clips come from the bundled "
                "validation sample.",
                "Confirming a quick action is presentation-only; nothing is "
                "executed or sent.",
            ],
        }

    def apply_preferences(self, payload: Mapping[str, object]) -> dict[str, object]:
        preferences = build_preferences(payload)
        with self._lock:
            self._session = SupervisorSession(preferences)
            return {"state": self._session.snapshot()}

    def classify(self, payload: Mapping[str, object]) -> dict[str, object]:
        clip_id = payload.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise AppError("clip_id is required")
        clip = self._catalog.get(clip_id)
        with self._lock:
            predictor = self._get_predictor()
            try:
                prediction = predictor.predict_file(clip.local_path)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                raise AppError(
                    f"could not classify clip {clip_id!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            label = str(getattr(prediction, "label", ""))
            confidence = float(getattr(prediction, "confidence", 0.0))
            top_k = [
                [str(name), float(prob)] for name, prob in prediction.top_k()
            ]
            self._session.set_prediction(label, confidence)
            state = self._session.snapshot()
        return {
            "state": state,
            "prediction": {
                "clip": clip.to_dict(),
                "label": label,
                "confidence": confidence,
                "top_k": top_k,
            },
        }

    def step(self, payload: Mapping[str, object]) -> dict[str, object]:
        advance = _as_float(payload.get("advance_seconds", 0.0), "advance_seconds")
        with self._lock:
            if "label" in payload:
                label = payload["label"]
                if label is not None and not isinstance(label, str):
                    raise AppError("label must be a string or null")
                confidence = (
                    _as_float(payload.get("confidence", 1.0), "confidence")
                    if label is not None
                    else None
                )
                decision = self._session.step(
                    label=label, confidence=confidence, advance_seconds=advance
                )
            else:
                decision = self._session.step(advance_seconds=advance)
            return {"state": self._session.snapshot(), "decision": decision_to_dict(decision)}

    def quick_action(self, payload: Mapping[str, object]) -> dict[str, object]:
        action = payload.get("action")
        if not isinstance(action, str):
            raise AppError("action is required and must be 'confirm' or 'reject'")
        with self._lock:
            result = self._session.quick_action(action)
            return {
                "state": self._session.snapshot(),
                "result": {
                    "class_name": result.class_name,
                    "label": result.label,
                    "confirmed": result.confirmed,
                    "executed": result.executed,
                    "note": result.note,
                },
            }
