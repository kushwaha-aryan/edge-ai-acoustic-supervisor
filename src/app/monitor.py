"""Session monitoring state for the local supervisor dashboard.

Holds only observational data: model-initialization stages, per-request
activity, recent inference history and summary statistics. Every duration is
measured with :func:`time.perf_counter` in the service/inference layers; this
module never estimates or invents a duration. All state is guarded by a lock so
concurrent HTTP requests keep separate tracks and cannot mix progress.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any

from src import config

__all__ = [
    "INIT_STAGES",
    "INFERENCE_STAGES",
    "PHASE_LABELS",
    "DashboardMonitor",
    "valid_request_id",
]

MAX_HISTORY = 25
MAX_TRACKS = 12
MAX_EVENTS = 40

# (stage key, dashboard label). ``pipeline_wiring``/``other_overhead`` are the
# explicitly reported remainder of the measured total, never invented stages.
INIT_STAGES: tuple[tuple[str, str], ...] = (
    ("environment", "Initializing inference environment"),
    ("scaler_load", "Loading feature scaler"),
    ("mlp_load", "Loading trained MLP model"),
    ("metadata_load", "Loading class metadata"),
    ("yamnet_load", "Loading pretrained YAMNet"),
    ("pipeline_wiring", "Initializing prediction pipeline"),
)
INFERENCE_STAGES: tuple[tuple[str, str], ...] = (
    ("reception", "Audio file reception"),
    ("audio_decode", "Audio decoding and preprocessing"),
    ("scaler_load", "Loading feature scaler (first use)"),
    ("mlp_load", "Loading MLP model (first use)"),
    ("metadata_load", "Loading class metadata (first use)"),
    ("yamnet_load", "Loading pretrained YAMNet (first use)"),
    ("embedding", "YAMNet embedding extraction"),
    ("scaling", "Feature scaling"),
    ("mlp_classification", "MLP classification"),
    ("aggregation", "Prediction aggregation and supervisor decision"),
    ("other_overhead", "Other request overhead"),
)
PHASE_LABELS: dict[str, str] = {
    "received": "Receiving audio file",
    "reception": "Receiving audio file",
    "audio_decode": "Decoding audio",
    "embedding": "Extracting YAMNet embeddings",
    "scaling": "Scaling extracted features",
    "mlp_classification": "Running MLP classification",
    "aggregation": "Aggregating predictions",
    "supervisor": "Evaluating supervisor policy",
    "completed": "Prediction completed",
    "failed": "Prediction failed",
}

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def valid_request_id(value: object) -> bool:
    return isinstance(value, str) and bool(_REQUEST_ID_RE.fullmatch(value))


def _ms(seconds: float) -> float:
    return round(max(0.0, float(seconds)) * 1000.0, 3)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass
class RequestTrack:
    """One inference request's live activity timeline."""

    request_id: str
    file_name: str
    source: str
    phase: str = "received"
    status: str = "running"
    error: str | None = None
    started_perf: float = field(default_factory=perf_counter)
    started_at: str = field(default_factory=_now_iso)
    events: list[dict[str, Any]] = field(default_factory=list)

    def label(self) -> str:
        return PHASE_LABELS.get(self.phase, self.phase)

    def view(self) -> dict[str, object]:
        elapsed = (
            perf_counter() - self.started_perf
            if self.status == "running"
            else 0.0
        )
        return {
            "request_id": self.request_id,
            "file_name": self.file_name,
            "source": self.source,
            "phase": self.phase,
            "phase_label": self.label(),
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "elapsed_ms": _ms(elapsed),
            "events": list(self.events),
        }


class DashboardMonitor:
    """Thread-safe, in-memory dashboard telemetry for one application session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._init_status = "not_started"
        self._init_phase: str | None = None
        self._init_error: str | None = None
        self._init_started_at: str | None = None
        self._init_completed_at: str | None = None
        self._init_total_s = 0.0
        self._init_attempts = 0
        self._init_already_loaded = False
        self._init_sink: Any = None
        self._init_stages: dict[str, float] = {}
        self._tracks: dict[str, RequestTrack] = {}
        self._track_order: deque[str] = deque()
        self._history: deque[dict[str, object]] = deque(maxlen=MAX_HISTORY)
        self._count = 0
        self._sum_ms = 0.0
        self._min_ms: float | None = None
        self._max_ms: float | None = None

    # ------------------------------------------------------------------ model

    def begin_init(self, sink: Any) -> bool:
        """Mark initialization as started; returns False if already running."""
        with self._lock:
            if self._init_status == "initializing":
                return False
            self._init_status = "initializing"
            self._init_phase = "environment"
            self._init_error = None
            self._init_started_at = _now_iso()
            self._init_completed_at = None
            self._init_total_s = 0.0
            self._init_already_loaded = False
            self._init_attempts += 1
            self._init_sink = sink
            self._init_stages = {}
            return True

    def init_phase(self, stage_key: str) -> None:
        with self._lock:
            if self._init_status == "initializing":
                self._init_phase = stage_key

    def _finish_init(self, status: str, total_s: float, error: str | None,
                     already_loaded: bool) -> None:
        with self._lock:
            sink = self._init_sink
            self._init_stages = dict(sink.snapshot()) if sink is not None else {}
            self._init_status = status
            self._init_total_s = max(0.0, float(total_s))
            self._init_error = error
            self._init_already_loaded = already_loaded
            self._init_completed_at = _now_iso()
            self._init_phase = None
            self._init_sink = None

    def finish_init(self, total_s: float, already_loaded: bool) -> None:
        self._finish_init("ready", total_s, None, already_loaded)

    def fail_init(self, total_s: float, error: str) -> None:
        self._finish_init("failed", total_s, error, False)

    def init_status(self) -> dict[str, object]:
        with self._lock:
            stages = dict(self._init_stages)
            if self._init_status == "initializing" and self._init_sink is not None:
                stages.update(self._init_sink.snapshot())
            known = {key for key, _ in INIT_STAGES}
            measured = sum(value for key, value in stages.items() if key in known)
            total_ms = _ms(self._init_total_s)
            overhead_ms = (
                round(total_ms - _ms(measured), 3)
                if self._init_status != "initializing"
                else 0.0
            )
            rows: list[dict[str, object]] = []
            for key, label in INIT_STAGES:
                if key == "pipeline_wiring":
                    duration = overhead_ms
                else:
                    duration = _ms(stages.get(key, 0.0))
                if self._init_status == "initializing":
                    state = "active" if self._init_phase == key else (
                        "done" if key in stages else "pending"
                    )
                else:
                    state = "done" if self._init_status == "ready" else (
                        "done" if duration > 0 else "pending"
                    )
                rows.append({
                    "key": key,
                    "label": label,
                    "duration_ms": duration,
                    "state": state,
                })
            return {
                "status": self._init_status,
                "phase": self._init_phase,
                "phase_label": PHASE_LABELS.get(self._init_phase or "", "Starting"),
                "error": self._init_error,
                "attempts": self._init_attempts,
                "already_loaded": self._init_already_loaded,
                "started_at": self._init_started_at,
                "completed_at": self._init_completed_at,
                "total_ms": total_ms,
                "stages": rows,
                "model": f"YAMNet + MLP ({config.YAMNET_MLP_RUN})",
                "model_dir": str(config.YAMNET_MLP_DIR),
            }

    # --------------------------------------------------------------- activity

    def begin_request(self, request_id: str, file_name: str,
                      source: str) -> None:
        with self._lock:
            track = RequestTrack(request_id, file_name, source)
            track.events.append({
                "stage": "received",
                "label": PHASE_LABELS["received"],
                "at_ms": 0.0,
            })
            self._tracks[request_id] = track
            self._track_order.append(request_id)
            while len(self._track_order) > MAX_TRACKS:
                old = self._track_order.popleft()
                self._tracks.pop(old, None)

    def phase(self, request_id: str, stage_key: str) -> None:
        with self._lock:
            track = self._tracks.get(request_id)
            if track is None or track.status != "running":
                return
            if stage_key in ("environment", "scaler_load", "mlp_load",
                             "metadata_load", "yamnet_load"):
                return  # initialization stages belong to the model panel
            elapsed = _ms(perf_counter() - track.started_perf)
            track.phase = stage_key
            if len(track.events) < MAX_EVENTS:
                track.events.append({
                    "stage": stage_key,
                    "label": PHASE_LABELS.get(stage_key, stage_key),
                    "at_ms": elapsed,
                })

    def finish_request(self, request_id: str,
                       record: dict[str, object]) -> None:
        with self._lock:
            track = self._tracks.get(request_id)
            if track is None:
                return
            track.status = "completed"
            track.phase = "completed"
            track.events.append({
                "stage": "completed",
                "label": PHASE_LABELS["completed"],
                "at_ms": _ms(perf_counter() - track.started_perf),
            })
            self._history.append(record)
            total_ms = float(record.get("total_ms", 0.0))
            self._count += 1
            self._sum_ms += total_ms
            self._min_ms = total_ms if self._min_ms is None else min(self._min_ms, total_ms)
            self._max_ms = total_ms if self._max_ms is None else max(self._max_ms, total_ms)

    def fail_request(self, request_id: str, error: str) -> None:
        with self._lock:
            track = self._tracks.get(request_id)
            if track is None:
                return
            track.status = "failed"
            track.phase = "failed"
            track.error = error
            track.events.append({
                "stage": "failed",
                "label": PHASE_LABELS["failed"],
                "at_ms": _ms(perf_counter() - track.started_perf),
            })

    def activity(self, request_id: str | None = None) -> dict[str, object]:
        with self._lock:
            running = [
                self._tracks[rid] for rid in reversed(self._track_order)
                if self._tracks.get(rid) is not None
                and self._tracks[rid].status == "running"
            ]
            current = running[0] if running else None
            selected = (
                self._tracks.get(request_id)
                if request_id is not None
                else (current or (
                    self._tracks[self._track_order[-1]]
                    if self._track_order else None
                ))
            )
            return {
                "current": current.view() if current else None,
                "track": selected.view() if selected else None,
                "running_count": len(running),
                "waiting": current is None,
            }

    # ---------------------------------------------------------------- history

    def history(self) -> dict[str, object]:
        with self._lock:
            init_ms = _ms(self._init_total_s)
            stats = {
                "count": self._count,
                "avg_ms": round(self._sum_ms / self._count, 3) if self._count else 0.0,
                "min_ms": round(self._min_ms, 3) if self._min_ms is not None else 0.0,
                "max_ms": round(self._max_ms, 3) if self._max_ms is not None else 0.0,
                "init_ms": init_ms,
            }
            return {"history": list(self._history), "stats": stats}
