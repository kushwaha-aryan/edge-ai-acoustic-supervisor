"""Optional per-operation stage timing for the local supervisor dashboard.

This module only measures time. No inference result depends on it: when no sink
is active, :func:`stage` is a no-op apart from entering/exiting a context.

A :class:`TimingSink` is bound to one operation with :func:`activate`; because
the sink lives in a :class:`~contextvars.ContextVar`, concurrent HTTP requests
(one thread each) record their stages in their own sink and can never mix
timings. All durations come from :func:`time.perf_counter`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterator

__all__ = ["TimingSink", "current_sink", "stage", "activate", "deactivate"]


@dataclass
class TimingSink:
    """Collects measured stage durations (seconds) for one operation."""

    stages: dict[str, float] = field(default_factory=dict)
    on_stage: Callable[[str], None] | None = None

    def add(self, name: str, duration_s: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + max(0.0, duration_s)

    def snapshot(self) -> dict[str, float]:
        return dict(self.stages)


current_sink: ContextVar[TimingSink | None] = ContextVar(
    "supervisor_timing_sink", default=None
)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Measure one named stage into the active sink, if any.

    The optional ``on_stage`` callback is notified on entry so the dashboard's
    activity panel reflects real backend progress, not a frontend animation.
    A failing callback must never affect the operation being measured.
    """
    sink = current_sink.get()
    if sink is None:
        yield
        return
    if sink.on_stage is not None:
        try:
            sink.on_stage(name)
        except Exception:  # noqa: BLE001 - monitoring is strictly observational
            pass
    started = perf_counter()
    try:
        yield
    finally:
        sink.add(name, perf_counter() - started)


def activate(sink: TimingSink | None) -> object:
    """Bind a sink to the current context; pass the token to :func:`deactivate`."""
    return current_sink.set(sink)


def deactivate(token: object) -> None:
    """Restore the context that existed before :func:`activate`."""
    current_sink.reset(token)  # type: ignore[arg-type]
