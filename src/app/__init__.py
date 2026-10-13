"""User-facing application layer for the Edge AI Acoustic Supervisor.

Wires the production inference path (`src.inference.yamnet_mlp`), the Phase 5
preference model (`src.routing.preferences`) and the decision engine
(`src.routing.supervisor`) into a locally-served, interactive supervisor
interface. This layer adds no new decision rules and no new perception: it
validates user input, resolves a small allow-list of existing validation
clips, runs the real predictor on one selected clip, and feeds the resulting
prediction through the real supervisor.

Honesty boundary (blueprint section 2A.3 and Phase 4): there is no live
microphone capture, no physical earbud control, and a presented quick action is
never executed - confirming one only records the user's choice. Everything the
interface shows is SIMULATED on the bundled validation audio.
"""

from src.app.service import (
    AppError,
    AppService,
    SampleCatalog,
    SampleClip,
    SupervisorSession,
    build_preferences,
)

__all__ = [
    "AppError",
    "AppService",
    "SampleCatalog",
    "SampleClip",
    "SupervisorSession",
    "build_preferences",
]
