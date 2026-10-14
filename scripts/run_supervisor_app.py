#!/usr/bin/env python
"""Launch the interactive Edge AI Acoustic Supervisor interface (local only).

Serves the single-page interface and its JSON API on 127.0.0.1 using only the
Python standard library, then prints the URL to open. The interface reuses the
project's real final inference path (YAMNet + MLP, decision D14) and the
real Phase 5 decision engine; it adds no new model, no retraining, and no live
microphone. It classifies bundled validation clips or a user-uploaded WAV file.
Everything shown is SIMULATED and no earbud hardware is controlled.

Usage:
  .venv/Scripts/python scripts/run_supervisor_app.py
  .venv/Scripts/python scripts/run_supervisor_app.py --port 8010

Then open the printed URL in a browser. Stop the server with Ctrl+C.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.server import main


if __name__ == "__main__":
    raise SystemExit(main())
