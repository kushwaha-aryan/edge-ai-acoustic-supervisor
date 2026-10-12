#!/usr/bin/env python
"""Demo the final YAMNet + logistic pipeline on a WAV file or a mic capture.

CLI wrapper around the application's final prediction entry point,
`src.inference.yamnet_logistic` (decision D13, 2026-10-12). All model loading
and preprocessing live in that module; this script only handles argument
parsing, the optional microphone capture and the top-3 ASCII display. It
classifies one audio block: mono downmix, soxr_hq resample to 16 kHz, keep only
the first 15 s (identical to the training extract), per-clip mean of the 1024-d
YAMNet frame embeddings, standardized by the saved scaler and scored by the
saved logistic head. Prints the top-3 classes with ASCII probability bars and
the configured simulated earbud response next to each.

The classifier is the saved YAMNet + logistic pipeline (validation clip
accuracy 0.7769); this demo never falls back to the locked CNN or any other
model. If the final pipeline cannot load or score the input, the error is
reported explicitly and the process exits nonzero.

Safety: every response line is labelled (SIMULATED); the demo NEVER plays
sound and NEVER contacts real emergency services. Short audio (< 1 s after
cleaning) prints a warning and is still classified; nothing is padded, looped
or otherwise altered. ASCII only; never writes any file.

Usage:
  .venv/Scripts/python scripts/demo_yamnet.py path/to/sound.wav
  .venv/Scripts/python scripts/demo_yamnet.py --mic 10

The --mic capture needs the sounddevice package. It is NOT installed in this
project and this script does NOT install it: if it is missing the script
lists the pip install command and exits nonzero for manual approval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from demo_earbud_config import EARBUD_RESPONSES
from src import config
from src.inference import yamnet_logistic

DEFAULT_HUB_DIR = config.YAMNET_HUB_DIR
DEFAULT_MODEL_ROOT = config.YAMNET_LOGISTIC_ROOT
YAMNET_URL = config.YAMNET_URL
EMBED_KEY = yamnet_logistic.EMBED_KEY
SCORE_KEY = yamnet_logistic.SCORE_KEY
BAR_WIDTH = 24
TOP_N = config.YAMNET_TOP_K


def _resolve_model_dir(explicit: str | None, root: Path) -> Path:
    """Return the run directory holding the saved scaler and classifier."""
    return yamnet_logistic.resolve_model_dir(explicit, root)


def _load_yamnet(hub_dir: Path) -> object:
    """Load the frozen YAMNet signature, preferring the offline copy."""
    return yamnet_logistic.load_yamnet_signature(hub_dir)


def _load_wav(path: Path) -> np.ndarray:
    """Mono 16 kHz first-15-s float array, exactly like the training extract."""
    return yamnet_logistic.load_audio(path)


def _embed_and_predict(signature: object, y: np.ndarray,
                       scaler: object, clf: object) -> tuple[list[int],
                                                             np.ndarray]:
    """Mean frame embedding, then standardized logistic probabilities."""
    return yamnet_logistic.embed_and_predict(signature, y, scaler, clf)


def _record_mic(seconds: int) -> np.ndarray:
    """Capture seconds of mono audio through the default input device."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "mic capture needs the sounddevice package, which is not "
            "installed. Install it with `.venv/Scripts/python -m pip install "
            "sounddevice` after review, then rerun --mic. Not installing it "
            "automatically."
        ) from exc
    samples = int(seconds * config.SAMPLE_RATE)
    taken = sd.rec(samples, samplerate=config.SAMPLE_RATE, channels=1,
                   dtype="float32")
    sd.wait()
    return np.ascontiguousarray(taken[:, 0], dtype=np.float32)


def _validate_responses(names: list[str]) -> None:
    """Abort unless every saved class name has an earbud response."""
    missing = [name for name in names if name not in EARBUD_RESPONSES]
    if missing:
        raise SystemExit(
            f"EARBUD_RESPONSES in demo_earbud_config.py is missing classes: "
            f"{', '.join(missing)} - add one entry per class before the demo"
        )


def main() -> int:
    """Parse inputs, run the final pipeline and print the top-3 SIMULATED output."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "wav", nargs="?", default=None,
        help="WAV/FLAC/etc. audio file to classify",
    )
    parser.add_argument(
        "--mic", type=int, default=None, metavar="SECONDS",
        help="classify SECONDS of mic audio instead of a file (needs the "
             "sounddevice package)",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="directory with the saved scaler + classifier (default: the "
             "single run under data/models/yamnet_logistic)",
    )
    parser.add_argument(
        "--hub-dir", default=None,
        help=f"offline YAMNet SavedModel directory (default {DEFAULT_HUB_DIR})",
    )
    args = parser.parse_args()
    if (args.wav is None) == (args.mic is None):
        raise SystemExit("provide exactly one input: a WAV path or --mic "
                         "SECONDS")

    hub_dir = Path(args.hub_dir) if args.hub_dir else DEFAULT_HUB_DIR
    pipeline = yamnet_logistic.load_saved_pipeline(args.model_dir)
    names = list(pipeline.class_names)
    _validate_responses(names)
    signature = _load_yamnet(hub_dir)

    source_label = args.wav if args.wav else f"mic ({args.mic} s)"
    y = _load_wav(Path(args.wav)) if args.wav else _record_mic(args.mic)
    duration = y.shape[0] / config.SAMPLE_RATE
    if duration < 1.0:
        print(f"WARNING: audio is only {duration:.2f} s after cleaning "
              "(< 1 s); confidence will be low. No padding, looping or "
              "other alteration is applied.")

    order, proba = _embed_and_predict(signature, y, pipeline.scaler,
                                      pipeline.classifier)
    print(f"classifying: {source_label} ({duration:.2f} s at "
          f"{config.SAMPLE_RATE:,} Hz mono, first "
          f"{config.TRUNCATE_SECONDS:.0f} s kept)")
    print(f"saved pipeline: {pipeline.model_dir}")
    print(f"top {TOP_N} predictions:")
    for rank, class_ix in enumerate(order, start=1):
        prob = proba[class_ix]
        bar = "#" * int(round(prob * BAR_WIDTH))
        pad = " " * (BAR_WIDTH - len(bar))
        response = EARBUD_RESPONSES[names[class_ix]]
        print(f"  {rank}. {names[class_ix]:<14} {prob:6.1%} [{bar}{pad}] "
              f"response (SIMULATED): {response}")
    print("All responses above are SIMULATED; this demo never plays sound "
          "and never contacts emergency services.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
