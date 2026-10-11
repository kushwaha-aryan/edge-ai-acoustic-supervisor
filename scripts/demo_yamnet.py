#!/usr/bin/env python
"""Demo the saved YAMNet + logistic pipeline on a WAV file or a mic capture.

Loads the frozen YAMNet SavedModel (offline copy in data/models/yamnet_hub if
present, otherwise the tfhub URL) and the saved scaler + logistic head
(worker --model-dir, default data/models/yamnet_logistic) and classifies one
audio block: mono downmix, soxr_hq resample to 16 kHz, keep only the first 15
s (identical to the training extract), per-clip mean of the 1024-d YAMNet
frame embeddings, standardized by the saved scaler and scored by the saved
classifier. Prints the top-3 classes with ASCII probability bars and the
configured simulated earbud response next to each.

Safety: every response line is labelled (SIMULATED); the demo NEVER plays
sound and NEVER contacts real emergency services. Short audio (< 1 s after
cleaning) prints a warning and is still classified; nothing is padded,
looped or otherwise altered. ASCII only; never writes any file.

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

import joblib
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

import tensorflow_hub as hub
from demo_earbud_config import EARBUD_RESPONSES

from src import config

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
EMBED_KEY = "output_1"
SCORE_KEY = "output_0"
DEFAULT_HUB_DIR = ROOT / "data" / "models" / "yamnet_hub"
DEFAULT_MODEL_ROOT = ROOT / "data" / "models" / "yamnet_logistic"
BAR_WIDTH = 24
TOP_N = 3


def _resolve_model_dir(explicit: str | None, root: Path) -> Path:
    """Return the run directory holding the saved scaler and classifier."""
    if explicit:
        candidate = Path(explicit)
        if not (candidate / "scaler.joblib").is_file():
            raise SystemExit(
                f"no saved pipeline found in {candidate} (missing "
                "scaler.joblib) - run the fit stage with --save-model first"
            )
        return candidate
    if (root / "scaler.joblib").is_file():
        return root
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if len(runs) == 1:
        return runs[0]
    raise SystemExit(
        f"cannot pick a saved pipeline under {root} (expected exactly one run "
        f"directory, found {len(runs)}); pass --model-dir explicitly"
    )


def _load_yamnet(hub_dir: Path) -> object:
    """Load the frozen YAMNet signature, preferring the offline copy."""
    attempts: list[tuple[str, object]] = []
    if hub_dir.is_dir():
        attempts.append((f"offline TF-Hub copy {hub_dir}",
                         lambda: hub.load(str(hub_dir))))
        attempts.append((f"offline SavedModel {hub_dir}",
                         lambda: tf.saved_model.load(str(hub_dir))))
    attempts.append((f"online TF-Hub {YAMNET_URL}",
                     lambda: hub.load(YAMNET_URL)))
    for label, loader in attempts:
        try:
            model = loader()
            signature = model.signatures["serving_default"]
            outputs = signature.structured_outputs
            if EMBED_KEY not in outputs or SCORE_KEY not in outputs:
                raise RuntimeError(f"YAMNet signature missing {EMBED_KEY} or "
                                   f"{SCORE_KEY}")
            print(f"YAMNet loaded from: {label}")
            return signature
        except Exception as exc:
            print(f"YAMNet load from {label} failed: "
                  f"{type(exc).__name__}: {exc}")
    raise SystemExit("YAMNet could not be loaded (offline copy missing and "
                     "no network access)")


def _load_wav(path: Path) -> np.ndarray:
    """Mono 16 kHz first-15-s float array, exactly like the training extract."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1, dtype=np.float32)
    if int(sr) != config.SAMPLE_RATE:
        y = librosa.resample(
            y,
            orig_sr=int(sr),
            target_sr=config.SAMPLE_RATE,
            res_type="soxr_hq",
        )
    y = y[: int(config.TRUNCATE_SECONDS * config.SAMPLE_RATE)]
    return np.ascontiguousarray(y, dtype=np.float32)


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


def _embed_and_predict(signature: object, y: np.ndarray,
                       scaler: object, clf: object) -> tuple[list[int],
                                                             np.ndarray]:
    """Mean frame embedding, then standardized logistic probabilities."""
    result = signature(tf.constant(y, dtype=tf.float32))
    frames = result[EMBED_KEY].numpy()
    if frames.ndim != 2 or frames.shape[1] != 1024:
        raise SystemExit(f"unexpected YAMNet embedding shape {frames.shape}")
    mean_embed = frames.mean(axis=0, dtype=np.float32).reshape(1, -1)
    proba = clf.predict_proba(scaler.transform(mean_embed))[0]
    order = np.argsort(proba)[::-1][:TOP_N].astype(int).tolist()
    return order, np.asarray(proba, dtype=np.float64)


def _validate_responses(names: list[str]) -> None:
    """Abort unless every saved class name has an earbud response."""
    missing = [name for name in names if name not in EARBUD_RESPONSES]
    if missing:
        raise SystemExit(
            f"EARBUD_RESPONSES in demo_earbud_config.py is missing classes: "
            f"{', '.join(missing)} - add one entry per class before the demo"
        )


def main() -> int:
    """Parse inputs, run the pipeline and print the top-3 SIMULATED output."""
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
    model_dir = _resolve_model_dir(args.model_dir, DEFAULT_MODEL_ROOT)
    scaler = joblib.load(model_dir / "scaler.joblib")
    clf = joblib.load(model_dir / "classifier.joblib")
    names = [
        line.strip()
        for line in (model_dir / "class_names.txt").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate_responses(names)
    signature = _load_yamnet(hub_dir)

    source_label = args.wav if args.wav else f"mic ({args.mic} s)"
    y = _load_wav(Path(args.wav)) if args.wav else _record_mic(args.mic)
    duration = y.shape[0] / config.SAMPLE_RATE
    if duration < 1.0:
        print(f"WARNING: audio is only {duration:.2f} s after cleaning "
              "(< 1 s); confidence will be low. No padding, looping or "
              "other alteration is applied.")

    order, proba = _embed_and_predict(signature, y, scaler, clf)
    print(f"classifying: {source_label} ({duration:.2f} s at "
          f"{config.SAMPLE_RATE:,} Hz mono, first "
          f"{config.TRUNCATE_SECONDS:.0f} s kept)")
    print(f"saved pipeline: {model_dir}")
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