#!/usr/bin/env python
"""Final prediction entry point: frozen YAMNet embeddings + logistic head.

This is the application's official final / best-performing inference path
(decision D13, 2026-10-12; blueprint section 2.6 Exception / section 2.7). It is
the single source of truth for the final prediction flow and is used by the
CLI/demo layer (`scripts/demo_yamnet.py`, `scripts/demo_table.py`). It does NOT
implement and NEVER falls back to the locked CNN (that CNN remains the on-device
model only) or to any other classifier.

Prediction flow, from input audio to output label:
  1. read the audio, mean-mono downmix, soxr_hq resample to 16 kHz, keep only
     the first 15 s (identical to the training extract),
  2. extract the frozen YAMNet frame embeddings (1024-d) and average them over
     the clip,
  3. standardize the mean embedding with the saved StandardScaler,
  4. score it with the saved multinomial LogisticRegression,
  5. map the argmax index to a project class label via class_names.txt.

Every step loads the existing trained artifacts from paths derived from
`src.config` (never from the working directory) and never retrains, alters, or
regenerates anything. The test split is never accessed. Failures (missing
artifacts, unreadable audio, wrong embedding shape, mismatched class ordering)
raise `FinalPipelineError` with an explicit message; nothing is swallowed and
nothing silently degrades to a different model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src import config

SCALER_NAME = "scaler.joblib"
CLASSIFIER_NAME = "classifier.joblib"
CLASS_NAMES_NAME = "class_names.txt"
EMBED_KEY = "output_1"
SCORE_KEY = "output_0"
EMBED_DIM = config.YAMNET_EMBED_DIM
TOP_K = config.YAMNET_TOP_K
YAMNET_URL = config.YAMNET_URL


class FinalPipelineError(RuntimeError):
    """A final-pipeline artifact, input or consistency problem (explicit)."""


@dataclass(frozen=True)
class SavedPipeline:
    """The loaded scaler + classifier + class labels of one saved run."""

    model_dir: Path
    scaler: object
    classifier: object
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def feature_dim(self) -> int:
        return int(getattr(self.scaler, "n_features_in_", EMBED_DIM))


@dataclass(frozen=True)
class Prediction:
    """One clip-level prediction in class_names.txt index order."""

    label: str
    index: int
    confidence: float
    probabilities: np.ndarray
    class_names: tuple[str, ...]

    def top_k(self, k: int = TOP_K) -> list[tuple[str, float]]:
        """Return the (label, probability) pairs of the k highest classes."""
        order = np.argsort(self.probabilities)[::-1][:k]
        return [(self.class_names[int(i)], float(self.probabilities[int(i)]))
                for i in order]


def resolve_hub_dir(explicit: str | Path | None = None) -> Path:
    """Return the YAMNet SavedModel directory (config default when unset)."""
    return Path(explicit) if explicit else config.YAMNET_HUB_DIR


def resolve_model_dir(explicit: str | Path | None = None,
                      root: str | Path | None = None) -> Path:
    """Return the run directory holding the saved scaler and classifier.

    `root` defaults to the run named by config (YAMNET_LOGISTIC_DIR). If it
    holds no scaler directly, a single run subdirectory is accepted; anything
    else is an explicit error rather than a guess.
    """
    root = Path(root) if root is not None else config.YAMNET_LOGISTIC_DIR
    if explicit:
        candidate = Path(explicit)
        if not (candidate / SCALER_NAME).is_file():
            raise FinalPipelineError(
                f"no saved pipeline found in {candidate} (missing "
                f"{SCALER_NAME}) - run the fit stage with --save-model first"
            )
        return candidate
    if (root / SCALER_NAME).is_file():
        return root
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if len(runs) == 1:
        return runs[0]
    raise FinalPipelineError(
        f"cannot pick a saved pipeline under {root} (expected exactly one run "
        f"directory, found {len(runs)}); pass an explicit model directory"
    )


def load_saved_pipeline(model_dir: str | Path | None = None) -> SavedPipeline:
    """Load and validate the saved scaler, classifier and class names.

    Raises FinalPipelineError if any required artifact is missing or if the
    artifacts disagree with the embedding dim / class count / class ordering the
    final pipeline requires.
    """
    run_dir = resolve_model_dir(model_dir)
    scaler_path = run_dir / SCALER_NAME
    clf_path = run_dir / CLASSIFIER_NAME
    names_path = run_dir / CLASS_NAMES_NAME
    missing = [p.name for p in (scaler_path, clf_path, names_path)
               if not p.is_file()]
    if missing:
        raise FinalPipelineError(
            f"missing artifacts in {run_dir}: {', '.join(missing)} - run the "
            "fit stage with --save-model first"
        )
    try:
        scaler = joblib.load(scaler_path)
        clf = joblib.load(clf_path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise FinalPipelineError(
            f"could not load the saved pipeline in {run_dir}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    names = tuple(
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not names:
        raise FinalPipelineError(f"class_names.txt in {run_dir} is empty")

    scaler_dim = int(getattr(scaler, "n_features_in_", EMBED_DIM))
    if scaler_dim != EMBED_DIM:
        raise FinalPipelineError(
            f"scaler expects {scaler_dim} features but the YAMNet embedding is "
            f"{EMBED_DIM}-d"
        )
    clf_dim = int(getattr(clf, "n_features_in_", EMBED_DIM))
    if clf_dim != EMBED_DIM:
        raise FinalPipelineError(
            f"classifier expects {clf_dim} features but the YAMNet embedding "
            f"is {EMBED_DIM}-d"
        )
    coef = np.shape(getattr(clf, "coef_", ()))
    if coef != (len(names), EMBED_DIM):
        raise FinalPipelineError(
            f"classifier coefficient shape {coef} does not match "
            f"{len(names)} classes x {EMBED_DIM} features"
        )
    classes = [int(c) for c in np.asarray(getattr(clf, "classes_", []))]
    if classes != list(range(len(names))):
        raise FinalPipelineError(
            f"classifier classes_ {classes} are not the index order "
            f"0..{len(names) - 1} that class_names.txt assumes"
        )
    if names != config.TRIGGER_CLASS_NAMES:
        raise FinalPipelineError(
            "class_names.txt does not match the project trigger taxonomy in "
            "src/config.py TRIGGER_CLASS_NAMES; refusing to label predictions "
            "with a mismatched class list"
        )
    return SavedPipeline(run_dir, scaler, clf, names)


def load_yamnet_signature(hub_dir: str | Path | None = None) -> object:
    """Load the frozen YAMNet serving signature, preferring the offline copy.

    Tries the offline TF-Hub copy first, then the offline SavedModel, then the
    online TF-Hub URL. Raises FinalPipelineError if none expose the needed
    outputs. TensorFlow is imported here so the rest of the module stays
    importable without it.
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    hub_path = resolve_hub_dir(hub_dir)
    attempts: list[tuple[str, object]] = []
    if hub_path.is_dir():
        attempts.append((f"offline TF-Hub copy {hub_path}",
                         lambda: hub.load(str(hub_path))))
        attempts.append((f"offline SavedModel {hub_path}",
                         lambda: tf.saved_model.load(str(hub_path))))
    attempts.append((f"online TF-Hub {YAMNET_URL}", lambda: hub.load(YAMNET_URL)))
    for label, loader in attempts:
        try:
            model = loader()
            signature = model.signatures["serving_default"]
            outputs = signature.structured_outputs
            if EMBED_KEY not in outputs or SCORE_KEY not in outputs:
                raise FinalPipelineError(
                    f"YAMNet signature missing {EMBED_KEY} or {SCORE_KEY}"
                )
            print(f"YAMNet loaded from: {label}")
            return signature
        except Exception as exc:  # noqa: BLE001 - try the next source
            print(f"YAMNet load from {label} failed: "
                  f"{type(exc).__name__}: {exc}")
    raise FinalPipelineError(
        "YAMNet could not be loaded (offline copy missing or invalid and no "
        "network access)"
    )


def load_audio(path: str | Path) -> np.ndarray:
    """Mono 16 kHz first-15-s float array, exactly like the training extract."""
    import librosa
    import soundfile as sf

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FinalPipelineError(f"audio file not found: {audio_path}")
    try:
        y, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise FinalPipelineError(
            f"could not read audio {audio_path}: {type(exc).__name__}: {exc}"
        ) from exc
    y = y.mean(axis=1, dtype=np.float32)
    if int(sr) != config.SAMPLE_RATE:
        y = librosa.resample(
            y,
            orig_sr=int(sr),
            target_sr=config.SAMPLE_RATE,
            res_type="soxr_hq",
        )
    y = y[: int(config.TRUNCATE_SECONDS * config.SAMPLE_RATE)]
    if y.shape[0] == 0:
        raise FinalPipelineError(f"audio {audio_path} is empty after cleaning")
    return np.ascontiguousarray(y, dtype=np.float32)


def embed_clip(signature: object, y: np.ndarray) -> np.ndarray:
    """Return the (1, EMBED_DIM) mean YAMNet frame embedding of one clip."""
    import tensorflow as tf

    result = signature(tf.constant(y, dtype=tf.float32))
    frames = result[EMBED_KEY].numpy()
    if frames.ndim != 2 or frames.shape[1] != EMBED_DIM:
        raise FinalPipelineError(
            f"unexpected YAMNet embedding shape {frames.shape}; expected "
            f"(n_frames, {EMBED_DIM})"
        )
    return frames.mean(axis=0, dtype=np.float32).reshape(1, -1)


def _standardized_proba(scaler: object, clf: object,
                        mean_embed: np.ndarray) -> np.ndarray:
    if mean_embed.shape[1] != EMBED_DIM:
        raise FinalPipelineError(
            f"embedding has {mean_embed.shape[1]} features, expected "
            f"{EMBED_DIM}"
        )
    proba = clf.predict_proba(scaler.transform(mean_embed))[0]
    return np.asarray(proba, dtype=np.float64)


def embed_and_predict(signature: object, y: np.ndarray, scaler: object,
                      clf: object, top_k: int = TOP_K
                      ) -> tuple[list[int], np.ndarray]:
    """Mean frame embedding, then standardized logistic probabilities.

    Returns (top_k indices in descending probability, full probability vector).
    Kept as a primitive so the bulk demo can reuse the exact same scoring.
    """
    mean_embed = embed_clip(signature, y)
    proba = _standardized_proba(scaler, clf, mean_embed)
    order = np.argsort(proba)[::-1][:top_k].astype(int).tolist()
    return order, proba


def predict_proba(pipeline: SavedPipeline, signature: object,
                  y: np.ndarray) -> np.ndarray:
    """Return the per-class probabilities of one clip in class_names order."""
    mean_embed = embed_clip(signature, y)
    return _standardized_proba(pipeline.scaler, pipeline.classifier, mean_embed)


def predict(pipeline: SavedPipeline, signature: object,
            y: np.ndarray) -> Prediction:
    """Return the full Prediction (label, confidence, probabilities) of a clip."""
    proba = predict_proba(pipeline, signature, y)
    index = int(np.argmax(proba))
    return Prediction(
        label=pipeline.class_names[index],
        index=index,
        confidence=float(proba[index]),
        probabilities=proba,
        class_names=pipeline.class_names,
    )


class Predictor:
    """Caches the loaded final pipeline and YAMNet signature across calls."""

    def __init__(self, model_dir: str | Path | None = None,
                 hub_dir: str | Path | None = None):
        self.pipeline = load_saved_pipeline(model_dir)
        self._hub_dir = resolve_hub_dir(hub_dir)
        self._signature: object | None = None

    @property
    def signature(self) -> object:
        if self._signature is None:
            self._signature = load_yamnet_signature(self._hub_dir)
        return self._signature

    def predict_audio(self, y: np.ndarray) -> Prediction:
        """Classify an already-loaded mono 16 kHz clip."""
        return predict(self.pipeline, self.signature, y)

    def predict_file(self, path: str | Path) -> Prediction:
        """Load one audio file and classify it through the final pipeline."""
        return self.predict_audio(load_audio(path))


def predict_file(path: str | Path, model_dir: str | Path | None = None,
                 hub_dir: str | Path | None = None) -> Prediction:
    """Convenience: classify one audio file with a fresh single-use Predictor."""
    return Predictor(model_dir=model_dir, hub_dir=hub_dir).predict_file(path)
