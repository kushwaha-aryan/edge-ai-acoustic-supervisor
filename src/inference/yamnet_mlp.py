#!/usr/bin/env python
"""Production prediction entry point: frozen YAMNet embeddings + MLP head.

This is the application's official final / best-performing inference path
(decision D14, 2026-10-14; blueprint section 2.6 Exception / section 2.7). It
supersedes the YAMNet + logistic path (decision D13,
`src/inference/yamnet_logistic`), which is kept unchanged as an archived
reference model. This module is the single source of truth for the production
flow and is used by the CLI/demo layer (`scripts/demo_yamnet.py`,
`scripts/demo_table.py`) and by the application service
(`src.app.service`). It does NOT implement and NEVER falls back to the logistic
head or the locked CNN (that CNN remains the on-device model only).

Prediction flow, from input audio to output label:
  1. read the audio, mean-mono downmix, soxr_hq resample to 16 kHz, keep only
     the first 15 s (identical to the training extract),
  2. extract the frozen YAMNet frame embeddings (1024-d) and average them over
     the clip,
  3. standardize the mean embedding with the saved StandardScaler,
  4. score it with the saved tf.keras MLP head, which returns a 17-class
     softmax,
  5. map the argmax index to a project class label via class_names.txt.

The YAMNet preprocessing and embedding primitives (`load_audio`, `embed_clip`,
`load_yamnet_signature`, `resolve_hub_dir`) are shared with
`src.inference.yamnet_logistic` so there is exactly one YAMNet implementation
and the production embeddings can never diverge from the training extract; only
the scoring head differs.

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
from src.inference import yamnet_logistic as _logistic

SCALER_NAME = "scaler.joblib"
HEAD_NAME = "mlp_head.keras"
CLASS_NAMES_NAME = "class_names.txt"
EMBED_KEY = _logistic.EMBED_KEY
SCORE_KEY = _logistic.SCORE_KEY
EMBED_DIM = config.YAMNET_EMBED_DIM
TOP_K = config.YAMNET_TOP_K
YAMNET_URL = config.YAMNET_URL


class FinalPipelineError(RuntimeError):
    """A final-pipeline artifact, input or consistency problem (explicit)."""


@dataclass(frozen=True)
class SavedPipeline:
    """The loaded scaler + MLP head + class labels of one saved run."""

    model_dir: Path
    scaler: object
    head: object
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
    return _logistic.resolve_hub_dir(explicit)


def resolve_model_dir(explicit: str | Path | None = None,
                      root: str | Path | None = None) -> Path:
    """Return the run directory holding the saved scaler and MLP head.

    `root` defaults to the run named by config (YAMNET_MLP_DIR). If it holds no
    scaler directly, a single run subdirectory is accepted; anything else is an
    explicit error rather than a guess.
    """
    root = Path(root) if root is not None else config.YAMNET_MLP_DIR
    if explicit:
        candidate = Path(explicit)
        if not (candidate / SCALER_NAME).is_file():
            raise FinalPipelineError(
                f"no saved pipeline found in {candidate} (missing "
                f"{SCALER_NAME}) - run train_yamnet_mlp.py with --confirm-train "
                "and promote the run first"
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


def _shape_last_dim(shape: object) -> int | None:
    """Return the last (feature/class) dimension of a Keras shape or None."""
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], tuple):
        shape = shape[0]
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    last = shape[-1]
    try:
        return int(last)
    except (TypeError, ValueError):
        return None


def _head_dims(head: object) -> tuple[int | None, int | None]:
    """Return the (input_dim, output_dim) of a loaded Keras model."""
    in_dim = _shape_last_dim(getattr(head, "input_shape", None))
    out_dim = _shape_last_dim(getattr(head, "output_shape", None))
    return in_dim, out_dim


def load_saved_pipeline(model_dir: str | Path | None = None) -> SavedPipeline:
    """Load and validate the saved scaler, MLP head and class names.

    Raises FinalPipelineError if any required artifact is missing or if the
    artifacts disagree with the embedding dim / class count / class ordering the
    production pipeline requires.
    """
    run_dir = resolve_model_dir(model_dir)
    scaler_path = run_dir / SCALER_NAME
    head_path = run_dir / HEAD_NAME
    names_path = run_dir / CLASS_NAMES_NAME
    missing = [p.name for p in (scaler_path, head_path, names_path)
               if not p.is_file()]
    if missing:
        raise FinalPipelineError(
            f"missing artifacts in {run_dir}: {', '.join(missing)} - run "
            "train_yamnet_mlp.py with --confirm-train and promote the run first"
        )
    try:
        import tensorflow as tf

        scaler = joblib.load(scaler_path)
        head = tf.keras.models.load_model(str(head_path))
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
    in_dim, out_dim = _head_dims(head)
    if in_dim is not None and in_dim != EMBED_DIM:
        raise FinalPipelineError(
            f"MLP head expects {in_dim} input features but the YAMNet embedding "
            f"is {EMBED_DIM}-d"
        )
    if out_dim is not None and out_dim != len(names):
        raise FinalPipelineError(
            f"MLP head outputs {out_dim} classes but class_names.txt lists "
            f"{len(names)}"
        )
    if names != config.TRIGGER_CLASS_NAMES:
        raise FinalPipelineError(
            "class_names.txt does not match the project trigger taxonomy in "
            "src/config.py TRIGGER_CLASS_NAMES; refusing to label predictions "
            "with a mismatched class list"
        )
    return SavedPipeline(run_dir, scaler, head, names)


def load_yamnet_signature(hub_dir: str | Path | None = None) -> object:
    """Load the frozen YAMNet serving signature (shared with the logistic path)."""
    return _logistic.load_yamnet_signature(hub_dir)


def load_audio(path: str | Path) -> np.ndarray:
    """Mono 16 kHz first-15-s float array, exactly like the training extract."""
    return _logistic.load_audio(path)


def embed_clip(signature: object, y: np.ndarray) -> np.ndarray:
    """Return the (1, EMBED_DIM) mean YAMNet frame embedding of one clip."""
    return _logistic.embed_clip(signature, y)


def _standardized_proba(scaler: object, head: object,
                        mean_embed: np.ndarray) -> np.ndarray:
    if mean_embed.shape[1] != EMBED_DIM:
        raise FinalPipelineError(
            f"embedding has {mean_embed.shape[1]} features, expected "
            f"{EMBED_DIM}"
        )
    scaled = scaler.transform(mean_embed).astype(np.float32)
    proba = head.predict(scaled, batch_size=1, verbose=0)[0]
    return np.asarray(proba, dtype=np.float64)


def embed_and_predict(signature: object, y: np.ndarray, scaler: object,
                      head: object, top_k: int = TOP_K
                      ) -> tuple[list[int], np.ndarray]:
    """Mean frame embedding, then standardized MLP softmax probabilities.

    Returns (top_k indices in descending probability, full probability vector).
    Kept as a primitive so the bulk demo can reuse the exact same scoring.
    """
    mean_embed = embed_clip(signature, y)
    proba = _standardized_proba(scaler, head, mean_embed)
    order = np.argsort(proba)[::-1][:top_k].astype(int).tolist()
    return order, proba


def predict_proba(pipeline: SavedPipeline, signature: object,
                  y: np.ndarray) -> np.ndarray:
    """Return the per-class probabilities of one clip in class_names order."""
    mean_embed = embed_clip(signature, y)
    return _standardized_proba(pipeline.scaler, pipeline.head, mean_embed)


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
    """Caches the loaded production pipeline and YAMNet signature across calls."""

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
        """Load one audio file and classify it through the production pipeline."""
        return self.predict_audio(load_audio(path))


def predict_file(path: str | Path, model_dir: str | Path | None = None,
                 hub_dir: str | Path | None = None) -> Prediction:
    """Convenience: classify one audio file with a fresh single-use Predictor."""
    return Predictor(model_dir=model_dir, hub_dir=hub_dir).predict_file(path)
