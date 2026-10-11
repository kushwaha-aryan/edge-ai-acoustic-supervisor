#!/usr/bin/env python
"""Train the experimental per-class binary CNN + MLP head pipeline.

Experimental architecture (isolated in experiment_binary_cnn_mlp/): this is the
same two-stage pipeline as experiment_binary_cnn_lr/ except the second stage is
a small tf.keras MLP instead of a sklearn LogisticRegression. The first stage
is unchanged: 17 independent binary CNNs, one per trigger class, each deciding
whether its class is present in a window, and their 17 sigmoid outputs form a
17-dim feature vector. The head that consumes that 17-dim vector is replaced
here by an MLP (Dense hidden -> relu -> Dense 17 softmax) trained on the locked
Phase 2 (preset A) artifacts in data/processed/ (features_<split>.npy,
labels_<split>.npy), cross-checked against feature_stats.json with the same
checks as scripts/run_phase3a_training.py. The test split is never touched
(Phase 3B evaluation only).

The 17 binary CNNs are NOT retrained here. They are loaded frozen from a
previous CNN + logistic run (default experiment_binary_cnn_lr/runs/exp01) and
used only as a feature extractor, so the two experiments share an identical
first stage and differ only in the head. This is why the run is fast: the
expensive part (training 17 CNNs) is skipped and reused.

Each window's 17 binary-CNN outputs are batched through predict() to build
Z_train and Z_val (N x 17). Because the CNNs are frozen and the input windows
are fixed, this Z is identical to what the CNN + logistic run used; the script
caches it once (z_train.npy / z_val.npy under --cache-dir, keyed by a small
fingerprint) so later head experiments reload it instantly instead of redoing
the forward pass. The MLP is then fitted on Z_train and validation is reported
at window level and at clip level with the project's clip_mean_softmax
aggregation (mean MLP softmax per clip, then argmax), grouped through
window_index_val.csv.

Every run needs a unique --run-name. Artifacts are written to
experiment_binary_cnn_mlp/runs/<run-name>/:
  class_<id>_<class_name>.keras   copies of the frozen first-stage CNNs
  mlp_head.keras                  the trained MLP head
  class_names.txt                 the 17 class names in id order
  meta.json                       run metadata and validation metrics
If that directory already exists the script aborts before doing any work, so
existing experiment runs are never overwritten. Nothing in data/ is rewritten
and no existing file is modified; this experiment is fully contained in
experiment_binary_cnn_mlp/.

Safety guards: model.fit() runs only when --confirm-train is passed; without
it the script stops after the setup checks and saves nothing. --dry-run builds
the MLP head, prints every layer output shape and the parameter count, then
exits without compiling, training or saving. --source-run must point at a
completed CNN + logistic run whose own runs/<name>/ holds class_names.txt and
the 17 class_<id>_<name>.keras files.

Usage:
  .venv/Scripts/python experiment_binary_cnn_mlp/train_binary_cnn_mlp.py \
      --run-name mlp01 --confirm-train
  .venv/Scripts/python experiment_binary_cnn_mlp/train_binary_cnn_mlp.py \
      --run-name mlp01 --confirm-train --hidden 128 --epochs 60
  .venv/Scripts/python experiment_binary_cnn_mlp/train_binary_cnn_mlp.py \
      --run-name mlp01 --confirm-train --source-run \
      experiment_binary_cnn_lr/runs/exp01
  .venv/Scripts/python experiment_binary_cnn_mlp/train_binary_cnn_mlp.py \
      --run-name dryrun --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config

DEFAULT_EPOCHS = 50
DEFAULT_HIDDEN = 64
PREDICT_BATCH = 512
RUNS_ROOT = Path(__file__).resolve().parent / "runs"
CACHE_ROOT = Path(__file__).resolve().parent / "cache"
DEFAULT_SOURCE_RUN = ROOT / "experiment_binary_cnn_lr" / "runs" / "exp01"
SPEC = config.PRESETS[config.PRESET_A]
NUM_CLASSES = config.NUM_TRIGGER_CLASSES


def _validate_run_name(raw: str) -> str:
    """Return a safe bare run name (no directories) or raise ValueError."""
    name = raw.strip()
    if not name:
        raise ValueError("--run-name must be a non-empty name")
    if name in {".", ".."}:
        raise ValueError("--run-name must not be '.' or '..'")
    bad = [ch for ch in '/\\<>:"|?*' if ch in name]
    if bad:
        raise ValueError(
            "--run-name must not contain path separators or Windows-unsafe "
            f"characters (found {bad!r})"
        )
    return name


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split's feature array and integer labels for preset A."""
    features = np.load(SPEC["processed_dir"] / f"features_{split}.npy")
    labels = np.load(SPEC["processed_dir"] / f"labels_{split}.npy").astype(np.int32)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split}: {features.shape[0]} feature rows vs {labels.shape[0]} labels"
        )
    return features, labels


def _load_feature_stats() -> dict:
    """Read the frozen preprocessing record for preset A."""
    with SPEC["feature_stats_path"].open(encoding="utf-8") as handle:
        return json.load(handle)


def _verify_split(
    stats: dict, split: str, features: np.ndarray, labels: np.ndarray
) -> None:
    """Fail loudly if the arrays disagree with the locked preprocessing record."""
    if list(features.shape[1:]) != list(SPEC["spec_shape"]):
        raise ValueError(
            f"{split}: feature shape {features.shape[1:]} != "
            f"preset {config.PRESET_A} {SPEC['spec_shape']}"
        )
    recorded = stats["splits"][split]
    if int(recorded["windows"]) != features.shape[0]:
        raise ValueError(
            f"{split}: {features.shape[0]} rows vs "
            f"{recorded['windows']} windows in feature_stats.json"
        )
    if int(labels.min()) < 0 or int(labels.max()) >= NUM_CLASSES:
        raise ValueError(f"{split}: labels outside [0, {NUM_CLASSES})")
    if features.dtype != np.float32:
        raise ValueError(f"{split}: dtype {features.dtype} != float32")
    if int(stats["sample_rate"]) != config.SAMPLE_RATE:
        raise ValueError("feature_stats.json sample_rate disagrees with config")
    if int(stats["n_mels"]) != int(SPEC["n_mels"]):
        raise ValueError(
            f"feature_stats.json n_mels disagrees with preset {config.PRESET_A}"
        )


def _resolve_source_run(raw: str) -> Path:
    """Return the completed CNN + logistic run directory to reuse."""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    if not candidate.is_dir():
        raise SystemExit(f"--source-run directory not found: {candidate}")
    if not (candidate / "class_names.txt").is_file():
        raise SystemExit(
            f"no class_names.txt in {candidate} - --source-run must point at a "
            "completed CNN + logistic run"
        )
    return candidate


def _source_class_names(source_run: Path) -> list[str]:
    """Read the 17 class names stored with the frozen CNNs."""
    names = [
        line.strip()
        for line in (source_run / "class_names.txt").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != NUM_CLASSES:
        raise SystemExit(
            f"{source_run / 'class_names.txt'} has {len(names)} names, "
            f"expected {NUM_CLASSES}"
        )
    return names


def _source_cnn_paths(source_run: Path, names: list[str]) -> list[Path]:
    """Return the class_<id>_<name>.keras path for each frozen CNN, checked."""
    fill = len(str(NUM_CLASSES - 1))
    paths: list[Path] = []
    for class_id, class_name in enumerate(names):
        path = source_run / f"class_{class_id:0{fill}d}_{class_name}.keras"
        if not path.is_file():
            raise SystemExit(f"missing frozen binary CNN: {path}")
        paths.append(path)
    return paths


def _load_frozen_cnns(paths: list[Path]) -> list[tf.keras.Model]:
    """Load the frozen first-stage CNNs without touching their weights."""
    models: list[tf.keras.Model] = []
    for path in paths:
        models.append(tf.keras.models.load_model(str(path)))
    return models


def _extract_z(models: list[tf.keras.Model], features: np.ndarray) -> np.ndarray:
    """Stack the 17 binary-CNN sigmoid outputs into an N x 17 feature matrix."""
    chunks = [
        model.predict(features, batch_size=PREDICT_BATCH, verbose=0)[:, 0]
        for model in models
    ]
    return np.column_stack(chunks).astype(np.float32)


def _cache_fingerprint(
    source_run: Path, names: list[str], train_rows: int, val_rows: int
) -> dict:
    """Small descriptor that must match for a cached z matrix to be reused."""
    return {
        "source_run": source_run.name,
        "class_names": names,
        "num_classes": NUM_CLASSES,
        "spec_shape": list(SPEC["spec_shape"]),
        "preset": config.PRESET_A,
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
    }


def _load_cache(
    cache_dir: Path, fingerprint: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return cached (z_train, z_val) when the fingerprint matches, else None."""
    meta_path = cache_dir / "z_fingerprint.json"
    z_train_path = cache_dir / "z_train.npy"
    z_val_path = cache_dir / "z_val.npy"
    if not (meta_path.is_file() and z_train_path.is_file() and z_val_path.is_file()):
        return None
    try:
        recorded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if recorded != fingerprint:
        return None
    z_train = np.load(z_train_path)
    z_val = np.load(z_val_path)
    if (z_train.shape[0] != fingerprint["train_rows"]
            or z_val.shape[0] != fingerprint["val_rows"]
            or z_train.shape[1] != NUM_CLASSES
            or z_val.shape[1] != NUM_CLASSES):
        return None
    return z_train, z_val


def _save_cache(
    cache_dir: Path, fingerprint: dict, z_train: np.ndarray, z_val: np.ndarray
) -> None:
    """Persist the z matrices plus the fingerprint that makes them reusable."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "z_train.npy", z_train)
    np.save(cache_dir / "z_val.npy", z_val)
    (cache_dir / "z_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )


def _balanced_weights(labels: np.ndarray) -> dict[int, float]:
    """Balanced class weights for a multiclass target: w = N / (K * n_c)."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = int(labels.shape[0])
    return {
        int(class_id): float(total / (NUM_CLASSES * int(counts[class_id])))
        for class_id in range(NUM_CLASSES)
        if int(counts[class_id]) > 0
    }


def build_mlp(hidden: int) -> tf.keras.Model:
    """Build the head: 17 binary outputs -> hidden relu -> 17 softmax."""
    inputs = tf.keras.Input(shape=(NUM_CLASSES,), name="binary_outputs")
    x = tf.keras.layers.Dense(hidden, activation="relu", name="hidden")(inputs)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="class_proba"
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="binary_mlp_head")


class EpochTimer(tf.keras.callbacks.Callback):
    """Record and print wall-clock seconds for every completed epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_seconds: list[float] = []
        self._started = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._started
        self.epoch_seconds.append(elapsed)
        print(f"epoch {epoch + 1} elapsed time: {elapsed:8.1f} s", flush=True)


def _print_history(epoch_seconds: list[float], history: dict) -> None:
    """Print the per-epoch metric table (time, train/val loss and accuracy)."""
    print("", flush=True)
    print(
        f"{'epoch':>5}  {'seconds':>8}  {'train_loss':>10}  {'train_acc':>9}"
        f"  {'val_loss':>10}  {'val_acc':>8}",
        flush=True,
    )
    for i, train_loss in enumerate(history["loss"]):
        print(
            f"{i + 1:>5}  {epoch_seconds[i]:>8.1f}  {train_loss:>10.4f}  "
            f"{history['accuracy'][i]:>9.4f}  {history['val_loss'][i]:>10.4f}  "
            f"{history['val_accuracy'][i]:>8.4f}",
            flush=True,
        )
    print("", flush=True)


def _clip_accuracy(probabilities: np.ndarray, index_path: Path) -> float:
    """Clip-level accuracy via clip_mean_softmax over window_index rows."""
    window_index = pd.read_csv(
        index_path, usecols=["clip_id", "class_id"],
        dtype={"clip_id": str, "class_id": int},
    )
    if len(window_index) != probabilities.shape[0]:
        raise ValueError(
            f"window_index rows {len(window_index)} != probabilities "
            f"{probabilities.shape[0]}"
        )
    frame = pd.DataFrame(
        {
            "clip_id": window_index["clip_id"].to_numpy(),
            "class_id": window_index["class_id"].to_numpy(),
            "proba": list(probabilities),
        }
    )
    predicted = frame.groupby("clip_id")["proba"].aggregate(
        lambda block: int(np.asarray(block).mean(axis=0).argmax())
    )
    true = frame.groupby("clip_id")["class_id"].first()
    return float(np.mean(predicted.to_numpy() == true.to_numpy()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="train the experimental 17 frozen binary CNNs + MLP head"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="NAME",
        help="unique run name; artifacts are saved to "
        "experiment_binary_cnn_mlp/runs/NAME/ (aborts if that directory "
        "already exists)",
    )
    parser.add_argument(
        "--source-run",
        default=str(DEFAULT_SOURCE_RUN),
        metavar="DIR",
        help="completed CNN + logistic run directory holding the frozen "
        f"first-stage CNNs (default: {DEFAULT_SOURCE_RUN})",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=DEFAULT_HIDDEN,
        help=f"hidden units in the MLP head (default: {DEFAULT_HIDDEN})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"training epochs for the MLP head (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(CACHE_ROOT),
        metavar="DIR",
        help="directory for the cached z matrices (default: "
        "experiment_binary_cnn_mlp/cache)",
    )
    parser.add_argument(
        "--balanced-class-weights",
        action="store_true",
        help="fit the MLP with balanced class weights; omitted by default so "
        "the head matches the unweighted logistic head of the other experiment",
    )
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="required to run model.fit(); without it the script stops after "
        "the setup checks and saves nothing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the MLP head, print every layer output shape and the "
        "parameter count, then exit without compiling, training or saving",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.hidden < 1:
        parser.error("--hidden must be >= 1")
    try:
        run_name = _validate_run_name(args.run_name)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = RUNS_ROOT / run_name
    if run_dir.exists():
        raise SystemExit(
            f"run name already in use: {run_dir} exists - pick a different "
            "--run-name; existing experiment runs are never overwritten"
        )
    source_run = _resolve_source_run(args.source_run)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (ROOT / cache_dir).resolve()

    started = time.perf_counter()
    tf.keras.utils.set_random_seed(config.SPLIT_SEED)

    print("experiment: 17 frozen binary CNNs + single MLP head", flush=True)
    print(f"tensorflow {tf.__version__}", flush=True)
    print(
        f"run: {run_name}, hidden: {args.hidden}, epochs: {args.epochs}, "
        f"batch_size: {config.BATCH_SIZE}, lr: {config.LEARNING_RATE}, "
        f"balanced_weights: "
        f"{'yes' if args.balanced_class_weights else 'no'}, "
        f"confirm_train: {'yes' if args.confirm_train else 'no'}, "
        f"dry_run: {'yes' if args.dry_run else 'no'}",
        flush=True,
    )
    print(f"reusing frozen first stage from: {source_run}", flush=True)

    stats = _load_feature_stats()
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")
    _verify_split(stats, "train", train_x, train_y)
    _verify_split(stats, "val", val_x, val_y)
    print(
        f"data: preset {config.PRESET_A}, train {train_x.shape[0]} windows, "
        f"val {val_x.shape[0]} windows, input {SPEC['spec_shape']}, "
        f"classes {NUM_CLASSES}",
        flush=True,
    )

    if args.dry_run:
        head = build_mlp(args.hidden)
        print(
            f"dry run: model input (N, {NUM_CLASSES}) of 17 binary-CNN outputs",
            flush=True,
        )
        print("layer output shapes:", flush=True)
        for layer in head.layers:
            print(
                f"  {layer.name:<22} {layer.__class__.__name__:<20} "
                f"{tuple(layer.output.shape)}",
                flush=True,
            )
        print(
            f"parameters in the MLP head: {int(head.count_params()):,}",
            flush=True,
        )
        print("dry run complete: no compile, no fit, no model saved.", flush=True)
        return 0

    if not args.confirm_train:
        print(
            "--confirm-train not given: setup checks passed, the MLP head is "
            "not trained and nothing is saved.",
            flush=True,
        )
        return 0

    names = _source_class_names(source_run)
    cnn_paths = _source_cnn_paths(source_run, names)
    fingerprint = _cache_fingerprint(
        source_run, names, train_x.shape[0], val_x.shape[0]
    )
    cached = _load_cache(cache_dir, fingerprint)
    if cached is not None:
        z_train, z_val = cached
        print(
            f"loaded cached z from {cache_dir} "
            f"(Z_train {tuple(z_train.shape)}, Z_val {tuple(z_val.shape)})",
            flush=True,
        )
    else:
        print(
            f"loading {len(cnn_paths)} frozen binary CNNs from {source_run} ...",
            flush=True,
        )
        models = _load_frozen_cnns(cnn_paths)
        print("building Z_train / Z_val from the frozen binary CNNs ...",
              flush=True)
        z_train = _extract_z(models, train_x)
        z_val = _extract_z(models, val_x)
        print(
            f"Z_train {tuple(z_train.shape)}, Z_val {tuple(z_val.shape)} "
            "(each window -> 17 binary-CNN outputs)",
            flush=True,
        )
        _save_cache(cache_dir, fingerprint, z_train, z_val)
        print(f"cached z -> {cache_dir} (reused by later head experiments)",
              flush=True)

    head = build_mlp(args.hidden)
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    weights = _balanced_weights(train_y) if args.balanced_class_weights else None
    timer = EpochTimer()
    callbacks = [
        timer,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
    ]
    print("fitting the MLP head on Z_train ...", flush=True)
    history = head.fit(
        z_train,
        train_y,
        validation_data=(z_val, val_y),
        epochs=args.epochs,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    _print_history(timer.epoch_seconds, history.history)

    head_proba = head.predict(z_val, batch_size=PREDICT_BATCH, verbose=0)
    window_pred = np.argmax(head_proba, axis=1)
    window_acc = float(np.mean(window_pred == val_y))
    print(f"window-level val accuracy (MLP on binary outputs): {window_acc:.4f}",
          flush=True)
    clip_acc = _clip_accuracy(
        head_proba, SPEC["processed_dir"] / "window_index_val.csv"
    )
    print(f"clip-level val accuracy (clip_mean_softmax): {clip_acc:.4f}",
          flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    for class_id, path in enumerate(cnn_paths):
        target = run_dir / path.name
        shutil.copy2(path, target)
        print(f"  copied frozen binary CNN: {target}", flush=True)
    head_path = run_dir / "mlp_head.keras"
    head.save(str(head_path))
    print(f"  saved MLP head: {head_path}", flush=True)
    names_path = run_dir / "class_names.txt"
    names_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"  saved class names: {names_path}", flush=True)
    meta = {
        "run_name": run_name,
        "class_names": names,
        "cnn_source": str(source_run),
        "hidden": args.hidden,
        "epochs": args.epochs,
        "input_features": NUM_CLASSES,
        "train_windows": int(train_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "balanced_class_weights": bool(args.balanced_class_weights),
        "z_cache": {"dir": str(cache_dir), "fingerprint": fingerprint},
        "mlp": {
            "hidden": args.hidden,
            "dropout": config.DROPOUT_RATE,
            "optimizer": "adam",
            "learning_rate": config.LEARNING_RATE,
            "loss": "sparse_categorical_crossentropy",
            "window_val_accuracy": window_acc,
            "clip_val_accuracy": clip_acc,
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  saved metadata: {meta_path}", flush=True)
    print(
        f"experiment run complete in {meta['elapsed_seconds']:.1f} s: "
        f"17 frozen binary CNNs + MLP head -> {run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
