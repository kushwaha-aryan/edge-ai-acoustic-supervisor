#!/usr/bin/env python
"""Train the experimental per-class binary CNN + logistic head pipeline.

Experimental architecture (isolated in experiment_binary_cnn_lr/): instead of
one multiclass CNN, train 17 independent binary CNNs, one per trigger class,
each deciding whether its class is present in a window. Their 17 sigmoid
outputs form a 17-dim feature vector that a single sklearn LogisticRegression
consumes to assign probabilities over all 17 classes; the class with the
highest logistic probability is the final prediction. This file trains that
pipeline on the locked Phase 2 (preset A) artifacts in data/processed/
(features_<split>.npy, labels_<split>.npy), cross-checked against
feature_stats.json with the same checks as scripts/run_phase3a_training.py.
The test split is never touched (Phase 3B evaluation only).

Every binary CNN uses the locked lightweight conv ladder (conv 16/32/64 with
two 2x2 pools) but ends in a single sigmoid unit. Each CNN is trained on all
train windows with the binary target y = (class_id == c) and balanced class
weights w0 = N / (2 * n0), w1 = N / (2 * n1) derived from the train split
only, so the rarer positive class is not ignored. Training follows Phase 3A:
Adam at config.LEARNING_RATE, binary cross-entropy, batch config.BATCH_SIZE,
ReduceLROnPlateau and EarlyStopping on val loss.

After the 17 CNNs are trained their train and val window outputs are batched
through predict() to build Z_train and Z_val (N x 17), and a single
LogisticRegression (lbfgs, max_iter 1000, seed config.SPLIT_SEED) is fitted
on Z_train. Validation is reported at window level and at clip level with the
project's clip_mean_softmax aggregation (mean LR probability per clip, then
argmax), grouped through window_index_val.csv. Nothing in data/ is rewritten
and no existing file is modified; this experiment is fully contained in
experiment_binary_cnn_lr/ and its runs/ subfolder.

Every run needs a unique --run-name. Artifacts are written to
experiment_binary_cnn_lr/runs/<run-name>/:
  class_<id>_<class_name>.keras   one binary CNN per class
  lr_classifier.joblib            the single logistic head
  class_names.txt                 the 17 class names in id order
  meta.json                       run metadata and validation metrics
If that directory already exists the script aborts before doing any work, so
existing experiment runs are never overwritten.

Safety guards: model.fit() runs only when --confirm-train is passed; without
it the script stops after the setup checks and saves nothing. --dry-run builds
one binary CNN, prints layer output shapes and the parameter count, then exits
without compiling, training or saving.

Usage:
  .venv/Scripts/python experiment_binary_cnn_lr/train_binary_cnn_lr.py \
      --run-name exp01 --confirm-train
  .venv/Scripts/python experiment_binary_cnn_lr/train_binary_cnn_lr.py \
      --run-name exp01 --confirm-train --epochs 12
  .venv/Scripts/python experiment_binary_cnn_lr/train_binary_cnn_lr.py \
      --run-name dryrun --dry-run
"""

from __future__ import annotations

import argparse
import joblib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from src import config

DEFAULT_EPOCHS = 8
PREDICT_BATCH = 512
RUNS_ROOT = Path(__file__).resolve().parent / "runs"
SPEC = config.PRESETS[config.PRESET_A]
OUTPUT_DENSE = config.DENSE_UNITS


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
    if int(labels.min()) < 0 or int(labels.max()) >= config.NUM_TRIGGER_CLASSES:
        raise ValueError(f"{split}: labels outside [0, {config.NUM_TRIGGER_CLASSES})")
    if features.dtype != np.float32:
        raise ValueError(f"{split}: dtype {features.dtype} != float32")
    if int(stats["sample_rate"]) != config.SAMPLE_RATE:
        raise ValueError("feature_stats.json sample_rate disagrees with config")
    if int(stats["n_mels"]) != int(SPEC["n_mels"]):
        raise ValueError(
            f"feature_stats.json n_mels disagrees with preset {config.PRESET_A}"
        )


def _balanced_weights(binary_labels: np.ndarray) -> dict[int, float]:
    """Balanced weights for a binary target: w = N / (2 * n) per label."""
    counts = np.bincount(binary_labels, minlength=2)
    total = int(binary_labels.shape[0])
    return {
        0: float(total / (2 * int(counts[0]))),
        1: float(total / (2 * int(counts[1]))),
    }


def build_binary_cnn() -> tf.keras.Model:
    """Build one binary CNN: locked conv 16/32/64 ladder + single sigmoid."""
    inputs = tf.keras.Input(shape=tuple(SPEC["spec_shape"]), name="mel_window")
    x = tf.keras.layers.Conv2D(
        config.CONV1_FILTERS, 3, activation="relu", padding="same"
    )(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        config.CONV2_FILTERS, 3, activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(
        config.CONV3_FILTERS, 3, activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(OUTPUT_DENSE, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="binary_proba")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="binary_cnn")


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
        description="train the experimental 17 binary CNNs + logistic head"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="NAME",
        help="unique run name; artifacts are saved to "
        "experiment_binary_cnn_lr/runs/NAME/ (aborts if that directory "
        "already exists)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"training epochs per binary CNN (default: {DEFAULT_EPOCHS})",
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
        help="build one binary CNN, print every layer output shape and the "
        "parameter count, then exit without compiling, training or saving",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
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

    started = time.perf_counter()
    tf.keras.utils.set_random_seed(config.SPLIT_SEED)

    print(f"experiment: 17 binary CNNs + single logistic regression", flush=True)
    print(f"tensorflow {tf.__version__}", flush=True)
    print(
        f"run: {run_name}, epochs: {args.epochs}, "
        f"batch_size: {config.BATCH_SIZE}, lr: {config.LEARNING_RATE}, "
        f"confirm_train: {'yes' if args.confirm_train else 'no'}, "
        f"dry_run: {'yes' if args.dry_run else 'no'}",
        flush=True,
    )

    stats = _load_feature_stats()
    train_x, train_y = _load_split("train")
    val_x, val_y = _load_split("val")
    _verify_split(stats, "train", train_x, train_y)
    _verify_split(stats, "val", val_x, val_y)
    print(
        f"data: preset {config.PRESET_A}, train {train_x.shape[0]} windows, "
        f"val {val_x.shape[0]} windows, input {SPEC['spec_shape']}, "
        f"classes {config.NUM_TRIGGER_CLASSES}",
        flush=True,
    )

    if args.dry_run:
        model = build_binary_cnn()
        batch = train_x[: min(config.BATCH_SIZE, int(train_x.shape[0]))]
        print(
            f"dry run: input shape {SPEC['spec_shape']}, dtype {train_x.dtype}",
            flush=True,
        )
        print(f"batch: {tuple(batch.shape)} taken from the train split", flush=True)
        print("layer output shapes:", flush=True)
        for layer in model.layers:
            print(
                f"  {layer.name:<22} {layer.__class__.__name__:<20} "
                f"{tuple(layer.output.shape)}",
                flush=True,
            )
        print(
            f"parameters per binary CNN: {int(model.count_params()):,} "
            f"({config.NUM_TRIGGER_CLASSES} CNNs + LR total)",
            flush=True,
        )
        print("dry run complete: no compile, no fit, no model saved.", flush=True)
        return 0

    if not args.confirm_train:
        print(
            "--confirm-train not given: setup checks passed, no binary CNN is "
            "trained and nothing is saved.",
            flush=True,
        )
        return 0

    parameters = int(build_binary_cnn().count_params())
    print(f"parameters per binary CNN: {parameters:,}", flush=True)

    class_names = list(config.TRIGGER_CLASS_NAMES)
    val_acc_per_class: dict[str, float] = {}
    binary_models: list[tf.keras.Model] = []
    for class_id, class_name in enumerate(class_names):
        print(f"training binary CNN {class_id + 1}/{len(class_names)} "
              f"for {class_name} ...", flush=True)
        binary_y = (train_y == class_id).astype(np.int32)
        binary_val_y = (val_y == class_id).astype(np.int32)
        weights = _balanced_weights(binary_y)
        positives = int(binary_y.sum())
        print(
            f"  target {class_name} (class_id {class_id}): positives "
            f"{positives}/{len(binary_y)} of train windows "
            f"(w0={weights[0]:.4f}, w1={weights[1]:.4f})",
            flush=True,
        )
        timer = EpochTimer()
        callbacks = [
            timer,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5,
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            ),
        ]
        binary_model = build_binary_cnn()
        binary_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        history = binary_model.fit(
            train_x,
            binary_y,
            validation_data=(val_x, binary_val_y),
            epochs=args.epochs,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            class_weight=weights,
            callbacks=callbacks,
            verbose=2,
        )
        _print_history(timer.epoch_seconds, history.history)
        last = len(history.history["val_accuracy"]) - 1
        val_acc_per_class[class_name] = float(history.history["val_accuracy"][last])
        print(
            f"  {class_name} final: train_acc "
            f"{history.history['accuracy'][last]:.4f} | "
            f"val_acc {history.history['val_accuracy'][last]:.4f}",
            flush=True,
        )
        binary_models.append(binary_model)

    print("building Z_train / Z_val from the 17 binary CNNs ...", flush=True)
    z_train_chunks: list[np.ndarray] = []
    z_val_chunks: list[np.ndarray] = []
    for class_id, class_name in enumerate(class_names):
        binary_model = binary_models[class_id]
        z_train_chunks.append(
            binary_model.predict(train_x, batch_size=PREDICT_BATCH, verbose=0)[:, 0]
        )
        z_val_chunks.append(
            binary_model.predict(val_x, batch_size=PREDICT_BATCH, verbose=0)[:, 0]
        )
    z_train = np.column_stack(z_train_chunks).astype(np.float32)
    z_val = np.column_stack(z_val_chunks).astype(np.float32)
    print(
        f"Z_train {tuple(z_train.shape)}, Z_val {tuple(z_val.shape)} "
        "(each window -> 17 binary-CNN outputs)",
        flush=True,
    )

    print("fitting the single logistic head on Z_train ...", flush=True)
    logistic = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=config.SPLIT_SEED
    )
    logistic.fit(z_train, train_y)
    window_pred = logistic.predict(z_val)
    window_acc = float(np.mean(window_pred == val_y))
    print(f"window-level val accuracy (LR on binary outputs): {window_acc:.4f}", flush=True)
    clip_acc = _clip_accuracy(
        logistic.predict_proba(z_val), SPEC["processed_dir"] / "window_index_val.csv"
    )
    print(f"clip-level val accuracy (clip_mean_softmax): {clip_acc:.4f}", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    for class_id, class_name in enumerate(class_names):
        path = run_dir / f"class_{class_id:02d}_{class_name}.keras"
        binary_models[class_id].save(str(path))
        print(f"  saved binary CNN: {path}", flush=True)
    logistic_path = run_dir / "lr_classifier.joblib"
    joblib.dump(logistic, logistic_path)
    print(f"  saved logistic head: {logistic_path}", flush=True)
    names_path = run_dir / "class_names.txt"
    names_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"  saved class names: {names_path}", flush=True)
    meta = {
        "run_name": run_name,
        "class_names": class_names,
        "epochs": args.epochs,
        "parameters_per_cnn": parameters,
        "train_windows": int(train_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "per_class_val_accuracy": val_acc_per_class,
        "logistic": {
            "solver": "lbfgs",
            "max_iter": 1000,
            "window_val_accuracy": window_acc,
            "clip_val_accuracy": clip_acc,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  saved metadata: {meta_path}", flush=True)
    print(
        f"experiment run complete in {meta['elapsed_seconds']:.1f} s: "
        f"17 binary CNNs + logistic head -> {run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())