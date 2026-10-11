#!/usr/bin/env python
"""Phase 3A - train the lightweight CNN on the locked Phase 2 artifacts.

Reads the frozen Phase 2A/2B outputs in data/processed/ (features_<split>.npy,
labels_<split>.npy, cross-checked against feature_stats.json and the fixed
config parameters) and trains the edge-deployable CNN of blueprint section 7
Phase 3 over the 17-class taxonomy (src/config.TRIGGER_CLASS_NAMES). Nothing
in data/ is recomputed, rewritten, or deleted; the test split is not touched
(Phase 3B evaluation only).

Epochs: DEFAULT_EPOCHS = 10 (change the constant below or pass --epochs N).
Each epoch reports wall-clock training time, training loss/accuracy and
validation loss/accuracy.

Every run needs a unique --run-name. After training the full Keras model
(architecture + learned parameters) is saved to data/models/<run-name>.keras
(the directory is created if it does not exist); if that file already exists
the script aborts before doing any work, so existing models are never
overwritten. The per-epoch training history is written alongside it to
data/models/<run-name>.history.json (run name, epochs, parameter count,
metric series, per-epoch seconds, elapsed seconds); that file is never
overwritten either. Reload later for Phase 4 inference with
tf.keras.models.load_model("data/models/<run-name>.keras").

Safety guards: model.fit() runs only when --confirm-train is passed; without it
the script stops after the setup checks (data verification, model build,
parameter count) and saves nothing. --dry-run builds the model, loads one batch
of the training split, prints the input shape, every layer's output shape and
the parameter count, then exits without compiling, training or saving.

Training config follows the blueprint: Adam at config.LEARNING_RATE,
sparse categorical cross-entropy (integer labels), batch size
config.BATCH_SIZE, learning-rate reduction on plateau, early stopping on
validation loss. Class weighting is OFF by default; pass --class-weight to
compute balanced weights inside this script from the training labels only,
with the formula w_c = N / (C * n_c), where N is the number of training
windows, C = 17 classes and n_c is the number of training windows of class c.
The sample-weighted average of the weights over the training windows is
exactly 1.0; any class below the uniform share N / C is weighted above 1.0 in
inverse proportion to its training frequency. The weights are passed to
model.fit through class_weight. Validation and test labels are never read for
weighting; config.CLASS_WEIGHTS stays None and is not used.

Usage:
  python scripts/run_phase3a_training.py --run-name baseline10 --confirm-train
  python scripts/run_phase3a_training.py --run-name weighted10 --class-weight --confirm-train
  python scripts/run_phase3a_training.py --run-name baseline10 --confirm-train --epochs 20
  python scripts/run_phase3a_training.py --run-name dryrun10 --dry-run
  python scripts/run_phase3a_training.py --preset D --pool-freq-only --run-name presetD_freqpool --dry-run

Analysis presets (src.config.PRESETS, Phase 3A controlled experiments): with
--preset P the script reads P's processed directory (data/processed_preset_<P>
for B/C/D, the locked data/processed root for A) and P's feature_stats.json
instead of the defaults, and builds the input on P's spec_shape. Preset A is
the untouched Phase 2B baseline; default --preset A reproduces the previous
behavior exactly (same files, same architecture, 56,657 parameters).

Pooling mode: the default builds the locked head (two 2x2 pools preserving the
conv-16/32/64 ladder, then Flatten + Dense(32) + Dropout + Dense(17)), which
reaches 56,657 parameters on the (64, 5, 1) grid. --pool-freq-only is the
variant for the longer presets: both pools become (2, 1) (frequency collapses
to 16 bands, the time axis is preserved all the way) and the head is a
GlobalAveragePooling2D over the 64 feature maps instead of a flat dense, which
keeps the parameter count inside the 100,000 edge budget; on preset D this
yields 25,937 parameters. Default mode (no flag) always builds the original
flattened head.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from src import config

DEFAULT_EPOCHS = 10
MAX_PARAMETERS = 100_000
MODEL_DIR = config.DATA_DIR / "models"


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


def _preset_spec(preset: str) -> dict:
    """Return the preset's window/FFT/mel configuration dict or raise."""
    try:
        spec = config.PRESETS[preset]
    except KeyError:
        raise ValueError(f"unknown preset {preset!r}: expected one of {sorted(config.PRESETS)}")
    return spec


def _load_split(split: str, preset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split's feature array and integer labels for a preset."""
    processed_dir = _preset_spec(preset)["processed_dir"]
    features = np.load(processed_dir / f"features_{split}.npy")
    labels = np.load(processed_dir / f"labels_{split}.npy").astype(np.int32)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split}: {features.shape[0]} feature rows vs {labels.shape[0]} labels"
        )
    return features, labels


def _load_feature_stats(preset: str) -> dict:
    """Read the frozen preprocessing record for a preset."""
    with _preset_spec(preset)["feature_stats_path"].open(encoding="utf-8") as handle:
        return json.load(handle)


def _verify_split(
    stats: dict, split: str, features: np.ndarray, labels: np.ndarray, preset: str
) -> None:
    """Fail loudly if the arrays disagree with the locked preprocessing record."""
    spec = _preset_spec(preset)
    if list(features.shape[1:]) != list(spec["spec_shape"]):
        raise ValueError(
            f"{split}: feature shape {features.shape[1:]} != "
            f"preset {preset} {spec['spec_shape']}"
        )
    recorded = stats["splits"][split]
    if int(recorded["windows"]) != features.shape[0]:
        raise ValueError(
            f"{split}: {features.shape[0]} rows vs "
            f"{recorded['windows']} windows in feature_stats.json"
        )
    if int(recorded["clips"]) > features.shape[0]:
        raise ValueError(f"{split}: more recorded clips than windows")
    if int(labels.min()) < 0 or int(labels.max()) >= config.NUM_TRIGGER_CLASSES:
        raise ValueError(f"{split}: labels outside [0, {config.NUM_TRIGGER_CLASSES})")
    if features.dtype != np.float32:
        raise ValueError(f"{split}: dtype {features.dtype} != float32")
    if int(stats["sample_rate"]) != config.SAMPLE_RATE:
        raise ValueError("feature_stats.json sample_rate disagrees with config")
    if int(stats["n_mels"]) != int(spec["n_mels"]):
        raise ValueError(
            f"feature_stats.json n_mels disagrees with preset {preset}"
        )


def _compute_class_weight(labels: np.ndarray) -> dict[int, float]:
    """Balanced class weights from training labels only: w_c = N / (C * n_c).

    N is the number of training windows (labels.shape[0]), C the number of
    classes (config.NUM_TRIGGER_CLASSES) and n_c the number of training
    windows whose label is c. The sample-weighted average of these weights
    over the training windows is exactly 1.0; classes rarer than the uniform
    share N / C receive weights above 1.0, common classes below 1.0.
    Only the array passed in (the training split) is read here - validation
    and test labels never influence the weights.
    """
    class_count = config.NUM_TRIGGER_CLASSES
    counts = np.bincount(labels, minlength=class_count)
    missing = [i for i in range(class_count) if int(counts[i]) == 0]
    if missing:
        raise ValueError(f"training split has no windows for class ids {missing}")
    total = int(labels.shape[0])
    return {
        class_id: float(total / (class_count * int(counts[class_id])))
        for class_id in range(class_count)
    }


def build_cnn(spec_shape: tuple[int, int, int], pool_freq_only: bool) -> tf.keras.Model:
    """Build the lightweight CNN: conv 16/32/64 + head (flatten or GAP).

    The locked input grid for preset A is 64 mels x 5 frames, so only two 2x2
    pools fit (64x5 -> 32x2 -> 16x1); the third conv block runs unpooled
    before the head. Default head = Flatten + Dense(32) + Dropout + Dense(17),
    which is 56,657 parameters on (64, 5, 1). With pool_freq_only both pools
    are (2, 1), the time axis is never downsampled, and the head becomes a
    GlobalAveragePooling2D over the 64 feature maps so the parameter count
    stays inside MAX_PARAMETERS for the longer-window presets (25,937 on
    preset D). Total parameters never exceed MAX_PARAMETERS (blueprint section 7).
    """
    pool_size = (2, 1) if pool_freq_only else (2, 2)
    inputs = tf.keras.Input(shape=spec_shape, name="mel_window")
    x = tf.keras.layers.Conv2D(
        config.CONV1_FILTERS, 3, activation="relu", padding="same"
    )(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size)(x)
    x = tf.keras.layers.Conv2D(
        config.CONV2_FILTERS, 3, activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size)(x)
    x = tf.keras.layers.Conv2D(
        config.CONV3_FILTERS, 3, activation="relu", padding="same"
    )(x)
    if pool_freq_only:
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(
        config.NUM_TRIGGER_CLASSES, activation="softmax", name="class_probs"
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lightweight_cnn")


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3A CNN training on the locked Phase 2 artifacts"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="NAME",
        help="unique run name; the model is saved to data/models/NAME.keras "
        "(aborts if that file already exists)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"training epochs (default: {DEFAULT_EPOCHS} for the initial benchmark)",
    )
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="pass balanced class weights w_c = N / (C * n_c) computed from the "
        "training split to model.fit (default: off)",
    )
    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="required to run model.fit(); without it the script stops after "
        "the setup checks and saves nothing",
    )
    parser.add_argument(
        "--preset",
        default=config.PRESET_A,
        metavar="P",
        help="analysis preset A-D (default A = the locked data/processed/"
        "baseline; default behavior is identical to before this option)",
    )
    parser.add_argument(
        "--pool-freq-only",
        action="store_true",
        help="use (2, 1) pools and a GlobalAveragePooling2D head so the time "
        "axis survives pooling inside the 100,000-parameter edge budget "
        "(default off: the locked (2, 2) pools + flatten head)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the model, load one batch, print the input shape, every "
        "layer output shape and the parameter count, then exit without "
        "training or saving",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    try:
        run_name = _validate_run_name(args.run_name)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        spec = _preset_spec(args.preset)
    except ValueError as exc:
        parser.error(str(exc))
    model_path = MODEL_DIR / f"{run_name}.keras"
    if model_path.exists():
        raise SystemExit(
            f"run name already in use: {model_path} exists - pick a different "
            "--run-name; existing models are never overwritten"
        )

    started = time.perf_counter()
    tf.keras.utils.set_random_seed(config.SPLIT_SEED)

    print(f"tensorflow {tf.__version__}", flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"devices: {len(gpus)} GPU(s)", flush=True)
    print(
        f"run: {run_name}, preset: {args.preset}, "
        f"pool_freq_only: {'on' if args.pool_freq_only else 'off'}, "
        f"epochs: {args.epochs} (DEFAULT_EPOCHS={DEFAULT_EPOCHS}), "
        f"batch_size: {config.BATCH_SIZE}, lr: {config.LEARNING_RATE}, "
        f"class_weight: {'on' if args.class_weight else 'off'}, "
        f"confirm_train: {'yes' if args.confirm_train else 'no'}, "
        f"dry_run: {'yes' if args.dry_run else 'no'}",
        flush=True,
    )

    stats = _load_feature_stats(args.preset)
    train_x, train_y = _load_split("train", args.preset)
    val_x, val_y = _load_split("val", args.preset)
    _verify_split(stats, "train", train_x, train_y, args.preset)
    _verify_split(stats, "val", val_x, val_y, args.preset)
    print(
        f"data: preset {args.preset}, train {train_x.shape[0]} windows, "
        f"val {val_x.shape[0]} windows, input {spec['spec_shape']}, "
        f"classes {config.NUM_TRIGGER_CLASSES}",
        flush=True,
    )
    class_weights = None
    if args.class_weight:
        class_weights = _compute_class_weight(train_y)
        print(
            f"class weights (w_c = N / (C * n_c), train only; "
            f"N={train_x.shape[0]}, C={config.NUM_TRIGGER_CLASSES}):",
            flush=True,
        )
        for class_id in range(config.NUM_TRIGGER_CLASSES):
            print(
                f"  {class_id:>2} {config.TRIGGER_CLASS_NAMES[class_id]:<28} "
                f"{class_weights[class_id]:.6f}",
                flush=True,
            )
    else:
        print(
            "class weights: off (pass --class-weight to use w_c = N / (C * n_c))",
            flush=True,
        )

    model = build_cnn(tuple(spec["spec_shape"]), args.pool_freq_only)
    parameters = int(model.count_params())
    if parameters > MAX_PARAMETERS:
        raise ValueError(
            f"model has {parameters} parameters, above the {MAX_PARAMETERS} edge limit"
        )
    print(f"parameters: {parameters} (limit {MAX_PARAMETERS})", flush=True)

    if args.dry_run:
        batch = train_x[: min(config.BATCH_SIZE, int(train_x.shape[0]))]
        print(
            f"dry run: input shape {spec['spec_shape']}, dtype {train_x.dtype}",
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
        print("dry run complete: no compile, no fit, no model saved.", flush=True)
        return 0

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    if not args.confirm_train:
        print(
            "--confirm-train not given: setup checks passed, model.fit "
            "skipped, nothing saved.",
            flush=True,
        )
        return 0

    timer = EpochTimer()
    callbacks = [
        timer,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"training for {args.epochs} epoch(s)...", flush=True)
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    metrics = history.history
    _print_history(timer.epoch_seconds, metrics)
    last = len(metrics["loss"]) - 1
    print(
        f"final: train_loss {metrics['loss'][last]:.4f} "
        f"train_acc {metrics['accuracy'][last]:.4f} | "
        f"val_loss {metrics['val_loss'][last]:.4f} "
        f"val_acc {metrics['val_accuracy'][last]:.4f}",
        flush=True,
    )
    if last + 1 < args.epochs:
        print(
            f"early stopping ended training after {last + 1} of {args.epochs} epoch(s)",
            flush=True,
        )
    if model_path.exists():
        raise SystemExit(f"refusing to overwrite existing model: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(
        f"saved model: {model_path} ({parameters} parameters); "
        "reload with tf.keras.models.load_model(...)",
        flush=True,
    )
    history_path = MODEL_DIR / f"{run_name}.history.json"
    if history_path.exists():
        raise SystemExit(f"refusing to overwrite existing history: {history_path}")
    history_payload = {
        "run_name": run_name,
        "epochs": args.epochs,
        "parameters": parameters,
        "history": {
            key: [float(value) for value in values]
            for key, values in metrics.items()
        },
        "epoch_seconds": [float(value) for value in timer.epoch_seconds],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    print(f"saved training history: {history_path}", flush=True)
    print(f"total wall time: {time.perf_counter() - started:.1f} s", flush=True)
    print("Phase 3A training run complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
