#!/usr/bin/env python
"""Train the experimental YAMNet-embedding + MLP head pipeline.

Experimental architecture (isolated in experiment_yamnet_mlp/): frozen YAMNet
frame embeddings followed by a small tf.keras MLP classifier. It is the same
two-stage shape as experiment_binary_cnn_lr/ and experiment_binary_cnn_mlp/ (a
frozen feature extractor plus a lightweight trainable head) but the first stage
is the pretrained YAMNet audio-event model instead of the project's 17 binary
CNNs.

First stage (frozen, never fine-tuned): every clip is cleaned exactly like the
training extract (mono downmix, soxr_hq resample to 16 kHz, first 15 s), the
whole waveform is run through YAMNet once, and the 1024-d frame embeddings are
averaged over the clip into a single 1024-d per-clip vector. This is the same
per-clip mean embedding convention used by the project's final YAMNet +
logistic pipeline (scripts/run_phase3a_embedding_baseline.py, src/config.py
D13). YAMNet is loaded through src.inference.yamnet_logistic, the single source
of truth for the YAMNet preprocessing and embedding path, so the embeddings
here are identical to the production ones.

Second stage (trained here): an MLP head (Dense hidden -> relu -> Dropout ->
Dense 17 softmax) fitted on the standardized per-clip embeddings. A
StandardScaler is fit on the train embeddings only, exactly like the logistic
head fit in run_phase3a_embedding_baseline.py.

Data: kept clips from train + val only, resolved the same way as the embedding
baseline (data/provenance/phase2a_clip_ledger.csv rows with status == kept,
inner joined on clip_id with data/splits/clip_splits.csv rows whose split is
train or val, sorted by [split, clip_id]). The test split is never loaded and
is left for a later Phase 3B evaluation.

Embeddings are read from an existing extract directory (default
data/embeddings/yamnet/yamnet_extract/, the production per-clip mean
embeddings) when present, which is fast and reproducible. If that directory is
missing the script extracts the embeddings itself with frozen YAMNet and caches
them under --cache-dir (keyed by a small fingerprint) so later runs reload them
instantly. Nothing in data/ is written.

Every run needs a unique --run-name. Artifacts are written to
experiment_yamnet_mlp/runs/<run-name>/:
  mlp_head.keras    the trained MLP head (input 1024-d standardized embedding)
  scaler.joblib     the StandardScaler fit on the train embeddings
  class_names.txt   the 17 class names in id order
  meta.json         run metadata and validation metrics (paths stored
                    repo-relative, POSIX form, so the run is portable)
If that directory already exists the script aborts before doing any work, so
existing experiment runs are never overwritten. The experiment is fully
contained in experiment_yamnet_mlp/.

Safety guards: model.fit() runs only when --confirm-train is passed; without it
the script stops after the setup checks and saves nothing. --dry-run builds the
MLP head, prints every layer output shape and the parameter count, then exits
without compiling, training or saving.

Usage:
  .venv/Scripts/python experiment_yamnet_mlp/train_yamnet_mlp.py \
      --run-name mlp01 --confirm-train
  .venv/Scripts/python experiment_yamnet_mlp/train_yamnet_mlp.py \
      --run-name mlp01 --confirm-train --hidden 256 --epochs 60
  .venv/Scripts/python experiment_yamnet_mlp/train_yamnet_mlp.py \
      --run-name dryrun --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src import config

DEFAULT_EPOCHS = 50
DEFAULT_HIDDEN = 256
PREDICT_BATCH = 512
EMBED_DIM = config.YAMNET_EMBED_DIM
NUM_CLASSES = config.NUM_TRIGGER_CLASSES
YAMNET_URL = config.YAMNET_URL
RUNS_ROOT = Path(__file__).resolve().parent / "runs"
CACHE_ROOT = Path(__file__).resolve().parent / "cache"
EMBED_ROOT = ROOT / "data" / "embeddings" / "yamnet"
DEFAULT_EXTRACT_DIR = EMBED_ROOT / "yamnet_extract"
LEDGER_CSV = ROOT / "data" / "provenance" / "phase2a_clip_ledger.csv"
SPLIT_CSV = ROOT / "data" / "splits" / "clip_splits.csv"
CLASS_NAMES = list(config.TRIGGER_CLASS_NAMES)


def _portable(path: Path) -> str:
    """Return a repo-relative POSIX path, or the absolute path if outside ROOT."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


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


def _resolve_path(raw: str) -> Path:
    """Resolve a ledger local_path, preferring it as-is then repo-relative."""
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    repo_relative = ROOT / raw
    if repo_relative.is_file():
        return repo_relative
    raise SystemExit(f"audio file not found: {raw}")


def _read_expected_samples() -> pd.DataFrame:
    """Return kept train+val clips exactly as the embedding baseline orders them.

    Same join and same stable [split, clip_id] sort as
    scripts/run_phase3a_embedding_baseline.py, so the row order matches the
    shared extract (and any locally extracted embeddings) one to one.
    """
    ledger = pd.read_csv(LEDGER_CSV)
    kept = ledger.loc[ledger["status"] == "kept", [
        "clip_id", "class_id", "local_path", "source_dataset", "sample_rate"
    ]]
    splits = pd.read_csv(SPLIT_CSV)
    parts = splits.loc[splits["split"].isin(("train", "val")), ["clip_id", "split"]]
    merged = kept.merge(parts, on="clip_id", how="inner", validate="one_to_one")
    merged = merged.sort_values(
        ["split", "clip_id"], kind="stable"
    ).reset_index(drop=True)
    names = {i: n for i, n in enumerate(CLASS_NAMES)}
    merged["class_name"] = merged["class_id"].map(names)
    if set(merged["split"]) - {"train", "val"}:
        raise SystemExit("expected samples contain a non train/val split")
    return merged


def _validate_index(index: pd.DataFrame, embed_rows: int) -> None:
    """Fail loudly if an embedding index is malformed or touches the test split."""
    required = {"clip_id", "split", "class_id", "class_name"}
    missing = required - set(index.columns)
    if missing:
        raise SystemExit(f"embedding index missing columns: {sorted(missing)}")
    splits = set(index["split"].astype(str))
    if not splits <= {"train", "val"}:
        raise SystemExit(
            f"embedding index has splits outside train/val: {sorted(splits)}"
        )
    if len(index) != embed_rows:
        raise SystemExit(
            f"embedding rows {embed_rows} != index rows {len(index)}"
        )
    class_ids = index["class_id"].to_numpy()
    if class_ids.min() < 0 or class_ids.max() >= NUM_CLASSES:
        raise SystemExit(f"class_id outside [0, {NUM_CLASSES})")


def _load_shared_extract(
    extract_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Load an existing per-clip mean embedding extract, or None if absent."""
    npz_path = extract_dir / f"yamnet_embeddings_{extract_dir.name}.npz"
    csv_path = extract_dir / f"yamnet_embeddings_{extract_dir.name}.csv"
    if not (npz_path.is_file() and csv_path.is_file()):
        return None
    embed = np.asarray(np.load(npz_path)["embed_mean"], dtype=np.float32)
    index = pd.read_csv(csv_path)
    if embed.ndim != 2 or embed.shape[1] != EMBED_DIM:
        raise SystemExit(
            f"{npz_path} embed_mean shape {embed.shape} is not "
            f"(n, {EMBED_DIM})"
        )
    _validate_index(index, embed.shape[0])
    return embed, index


def _clip_ids_sha256(clip_ids: list[str]) -> str:
    """Stable digest of the ordered clip ids, used as a cache fingerprint."""
    joined = "\n".join(clip_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _embedding_fingerprint(samples: pd.DataFrame) -> dict:
    """Small descriptor that must match for a cached embedding matrix to be reused."""
    return {
        "model": YAMNET_URL,
        "embed_dim": EMBED_DIM,
        "pooling": "mean_over_frames",
        "rows": int(samples.shape[0]),
        "clip_ids_sha256": _clip_ids_sha256(
            samples["clip_id"].astype(str).tolist()
        ),
    }


def _load_cached_embeddings(
    cache_dir: Path, fingerprint: dict
) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Return cached (embed, index) when the fingerprint matches, else None."""
    meta_path = cache_dir / "embedding_fingerprint.json"
    embed_path = cache_dir / "embeddings.npy"
    index_path = cache_dir / "embedding_index.csv"
    if not (meta_path.is_file() and embed_path.is_file() and index_path.is_file()):
        return None
    try:
        recorded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if recorded != fingerprint:
        return None
    embed = np.asarray(np.load(embed_path), dtype=np.float32)
    index = pd.read_csv(index_path)
    if embed.ndim != 2 or embed.shape[1] != EMBED_DIM:
        return None
    if embed.shape[0] != fingerprint["rows"] or len(index) != embed.shape[0]:
        return None
    return embed, index


def _save_cached_embeddings(
    cache_dir: Path, fingerprint: dict, embed: np.ndarray, index: pd.DataFrame
) -> None:
    """Persist the embedding matrix plus the fingerprint that makes it reusable."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embed)
    index.to_csv(cache_dir / "embedding_index.csv", index=False)
    (cache_dir / "embedding_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )


def _extract_embeddings_locally(
    samples: pd.DataFrame, cache_dir: Path, hub_dir: str | None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed the kept train+val clips with frozen YAMNet, caching the result.

    Reuses src.inference.yamnet_logistic (load_yamnet_signature, load_audio,
    embed_clip), the production YAMNet preprocessing and embedding path, so the
    vectors match the shared extract and the final logistic pipeline.
    """
    fingerprint = _embedding_fingerprint(samples)
    cached = _load_cached_embeddings(cache_dir, fingerprint)
    if cached is not None:
        embed, index = cached
        print(
            f"loaded cached YAMNet embeddings from {cache_dir} "
            f"({tuple(embed.shape)})",
            flush=True,
        )
        return embed, index

    from src.inference import yamnet_logistic as yl

    print("extracting YAMNet embeddings (frozen model, per-clip mean) ...",
          flush=True)
    signature = yl.load_yamnet_signature(hub_dir)
    embeds: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    started = time.time()
    for i, row in enumerate(samples.itertuples(index=False), start=1):
        audio = yl.load_audio(_resolve_path(str(row.local_path)))
        embed = yl.embed_clip(signature, audio)[0]
        embeds.append(embed)
        rows.append({
            "clip_id": str(row.clip_id),
            "split": str(row.split),
            "class_id": int(row.class_id),
            "class_name": str(row.class_name),
            "sample_rate": int(row.sample_rate),
        })
        if i % 500 == 0 or i == len(samples):
            print(f"  embedded {i}/{len(samples)} clips "
                  f"({time.time() - started:.0f} s)", flush=True)
    embed = np.stack(embeds).astype(np.float32)
    index = pd.DataFrame(rows)
    _save_cached_embeddings(cache_dir, fingerprint, embed, index)
    print(f"cached YAMNet embeddings -> {cache_dir} (reused by later runs)",
          flush=True)
    return embed, index


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
    """Build the head: 1024-d embedding -> hidden relu -> 17 softmax."""
    inputs = tf.keras.Input(shape=(EMBED_DIM,), name="yamnet_embedding")
    x = tf.keras.layers.Dense(hidden, activation="relu", name="hidden")(inputs)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="class_proba"
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="yamnet_mlp_head")


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


def _summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy / macro / weighted precision-recall-F1 as floats."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred,
                                                    average="weighted",
                                                    zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted",
                                              zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
    }


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Integer confusion matrix of shape (NUM_CLASSES, NUM_CLASSES)."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[true_id, pred_id] += 1
    return matrix


def _per_class_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Per-class precision/recall/F1/support from the confusion matrix."""
    confusion = _confusion(y_true, y_pred)
    rows: list[dict] = []
    for c in range(NUM_CLASSES):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        support = int(confusion[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
        rows.append({
            "class": CLASS_NAMES[c],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="train the experimental frozen YAMNet embeddings + MLP head"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="NAME",
        help="unique run name; artifacts are saved to "
        "experiment_yamnet_mlp/runs/NAME/ (aborts if that directory already "
        "exists)",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(DEFAULT_EXTRACT_DIR),
        metavar="DIR",
        help="existing per-clip mean embedding extract to reuse "
        f"(default: {DEFAULT_EXTRACT_DIR}); if missing, embeddings are "
        "extracted locally with frozen YAMNet and cached under --cache-dir",
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
        "--seed",
        type=int,
        default=config.SPLIT_SEED,
        help=f"random seed for weight initialization and batch shuffle order "
             f"(split/dataset are read from the CSV and never re-randomized; "
             f"default: {config.SPLIT_SEED})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(CACHE_ROOT),
        metavar="DIR",
        help="directory for the locally extracted embedding cache (default: "
        "experiment_yamnet_mlp/cache)",
    )
    parser.add_argument(
        "--hub-dir",
        default=None,
        help="YAMNet SavedModel directory used only by local extraction "
             "(default: the offline copy from src.config, then the URL)",
    )
    parser.add_argument(
        "--balanced-class-weights",
        action="store_true",
        help="fit the MLP with balanced class weights; omitted by default so "
        "the head matches the unweighted logistic head of the final pipeline",
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
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (ROOT / cache_dir).resolve()
    extract_dir = Path(args.extract_dir)
    if not extract_dir.is_absolute():
        extract_dir = (ROOT / extract_dir).resolve()

    started = time.perf_counter()
    tf.keras.utils.set_random_seed(args.seed)

    print("experiment: frozen YAMNet embeddings + single MLP head", flush=True)
    print(f"tensorflow {tf.__version__}", flush=True)
    print(
        f"run: {run_name}, seed: {args.seed}, hidden: {args.hidden}, "
        f"epochs: {args.epochs}, "
        f"batch_size: {config.BATCH_SIZE}, lr: {config.LEARNING_RATE}, "
        f"balanced_weights: "
        f"{'yes' if args.balanced_class_weights else 'no'}, "
        f"confirm_train: {'yes' if args.confirm_train else 'no'}, "
        f"dry_run: {'yes' if args.dry_run else 'no'}",
        flush=True,
    )

    samples = _read_expected_samples()
    train_clips = int((samples["split"] == "train").sum())
    val_clips = int((samples["split"] == "val").sum())
    print(
        f"data: kept train+val clips {len(samples)} "
        f"(train {train_clips}, val {val_clips}), input {EMBED_DIM}-d YAMNet "
        f"mean embedding, classes {NUM_CLASSES}, test split never loaded",
        flush=True,
    )

    if args.dry_run:
        head = build_mlp(args.hidden)
        print(
            f"dry run: model input (N, {EMBED_DIM}) of per-clip mean YAMNet "
            "embeddings",
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

    loaded = _load_shared_extract(extract_dir)
    if loaded is not None:
        embed, index = loaded
        if len(index) != len(samples) or int((index["split"] == "train").sum()) \
                != train_clips:
            raise SystemExit(
                f"extract {extract_dir} has {len(index)} clips "
                f"(train {int((index['split'] == 'train').sum())}) but the "
                f"locked split expects {len(samples)} (train {train_clips}); "
                "regenerate the extract or pass a matching --extract-dir"
            )
        if index["clip_id"].astype(str).tolist() != \
                samples["clip_id"].astype(str).tolist():
            raise SystemExit(
                f"extract {extract_dir} clip order does not match the locked "
                "train+val clip order; regenerate the extract or pass a "
                "matching --extract-dir so embeddings align with labels"
            )
        source_kind = "shared_extract"
        print(
            f"reusing shared YAMNet extract from {extract_dir} "
            f"({tuple(embed.shape)})",
            flush=True,
        )
    else:
        print(
            f"no shared extract at {extract_dir}; extracting locally",
            flush=True,
        )
        embed, index = _extract_embeddings_locally(samples, cache_dir,
                                                   args.hub_dir)
        if len(index) != len(samples):
            raise SystemExit(
                f"locally extracted {len(index)} clips but expected "
                f"{len(samples)}"
            )
        source_kind = "local_extract"

    y = index["class_id"].to_numpy(dtype=np.int64)
    is_train = (index["split"] == "train").to_numpy(bool)
    is_val = ~is_train
    x_all = embed.astype(np.float64)
    scaler = StandardScaler().fit(x_all[is_train])
    x_train = scaler.transform(x_all[is_train]).astype(np.float32)
    x_val = scaler.transform(x_all[is_val]).astype(np.float32)
    y_train = y[is_train]
    y_val = y[is_val]
    print(
        f"standardized (StandardScaler fit on train): X_train "
        f"{tuple(x_train.shape)}, X_val {tuple(x_val.shape)}",
        flush=True,
    )

    head = build_mlp(args.hidden)
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    weights = _balanced_weights(y_train) if args.balanced_class_weights else None
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
    print("fitting the MLP head on standardized YAMNet embeddings ...",
          flush=True)
    history = head.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    _print_history(timer.epoch_seconds, history.history)

    val_proba = head.predict(x_val, batch_size=PREDICT_BATCH, verbose=0)
    val_pred = np.argmax(val_proba, axis=1)
    metrics = _summary(y_val, val_pred)
    per_class = _per_class_rows(y_val, val_pred)
    confusion = _confusion(y_val, val_pred)
    print(
        f"clip-level val accuracy (MLP on YAMNet embeddings): "
        f"{metrics['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"clip-level val macro F1: {metrics['macro_f1']:.4f}, weighted F1: "
        f"{metrics['weighted_f1']:.4f}",
        flush=True,
    )
    print("", flush=True)
    print(
        f"{'class':<16}{'precision':>10}{'recall':>9}{'f1':>9}{'support':>9}",
        flush=True,
    )
    for row in per_class:
        print(
            f"{row['class']:<16}{row['precision']:>10.4f}"
            f"{row['recall']:>9.4f}{row['f1']:>9.4f}{row['support']:>9d}",
            flush=True,
        )
    print("", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    head_path = run_dir / "mlp_head.keras"
    head.save(str(head_path))
    print(f"  saved MLP head: {head_path}", flush=True)
    scaler_path = run_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  saved scaler: {scaler_path}", flush=True)
    names_path = run_dir / "class_names.txt"
    names_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    print(f"  saved class names: {names_path}", flush=True)
    meta = {
        "run_name": run_name,
        "seed": args.seed,
        "class_names": CLASS_NAMES,
        "embedding_source": {
            "kind": source_kind,
            "dir": _portable(extract_dir if source_kind == "shared_extract"
                             else cache_dir),
            "model": "yamnet",
            "url": YAMNET_URL,
            "dim": EMBED_DIM,
            "pooling": "mean_over_frames",
        },
        "hidden": args.hidden,
        "epochs": args.epochs,
        "input_features": EMBED_DIM,
        "train_clips": train_clips,
        "val_clips": val_clips,
        "balanced_class_weights": bool(args.balanced_class_weights),
        "scaler": "StandardScaler fit on train per-clip mean YAMNet embeddings",
        "mlp": {
            "hidden": args.hidden,
            "dropout": config.DROPOUT_RATE,
            "optimizer": "adam",
            "learning_rate": config.LEARNING_RATE,
            "loss": "sparse_categorical_crossentropy",
            "clip_val_accuracy": metrics["accuracy"],
            "clip_val_macro_f1": metrics["macro_f1"],
            "clip_val_weighted_f1": metrics["weighted_f1"],
            "summary": {key: round(value, 4) for key, value in metrics.items()},
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
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
        f"YAMNet embeddings + MLP head -> {run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
