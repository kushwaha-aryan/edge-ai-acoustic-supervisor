#!/usr/bin/env python
"""Run the experimental YAMNet-embedding + MLP pipeline on a random sample of
VALIDATION clips.

Isolated experiment (experiment_yamnet_mlp/): frozen YAMNet frame embeddings
followed by a small tf.keras MLP head, trained by train_yamnet_mlp.py. It drives
that trained pipeline: a clip is cleaned exactly like the training extract
(mono downmix, soxr_hq resample to 16 kHz, first 15 s), run through the frozen
YAMNet SavedModel once, the 1024-d frame embeddings are averaged over the clip
into one 1024-d vector (the same per-clip mean convention as the final YAMNet +
logistic pipeline), the vector is standardized with the run's saved
StandardScaler, and the saved MLP head returns a 17-class softmax. The clip
prediction is the argmax of that softmax and the confidence is the top
probability.

The YAMNet preprocessing and embedding helpers are imported from
src.inference.yamnet_logistic (the project's single source of truth for the
YAMNet path), so the embeddings match the training extract.

Draws NUM_CASES clips at random (without replacement) from the whole validation
split only: data/splits/clip_splits.csv filtered to split == val, joined on
clip_id with data/provenance/phase2a_clip_ledger.csv filtered to status == kept.
Output is an ASCII table plus a summary block and is printed to stdout only;
nothing is written, so this script never overwrites any file.

Editable module constants (change them here, or override per run via --n and
--seed):
    NUM_CASES = 20      number of clips to sample per run
    SEED = None         repeat seed for a repeatable sample; None picks a
                        random seed every run and prints it

Usage:
  .venv/Scripts/python experiment_yamnet_mlp/demo_yamnet_mlp.py
  .venv/Scripts/python experiment_yamnet_mlp/demo_yamnet_mlp.py --n 50
  .venv/Scripts/python experiment_yamnet_mlp/demo_yamnet_mlp.py --seed 12345
  .venv/Scripts/python experiment_yamnet_mlp/demo_yamnet_mlp.py \
      --run-dir experiment_yamnet_mlp/runs/MY_RUN

The sample is small, so its accuracy varies from run to run. This demo makes
no claim about full-validation accuracy; see the run's meta.json for the
pipeline's clip-level validation accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.inference import yamnet_logistic as yl

SPLIT_CSV = ROOT / "data" / "splits" / "clip_splits.csv"
LEDGER_CSV = ROOT / "data" / "provenance" / "phase2a_clip_ledger.csv"
VALIDATION_SPLIT = "val"
KEPT_STATUS = "kept"
EXPECTED_POOL_SIZE = 1618
DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent / "runs"

NUM_CASES = 720
SEED = None

TABLE_HEADERS = ["#", "source_dataset", "file name", "ACTUAL class",
                 "PREDICTED class", "confidence", "duration", "OK"]
CLASS_HEADERS = ["class", "in pool", "shown", "correct", "wrong",
                 "precision", "recall", "F1", "errors"]


def _render_cells(cells: list[str], widths: list[int],
                  right: list[bool]) -> str:
    """Render one ASCII table row from cells and per-column widths."""
    parts = []
    for col, cell in enumerate(cells):
        if right[col]:
            parts.append(cell.rjust(widths[col]))
        else:
            parts.append(cell.ljust(widths[col]))
    return "  ".join(parts)


def _ledger_map(ledger_path: Path) -> dict[str, dict[str, str]]:
    """Return clip_id -> ledger row for every clip whose status is kept."""
    kept: dict[str, dict[str, str]] = {}
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == KEPT_STATUS:
                clip_id = row["clip_id"]
                if clip_id in kept:
                    raise SystemExit(
                        f"ledger has duplicate kept clip_id: {clip_id}")
                kept[clip_id] = row
    return kept


def _validation_pool(split_path: Path,
                     kept: dict[str, dict[str, str]]
                     ) -> tuple[list[dict[str, str]], list[str]]:
    """Return (pool, unmatched) for the one-to-one val<->kept join.

    The pool holds the ledger rows for every validation clip whose clip_id
    also exists as a kept ledger row (inner join on clip_id, not on any path
    string). The second element holds the clip_ids of validation clips that
    had no kept ledger match.
    """
    pool: list[dict[str, str]] = []
    unmatched: list[str] = []
    seen: set[str] = set()
    with split_path.open(encoding="utf-8", newline="") as handle:
        for split_row in csv.DictReader(handle):
            if split_row.get("split") != VALIDATION_SPLIT:
                continue
            clip_id = split_row["clip_id"]
            if clip_id in seen:
                raise SystemExit(
                    f"split file has duplicate val clip_id: {clip_id}")
            seen.add(clip_id)
            ledger_row = kept.get(clip_id)
            if ledger_row is not None:
                pool.append(ledger_row)
            else:
                unmatched.append(clip_id)
    return pool, unmatched


def _duration_str(ledger_row: dict[str, str]) -> str:
    """Render the ledger duration column of a clip."""
    for key in ("processed_duration_s", "duration_s"):
        value = ledger_row.get(key, "")
        if value:
            try:
                return f"{float(value):.2f}"
            except ValueError:
                return value
    return "-"


def _print_table(rows: list[list[str]]) -> None:
    """Print the results as a fixed-width ASCII table."""
    widths = [len(header) for header in TABLE_HEADERS]
    for row in rows:
        for col, cell in enumerate(row):
            widths[col] = max(widths[col], len(cell))

    print(_render_cells(TABLE_HEADERS, widths,
                        [False] * len(TABLE_HEADERS)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(_render_cells(row, widths,
                            [True, False, False, False, False, True, True,
                             False]))


def _prec_recall_f1(correct: int, wrong: int, fp: int
                    ) -> tuple[float | None, float | None, float | None]:
    """Return (precision, recall, F1) for one class, None where undefined.

    Correct = true positives, wrong = false negatives, fp = false positives.
    """
    precision = correct / (correct + fp) if correct + fp > 0 else None
    recall = correct / (correct + wrong) if correct + wrong > 0 else None
    f1 = None
    if precision is not None and recall is not None:
        denom = precision + recall
        if denom > 0:
            f1 = 2 * precision * recall / denom
    return precision, recall, f1


def _fmt_rate(value: float | None) -> str:
    """Render a rate as a percentage, or a dash when it is undefined."""
    return "-" if value is None else f"{value:.1%}"


def _print_class_table(per_class: dict[str, dict[str, int]],
                       pool_counts: dict[str, int],
                       total_cases: int,
                       total_correct: int,
                       total_errors: int) -> None:
    """Print the per-class summary block in class_names.txt order."""
    rows: list[list[str]] = []
    macro_precision: list[float] = []
    macro_recall: list[float] = []
    macro_f1: list[float] = []
    for class_name in per_class:
        counts = per_class[class_name]
        precision, recall, f1 = _prec_recall_f1(
            counts["correct"], counts["wrong"], counts["fp"])
        for bucket, value in ((macro_precision, precision),
                              (macro_recall, recall), (macro_f1, f1)):
            if value is not None:
                bucket.append(value)
        rows.append([
            class_name,
            str(pool_counts.get(class_name, 0)),
            str(counts["shown"]),
            str(counts["correct"]),
            str(counts["wrong"]),
            _fmt_rate(precision),
            _fmt_rate(recall),
            _fmt_rate(f1),
            str(counts["errors"]),
        ])
    total_shown = sum(counts["shown"] for counts in per_class.values())
    sum_errors = sum(counts["errors"] for counts in per_class.values())
    sum_wrong = total_cases - total_correct - total_errors
    precision = sum(macro_precision) / len(macro_precision) \
        if macro_precision else None
    recall = sum(macro_recall) / len(macro_recall) if macro_recall else None
    f1 = sum(macro_f1) / len(macro_f1) if macro_f1 else None
    total_row = [
        "TOTAL",
        str(sum(pool_counts.values())),
        str(total_shown),
        str(total_correct),
        str(sum_wrong),
        _fmt_rate(precision),
        _fmt_rate(recall),
        _fmt_rate(f1),
        str(sum_errors),
    ]
    rows.append(total_row)

    widths = [len(header) for header in CLASS_HEADERS]
    for row in rows:
        for col, cell in enumerate(row):
            widths[col] = max(widths[col], len(cell))

    print(_render_cells(CLASS_HEADERS, widths,
                        [False] * len(CLASS_HEADERS)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(_render_cells(row, widths,
                            [False, True, True, True, True,
                             True, True, True, True]))


def _resolve_run_dir(explicit: str | None, root: Path) -> Path:
    """Return the run directory holding the saved scaler and MLP head."""
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        if not (candidate / "mlp_head.keras").is_file():
            raise SystemExit(
                f"no saved pipeline found in {candidate} (missing "
                "mlp_head.keras) - run train_yamnet_mlp.py with "
                "--confirm-train first"
            )
        return candidate
    if (root / "mlp_head.keras").is_file():
        return root
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if len(runs) == 1:
        return runs[0]
    raise SystemExit(
        f"cannot pick a saved pipeline under {root} (expected exactly one "
        f"run directory, found {len(runs)}); pass --run-dir explicitly"
    )


def _load_pipeline(run_dir: Path) -> tuple[object, object, list[str]]:
    """Load the saved scaler, MLP head and class names."""
    names = [
        line.strip()
        for line in (run_dir / "class_names.txt").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise SystemExit(f"no class names found in {run_dir / 'class_names.txt'}")
    if tuple(names) != config.TRIGGER_CLASS_NAMES:
        raise SystemExit(
            "class_names.txt does not match the project trigger taxonomy in "
            "src/config.py TRIGGER_CLASS_NAMES; refusing to label predictions "
            "with a mismatched class list"
        )
    scaler = joblib.load(run_dir / "scaler.joblib")
    head = tf.keras.models.load_model(str(run_dir / "mlp_head.keras"))
    return scaler, head, names


def _resolve_clip_path(local_path: str) -> Path:
    """Resolve a ledger local_path, preferring it as-is then repo-relative."""
    candidate = Path(local_path)
    if candidate.is_file():
        return candidate
    repo_relative = ROOT / local_path
    if repo_relative.is_file():
        return repo_relative
    raise SystemExit(f"audio file not found: {local_path}")


def _classify(signature: object, scaler: object, head: object,
              audio_path: Path) -> tuple[int, np.ndarray]:
    """Clip-level probabilities: mean YAMNet embedding, standardized, MLP head.

    The clip is cleaned and embedded through src.inference.yamnet_logistic (the
    exact training/preprocessing path), the per-clip mean embedding is
    standardized with the run's saved scaler, and the saved MLP head returns a
    17-class softmax. The prediction is the argmax and the confidence is that
    top probability.
    """
    audio = yl.load_audio(audio_path)
    embed = yl.embed_clip(signature, audio)
    scaled = scaler.transform(embed).astype(np.float32)
    proba = head.predict(scaled, batch_size=1, verbose=0)[0]
    order = int(np.argmax(proba))
    return order, np.asarray(proba, dtype=np.float64)


def _read_clip_accuracy(run_dir: Path) -> float | None:
    """Return the run's saved clip-level validation accuracy if available."""
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return float(meta["mlp"]["clip_val_accuracy"])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    """Sample validation clips, classify them and print the ASCII table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n", type=int, default=None,
        help=f"number of clips to sample (overrides the NUM_CASES constant, "
             f"default {NUM_CASES})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for a repeatable sample (overrides the SEED "
             "constant)",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="directory with the saved scaler + MLP head (default: the single "
             "run under experiment_yamnet_mlp/runs)",
    )
    parser.add_argument(
        "--hub-dir", default=None,
        help="YAMNet SavedModel directory (default: the offline copy from "
             "src.config, then the online TF-Hub URL)",
    )
    parser.add_argument(
        "--allow-pool-mismatch", action="store_true",
        help="proceed even when the validation pool size differs from the "
             "expected value",
    )
    args = parser.parse_args()

    if args.n is None:
        num_cases = NUM_CASES
        print(f"num_cases: {num_cases} (from constant NUM_CASES)")
    else:
        num_cases = args.n
        print(f"num_cases: {num_cases} (from --n)")

    if args.seed is not None:
        seed = args.seed
        print(f"seed: {seed} (from --seed)")
    elif SEED is not None:
        seed = SEED
        print(f"seed: {seed} (from constant SEED)")
    else:
        seed = secrets.randbelow(2**31 - 1) + 1
        print(f"no seed given; drawn seed for this run: {seed}")

    kept = _ledger_map(LEDGER_CSV)
    pool, unmatched = _validation_pool(SPLIT_CSV, kept)
    print(f"validation split: {len(pool) + len(unmatched)} val clips in "
          f"clip_splits.csv")
    print(f"val clips with no kept ledger match: {len(unmatched)}")
    for clip_id in unmatched[:5]:
        print(f"  no ledger match: {clip_id}")
    pool_classes: dict[str, int] = {}
    for ledger_row in pool:
        project_class = ledger_row.get("project_class", "-")
        pool_classes[project_class] = pool_classes.get(project_class, 0) + 1
    print(f"validation pool: {len(pool)} clips (joined on clip_id, "
          f"status={KEPT_STATUS})")
    for project_class in sorted(pool_classes):
        print(f"  pool {project_class}: {pool_classes[project_class]}")
    if len(pool) != EXPECTED_POOL_SIZE:
        print("WARNING: validation pool size "
              f"({len(pool)}) != expected ({EXPECTED_POOL_SIZE})")
        if not args.allow_pool_mismatch:
            raise SystemExit(
                "validation pool size mismatch; rerun with "
                "--allow-pool-mismatch to override"
            )
    if num_cases < 1:
        raise SystemExit(f"num_cases must be at least 1, got {num_cases}")
    if num_cases > len(pool):
        raise SystemExit(
            f"num_cases {num_cases} exceeds the validation pool size "
            f"{len(pool)}; pick a smaller value"
        )
    print(f"sampling {num_cases} clips from the validation pool "
          f"({len(pool)} clips, split={VALIDATION_SPLIT}, "
          f"status={KEPT_STATUS})")

    rng = random.Random(seed)
    sample = rng.sample(pool, num_cases)

    run_dir = _resolve_run_dir(args.run_dir, DEFAULT_RUNS_ROOT)
    scaler, head, names = _load_pipeline(run_dir)
    signature = yl.load_yamnet_signature(args.hub_dir)
    print(f"saved experimental pipeline: {run_dir}")
    print(f"scaler: {scaler.__class__.__name__}, MLP head: "
          f"{head.__class__.__name__}")
    clip_accuracy = _read_clip_accuracy(run_dir)

    rows: list[list[str]] = []
    correct = 0
    errors: list[tuple[str, str]] = []
    per_class: dict[str, dict[str, int]] = {}
    for class_name in names:
        per_class[class_name] = {
            "shown": 0,
            "correct": 0,
            "wrong": 0,
            "fp": 0,
            "errors": 0,
        }

    for index, ledger_row in enumerate(sample, start=1):
        clip_id = ledger_row["clip_id"]
        source_dataset = ledger_row.get("source_dataset", "-")
        file_name = ledger_row.get("source_file", Path(clip_id).name)
        actual = ledger_row.get("project_class", "-")
        class_counts = per_class.get(actual)
        if class_counts is not None:
            class_counts["shown"] += 1
        try:
            audio_path = _resolve_clip_path(
                ledger_row.get("local_path", clip_id))
            order, proba = _classify(signature, scaler, head, audio_path)
            predicted = names[order]
            confidence = f"{proba[order]:.2%}"
            duration = _duration_str(ledger_row)
        except Exception as exc:
            predicted = "ERROR"
            confidence = "-"
            duration = _duration_str(ledger_row)
            reason = f"{type(exc).__name__}: {exc}"
            errors.append((file_name, reason))
            ok = "N"
            if class_counts is not None:
                class_counts["errors"] += 1
        else:
            ok = "Y" if predicted == actual else "N"
            if ok == "Y":
                correct += 1
            if class_counts is not None:
                if ok == "Y":
                    class_counts["correct"] += 1
                else:
                    class_counts["wrong"] += 1
            if ok == "N":
                predicted_counts = per_class.get(predicted)
                if predicted_counts is not None:
                    predicted_counts["fp"] += 1

        rows.append([str(index), source_dataset, file_name, actual,
                     predicted, confidence, str(duration), ok])

    _print_table(rows)

    for file_name, reason in errors:
        print(f"error: {file_name} - {reason}")

    total = len(rows)
    error_count = len(errors)
    done = total - error_count
    wrong = done - correct
    print(f"Cases tested: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Errors (could not classify): {error_count}")
    if total > 0:
        print(f"Accuracy on this sample: {correct}/{total} "
              f"({correct / total:.1%})")
    else:
        print("Accuracy on this sample: 0/0 (n/a)")
    print(f"Seed used: {seed}")
    if clip_accuracy is not None:
        print(f"Note: this sample is small, so its accuracy will vary from "
              f"run to run; the clip-level accuracy of this experimental "
              f"pipeline on all 1,618 validation clips "
              f"is {clip_accuracy:.4f} (see meta.json).")
    else:
        print("Note: this sample is small, so its accuracy will vary from "
              "run to run; run meta.json is missing, so no full-validation "
              "accuracy is quoted.")
    _print_class_table(per_class, pool_counts=pool_classes,
                       total_cases=total, total_correct=correct,
                       total_errors=error_count)
    print("Per-class precision/recall/F1 are macro-averaged in the TOTAL "
          "row; from a small number of clips per class they are very noisy, "
          "and a class with 0 shown was not tested in this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
