#!/usr/bin/env python
"""Run the YAMNet + logistic pipeline on a random sample of VALIDATION clips.

Draws NUM_CASES clips at random (without replacement) from the whole
validation split only: data/splits/clip_splits.csv filtered to split == val,
joined on clip_id with data/provenance/phase2a_clip_ledger.csv filtered to
status == kept. Every clip runs the application's final prediction path
src/inference/yamnet_logistic.py (its audio loader, YAMNet loader and predictor
are reused by import; the saved model and YAMNet signature are loaded once per
run). A clip that fails to load
or classify is printed as a row with ERROR and its reason, counted separately
and never skipped silently. Output is an ASCII table plus a summary block and
is printed to stdout only; nothing is written, so this script never overwrites
any file.

Editable module constants (change them here, or override per run via --n and
--seed):
    NUM_CASES = 20      number of clips to sample per run
    SEED = None         repeat seed for a repeatable sample; None picks a
                        random seed every run and prints it

Usage:
  .venv/Scripts/python scripts/demo_table.py
  .venv/Scripts/python scripts/demo_table.py --n 50
  .venv/Scripts/python scripts/demo_table.py --seed 12345

The sample is small, so its accuracy varies from run to run; the accuracy on
all 1,618 validation clips is 0.7769.
"""

from __future__ import annotations

import argparse
import csv
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.inference import yamnet_logistic

SPLIT_CSV = ROOT / "data" / "splits" / "clip_splits.csv"
LEDGER_CSV = ROOT / "data" / "provenance" / "phase2a_clip_ledger.csv"
VALIDATION_SPLIT = "val"
KEPT_STATUS = "kept"
EXPECTED_POOL_SIZE = 1618

NUM_CASES = 720
SEED = None

TABLE_HEADERS = ["#", "source_dataset", "file name", "ACTUAL class",
                 "PREDICTED class", "confidence", "duration", "OK"]
CLASS_HEADERS = ["class", "in pool", "shown", "correct", "wrong",
                 "errors", "accuracy"]


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


def _clip_path(local_path: str) -> Path:
    """Resolve a ledger local_path (either slash style) against the repo."""
    path = Path(local_path.replace("\\", "/"))
    if path.is_absolute():
        return path
    return ROOT / path


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


def _print_class_table(per_class: dict[str, dict[str, int]],
                       pool_counts: dict[str, int],
                       total_cases: int,
                       total_correct: int,
                       total_errors: int) -> None:
    """Print the per-class summary block in class_names.txt order."""
    rows: list[list[str]] = []
    for class_name in per_class:
        counts = per_class[class_name]
        classified = counts["correct"] + counts["wrong"]
        accuracy = "-"
        if classified > 0:
            accuracy = f"{counts['correct'] / classified:.1%}"
        rows.append([
            class_name,
            str(pool_counts.get(class_name, 0)),
            str(counts["shown"]),
            str(counts["correct"]),
            str(counts["wrong"]),
            str(counts["errors"]),
            accuracy,
        ])
    total_shown = sum(counts["shown"] for counts in per_class.values())
    sum_errors = sum(counts["errors"] for counts in per_class.values())
    sum_wrong = total_cases - total_correct - total_errors
    accuracy = "-"
    if total_cases > 0:
        accuracy = f"{total_correct / total_cases:.1%}"
    total_row = [
        "TOTAL",
        str(sum(pool_counts.values())),
        str(total_shown),
        str(total_correct),
        str(sum_wrong),
        str(sum_errors),
        accuracy,
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
                            [False, True, True, True, True, True, True]))


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

    pipeline = yamnet_logistic.load_saved_pipeline()
    names = list(pipeline.class_names)
    print(f"saved pipeline: {pipeline.model_dir}")
    signature = yamnet_logistic.load_yamnet_signature()

    rows: list[list[str]] = []
    correct = 0
    errors: list[tuple[str, str]] = []
    per_class: dict[str, dict[str, int]] = {}
    for class_name in names:
        per_class[class_name] = {
            "shown": 0,
            "correct": 0,
            "wrong": 0,
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
            audio = yamnet_logistic.load_audio(
                _clip_path(ledger_row.get("local_path", clip_id)))
            order, proba = yamnet_logistic.embed_and_predict(
                signature, audio, pipeline.scaler, pipeline.classifier)
            predicted = names[order[0]]
            confidence = f"{proba[order[0]]:.2%}"
            duration = f"{audio.shape[0] / config.SAMPLE_RATE:.2f}"
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

        rows.append([str(index), source_dataset, file_name, actual,
                     predicted, confidence, duration, ok])

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
    print("Note: this sample is small, so its accuracy will vary from run to "
          "run; the accuracy on all 1,618 validation clips is 0.7769.")
    _print_class_table(per_class, pool_counts=pool_classes,
                       total_cases=total, total_correct=correct,
                       total_errors=error_count)
    print("Per-class accuracies from a small number of clips per class are "
          "very noisy, and a class with 0 shown was not tested in this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())