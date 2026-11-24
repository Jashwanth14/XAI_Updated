#!/usr/bin/env python3
"""Summarize paired judge scores and optional claim-to-annotation matches."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PAIRS = [
    ("au_reference", "au_foil", "au_reference_minus_foil"),
    ("label_reference", "label_foil", "label_reference_minus_foil"),
    ("au_style_expanded", "au_reference", "au_style_minus_reference"),
    ("label_style_expanded", "label_reference", "label_style_minus_reference"),
]
EMOTION_ALIASES = {"happy": "happiness", "sad": "sadness", "angry": "anger", "fearful": "fear", "surprised": "surprise", "disgusted": "disgust"}


def parse_list(value: Any) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        try:
            decoded = json.loads(text)
            items = decoded if isinstance(decoded, list) else [decoded]
        except json.JSONDecodeError:
            items = [piece.strip() for piece in text.split(",")]
    return {str(item).strip().lower() for item in items if str(item).strip()}


def read_items(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["sample_id"])] = row
    return result


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 10_000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    lo = estimates[int(0.025 * (samples - 1))]
    hi = estimates[int(0.975 * (samples - 1))]
    return lo, hi


def claim_metrics(predicted: set[str], recorded: set[str]) -> dict[str, float | int]:
    matched = len(predicted & recorded)
    precision = matched / len(predicted) if predicted else float("nan")
    recall = matched / len(recorded) if recorded else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    return {"claim_count": len(predicted), "recorded_count": len(recorded), "intersection": matched,
            "annotation_precision": precision, "annotation_recall": recall, "annotation_f1": f1}


def normalize_emotion_set(values: set[str]) -> set[str]:
    return {EMOTION_ALIASES.get(value, value) for value in values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, nargs="+", required=True, help="One or more CSVs with sample_id, condition, judge_score")
    parser.add_argument("--items", type=Path, help="JSONL produced by prepare_mfea_audit.py; enables annotation-match metrics")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_index = read_items(args.items)

    scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    label_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    errors: dict[tuple[str, str], int] = defaultdict(int)
    rows_seen: dict[tuple[str, str], int] = defaultdict(int)
    claim_rows: list[dict[str, Any]] = []
    required = {"sample_id", "condition", "judge_score"}
    for judgment_path in args.judgments:
        with judgment_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Judgment CSV {judgment_path} is missing required columns: {sorted(missing)}")
            for row in reader:
                sample_id = str(row["sample_id"])
                condition = str(row["condition"])
                judge_id = str(row.get("judge_id") or "judge_1")
                rows_seen[(judge_id, condition)] += 1
                if row.get("error"):
                    errors[(judge_id, condition)] += 1
                try:
                    score = float(row["judge_score"])
                except (TypeError, ValueError):
                    score = None
                if score is not None:
                    scores[judge_id][sample_id][condition].append(score)
                try:
                    label_score = float(row.get("dataset_label_consistency_score", ""))
                except (TypeError, ValueError):
                    label_score = None
                if label_score is not None:
                    label_scores[judge_id][sample_id][condition].append(label_score)

                item = item_index.get(sample_id)
                if item and (row.get("claimed_aus") or row.get("claimed_emotions")):
                    gold_aus = {str(value).lower() for value in item.get("recorded_aus", [])}
                    gold_emotions = normalize_emotion_set({str(value).lower() for value in item.get("recorded_emotions", [])})
                    pred_aus = parse_list(row.get("claimed_aus"))
                    pred_emotions = normalize_emotion_set(parse_list(row.get("claimed_emotions")))
                    claim_rows.append({
                        "sample_id": sample_id,
                        "judge_id": judge_id,
                        "condition": condition,
                        **{f"au_{k}": v for k, v in claim_metrics(pred_aus, gold_aus).items()},
                        **{f"emotion_{k}": v for k, v in claim_metrics(pred_emotions, gold_emotions).items()},
                    })

    paired_rows: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    all_judges = sorted(set(scores) | {judge_id for judge_id, _ in errors})
    for judge_id in all_judges:
        conditions_seen = sorted(
            {condition for sample in scores.get(judge_id, {}).values() for condition in sample}
            | {condition for error_judge, condition in errors if error_judge == judge_id}
        )
        for condition in conditions_seen:
            values = [sample[condition] for sample in scores.get(judge_id, {}).values() if condition in sample]
            flattened = [score for sample_values in values for score in sample_values]
            condition_rows.append({
                "judge_id": judge_id,
                "condition": condition,
                "n_images_with_numeric_score": len(values),
                "n_numeric_scores": len(flattened),
                "n_rows_seen": rows_seen[(judge_id, condition)],
                "mean_score": statistics.mean(flattened) if flattened else float("nan"),
                "median_score": statistics.median(flattened) if flattened else float("nan"),
                "parse_or_inference_errors": errors[(judge_id, condition)],
            })
        for left, right, name in PAIRS:
            diffs: list[float] = []
            for sample_id in sorted(scores.get(judge_id, {})):
                conditions = scores[judge_id][sample_id]
                if left not in conditions or right not in conditions:
                    continue
                left_score = statistics.mean(conditions[left])
                right_score = statistics.mean(conditions[right])
                difference = left_score - right_score
                diffs.append(difference)
                paired_rows.append({
                    "judge_id": judge_id,
                    "sample_id": sample_id,
                    "comparison": name,
                    "left_condition": left,
                    "right_condition": right,
                    "left_score": left_score,
                    "right_score": right_score,
                    "left_minus_right": difference,
                    "win": int(difference > 0),
                    "tie": int(difference == 0),
                })
            if diffs:
                lo, hi = bootstrap_mean_ci(diffs, args.seed, args.bootstrap_samples)
                aggregate.append({
                    "judge_id": judge_id,
                    "comparison": name,
                    "n_paired_images": len(diffs),
                    "mean_left_minus_right": statistics.mean(diffs),
                    "median_left_minus_right": statistics.median(diffs),
                    "mean_absolute_difference": statistics.mean(abs(value) for value in diffs),
                    "median_absolute_difference": statistics.median(abs(value) for value in diffs),
                    "bootstrap_95_ci_low": lo,
                    "bootstrap_95_ci_high": hi,
                    "left_win_rate": sum(value > 0 for value in diffs) / len(diffs),
                    "tie_rate": sum(value == 0 for value in diffs) / len(diffs),
                })

    label_paired_rows: list[dict[str, Any]] = []
    label_aggregate: list[dict[str, Any]] = []
    label_condition_rows: list[dict[str, Any]] = []
    for judge_id in sorted(label_scores):
        conditions_seen = sorted({condition for sample in label_scores[judge_id].values() for condition in sample})
        for condition in conditions_seen:
            sample_values = [sample[condition] for sample in label_scores[judge_id].values() if condition in sample]
            flattened = [score for values in sample_values for score in values]
            label_condition_rows.append({
                "judge_id": judge_id, "condition": condition,
                "n_images_with_numeric_score": len(sample_values), "n_numeric_scores": len(flattened),
                "mean_label_consistency_score": statistics.mean(flattened) if flattened else float("nan"),
                "median_label_consistency_score": statistics.median(flattened) if flattened else float("nan"),
            })
        for left, right, name in PAIRS:
            diffs = []
            for sample_id in sorted(label_scores[judge_id]):
                conditions = label_scores[judge_id][sample_id]
                if left not in conditions or right not in conditions:
                    continue
                left_score = statistics.mean(conditions[left])
                right_score = statistics.mean(conditions[right])
                difference = left_score - right_score
                diffs.append(difference)
                label_paired_rows.append({
                    "judge_id": judge_id, "sample_id": sample_id, "comparison": name,
                    "left_condition": left, "right_condition": right,
                    "left_label_score": left_score, "right_label_score": right_score,
                    "left_minus_right": difference,
                })
            if diffs:
                lo, hi = bootstrap_mean_ci(diffs, args.seed + 1, args.bootstrap_samples)
                label_aggregate.append({
                    "judge_id": judge_id, "comparison": name, "n_paired_images": len(diffs),
                    "mean_left_minus_right": statistics.mean(diffs),
                    "mean_absolute_difference": statistics.mean(abs(value) for value in diffs),
                    "bootstrap_95_ci_low": lo, "bootstrap_95_ci_high": hi,
                })

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "paired_deltas.csv", paired_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate)
    write_csv(args.output_dir / "condition_scores.csv", condition_rows)
    write_csv(args.output_dir / "label_consistency_condition_scores.csv", label_condition_rows)
    write_csv(args.output_dir / "label_consistency_paired_deltas.csv", label_paired_rows)
    write_csv(args.output_dir / "label_consistency_aggregate_results.csv", label_aggregate)
    write_csv(args.output_dir / "annotation_claim_matches.csv", claim_rows)

    notes = {
        "interpretation": [
            "Scores measure the configured judge rubric, not a person's internal emotional state.",
            "A claim not listed by the dataset is not automatically false; annotation precision/recall are agreement proxies only.",
            "Controlled reference/foil panels are deterministic probes, not natural-language gold rationales.",
            "Natural VLM outputs are summarized separately and are not treated as matched annotation controls.",
            "Bootstrap resamples are paired by sample_id within judge_id.",
        ],
        "judgment_rows_with_numeric_score": sum(sum(len(values) for values in sample.values()) for samples in scores.values() for sample in samples.values()),
        "parse_or_inference_errors": dict(sorted(((f"{judge}|{condition}", count) for (judge, condition), count in errors.items()))),
        "paired_comparisons": len(paired_rows),
        "claim_rows": len(claim_rows),
    }
    (args.output_dir / "interpretation_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate_rows": len(aggregate), "paired_rows": len(paired_rows), "claim_rows": len(claim_rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
