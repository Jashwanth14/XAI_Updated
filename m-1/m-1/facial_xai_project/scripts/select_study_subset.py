#!/usr/bin/env python3
"""Select a deterministic source-stratified sample without splitting image groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            if not value.get("sample_id"):
                raise ValueError(f"Line {line_no} has no sample_id")
            rows.append(value)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def allocate_quotas(group_sizes: dict[str, int], requested: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    target = min(max(requested, 0), total)
    quotas = {key: (target * size) // total for key, size in group_sizes.items()}
    remainder = target - sum(quotas.values())
    ranked = sorted(
        group_sizes,
        key=lambda key: (-(target * group_sizes[key] % total), key),
    )
    for key in ranked[:remainder]:
        quotas[key] += 1
    return quotas


def read_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "sample_id" not in (reader.fieldnames or []):
                raise ValueError(f"Exclusion manifest {path} needs a sample_id column")
            excluded.update(str(row["sample_id"]) for row in reader if row.get("sample_id"))
    return excluded


def readable_image(row: dict[str, Any]) -> bool:
    image_path = row.get("image_path")
    if not image_path or not Path(str(image_path)).is_file():
        return False
    try:
        from PIL import Image

        with Image.open(str(image_path)) as image:
            image.verify()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared audit JSONL")
    parser.add_argument("--n", type=int, required=True, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--strata-field", default="source", help="Record field used for proportional stratification")
    parser.add_argument("--require-existing-images", action="store_true", help="Exclude records without a readable local image path")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[], help="CSV manifest whose sample IDs are excluded")
    parser.add_argument("--output-items", type=Path, required=True, help="Subset JSONL preserving all probe rows per selected image")
    parser.add_argument("--output-manifest", type=Path, required=True, help="CSV with one row per selected image")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")

    rows = read_jsonl(args.input)
    per_image: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("generation_source") or row.get("condition") == "annotation_reference":
            per_image.setdefault(str(row["sample_id"]), row)
    if not per_image:
        raise ValueError("Input has no generation_source records")

    excluded = read_excluded_ids(args.exclude_manifest)
    groups: dict[str, list[str]] = defaultdict(list)
    image_ineligible: set[str] = set()
    for sample_id, row in per_image.items():
        if sample_id in excluded:
            continue
        if args.require_existing_images and not readable_image(row):
            image_ineligible.add(sample_id)
            continue
        source = row.get(args.strata_field)
        group = str(source).strip() if source not in (None, "") else "unknown"
        groups[group].append(sample_id)
    if not groups:
        raise ValueError("No eligible images remain after exclusions")

    quotas = allocate_quotas({name: len(ids) for name, ids in groups.items()}, args.n)
    rng = random.Random(args.seed)
    selected: set[str] = set()
    for group, ids in sorted(groups.items()):
        shuffled = sorted(ids)
        rng.shuffle(shuffled)
        selected.update(shuffled[: quotas[group]])

    selected_rows = [row for row in rows if str(row["sample_id"]) in selected]
    args.output_items.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_items.open("w", encoding="utf-8") as stream:
        for row in selected_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.output_manifest.open("w", encoding="utf-8", newline="") as stream:
        fields = ["sample_id", "source", "recorded_emotions", "recorded_aus"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample_id in sorted(selected):
            row = per_image[sample_id]
            writer.writerow({
                "sample_id": sample_id,
                "source": row.get(args.strata_field, "unknown"),
                "recorded_emotions": json.dumps(row.get("recorded_emotions", []), ensure_ascii=False),
                "recorded_aus": json.dumps(row.get("recorded_aus", []), ensure_ascii=False),
            })

    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    meta = {
        "input": str(args.input.resolve()),
        "input_sha256": digest,
        "seed": args.seed,
        "strata_field": args.strata_field,
        "requested_n": args.n,
        "selected_n": len(selected),
        "excluded_n": len(excluded & set(per_image)),
        "missing_or_unreadable_image_n": len(image_ineligible),
        "strata_selected": {name: quotas[name] for name in sorted(quotas)},
        "selection": "proportional largest-remainder allocation; randomized within source strata",
        "output_items_sha256": hashlib.sha256(args.output_items.read_bytes()).hexdigest(),
        "output_manifest_sha256": hashlib.sha256(args.output_manifest.read_bytes()).hexdigest(),
    }
    args.output_manifest.with_suffix(args.output_manifest.suffix + ".metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps({**meta, "output_items": str(args.output_items), "output_manifest": str(args.output_manifest)}, indent=2))


if __name__ == "__main__":
    main()
