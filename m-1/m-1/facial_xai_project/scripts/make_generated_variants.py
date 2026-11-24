#!/usr/bin/env python3
"""Combine clean controlled probes with successful natural VLM explanations.

Controlled AU/label panels come from prepare_mfea_audit.py. Natural VLM
outputs are a separate arm; this script never rewrites a natural rationale
into a contradictory pseudo-edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def as_strings(value: Any, uppercase: bool = False) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result = {str(item).strip() for item in value if str(item).strip()}
    return sorted(item.upper() if uppercase else item.lower() for item in result)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True, help="JSONL from run_qwen_vl.py --mode generate")
    parser.add_argument("--items", type=Path, required=True, help="JSONL from prepare_mfea_audit.py")
    parser.add_argument("--output", type=Path, required=True, help="Combined study variants JSONL")
    args = parser.parse_args()

    generated = read_jsonl(args.generated)
    items = read_jsonl(args.items)
    sources = {str(row["sample_id"]): row for row in items if row.get("generation_source")}
    if not sources:
        raise ValueError("Items lack generation_source rows. Regenerate them with prepare_mfea_audit.py.")

    output = [row for row in items if not row.get("generation_source")]
    failures: list[dict[str, Any]] = []
    natural_ids: set[str] = set()
    for generation in generated:
        sample_id = str(generation.get("sample_id", ""))
        if sample_id in natural_ids:
            failures.append({"sample_id": sample_id, "reason": "duplicate_generation_record"})
            continue
        source = sources.get(sample_id)
        if source is None:
            failures.append({"sample_id": sample_id, "reason": "sample_id_not_in_items"})
            continue
        if generation.get("error"):
            failures.append({"sample_id": sample_id, "reason": str(generation["error"])})
            continue
        rationale = generation.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            failures.append({"sample_id": sample_id, "reason": "missing_rationale"})
            continue
        natural_ids.add(sample_id)
        output.append({
            "sample_id": sample_id,
            "source": source.get("source", "unknown"),
            "image_path": source.get("image_path") or generation.get("image_path"),
            "image_exists": source.get("image_exists"),
            "recorded_emotions": source.get("recorded_emotions", []),
            "recorded_aus": source.get("recorded_aus", []),
            "condition": "natural_vlm",
            "family": "natural_output",
            "rationale": rationale.strip(),
            "claimed_aus": as_strings(generation.get("claimed_aus"), uppercase=True),
            "claimed_emotions": as_strings(generation.get("claimed_emotions")),
            "generator_id": generation.get("generator_id"),
            "generator_revision": generation.get("generator_revision"),
            "prompt_version": generation.get("prompt_version"),
            "explanation_grounding": generation.get("explanation_grounding", "image included in VLM input"),
            "raw_output": generation.get("raw_output", ""),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in output:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    failures_path = args.output.with_suffix(args.output.suffix + ".generation_failures.jsonl")
    with failures_path.open("w", encoding="utf-8") as stream:
        for record in failures:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    condition_counts: dict[str, int] = {}
    for record in output:
        condition_counts[str(record["condition"])] = condition_counts.get(str(record["condition"]), 0) + 1
    metadata = {
        "schema_version": 2,
        "items": str(args.items.resolve()), "items_sha256": sha256(args.items),
        "generated": str(args.generated.resolve()), "generated_sha256": sha256(args.generated),
        "natural_output_records": len(natural_ids), "generation_failures": len(failures),
        "condition_counts": condition_counts, "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "design": "controlled claim panels and natural VLM outputs are separate analysis arms",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "failure_report": str(failures_path), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
