#!/usr/bin/env python3
"""Prepare MFEA records for a natural-output arm and clean paired claim probes.

The controlled probes are deliberately short canonical claim panels. They do
not edit a natural rationale in place: that would leave the old cue wording in
the text and create an internally contradictory example. Natural VLM outputs
are preserved as their own evaluation arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral", "contempt"]
EMOTION_ALIASES = {
    "angry": "anger", "disgusted": "disgust", "fearful": "fear",
    "happy": "happiness", "sad": "sadness", "surprised": "surprise",
}
AU_PATTERN = re.compile(r"^AU\d+$", re.IGNORECASE)
AU_NAMES = {
    "AU1": "inner brow raiser", "AU2": "outer brow raiser", "AU4": "brow lowerer",
    "AU5": "upper lid raiser", "AU6": "cheek raiser", "AU7": "lid tightener",
    "AU9": "nose wrinkler", "AU10": "upper lip raiser", "AU12": "lip corner puller",
    "AU14": "dimpler", "AU15": "lip corner depressor", "AU17": "chin raiser",
    "AU20": "lip stretcher", "AU23": "lip tightener", "AU24": "lip presser",
    "AU25": "lips part", "AU26": "jaw drop", "AU27": "mouth stretch",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def canonical_emotion(label: str) -> str:
    value = label.strip().lower()
    return EMOTION_ALIASES.get(value, value)


def safe_image_path(image_root: Path | None, relative_path: Any) -> tuple[str | None, bool | None]:
    if image_root is None or not relative_path:
        return None, None
    root = image_root.resolve()
    candidate = (root / str(relative_path)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Image path escapes --image-root: {relative_path}")
    return str(candidate), candidate.is_file()


def au_panel(au: str, style: str = "brief") -> str:
    name = AU_NAMES.get(au, "action-unit code")
    if style == "expanded":
        return f"The candidate explanation makes one specific visible-cue claim: {au}, the {name}. No other cue claim is added."
    return f"Visible facial-cue claim: {au} ({name})."


def label_panel(label: str, style: str = "brief") -> str:
    if style == "expanded":
        return f"The candidate explanation states one dataset-style visual category: {label}. This is a category claim only, not a claim about the person's inner state."
    return f"Dataset-style visual category claim: {label}."


def control_record(
    common: dict[str, Any], *, condition: str, family: str, rationale: str,
    pair_id: str, claimed_aus: list[str] | None = None,
    claimed_emotions: list[str] | None = None, changed_claim: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        **common,
        "condition": condition,
        "family": family,
        "pair_id": pair_id,
        "rationale": rationale,
        "claimed_aus": claimed_aus or [],
        "claimed_emotions": claimed_emotions or [],
        "generator_id": "deterministic_control_template_v2",
        "prompt_version": "controlled_claim_panel_v2",
        "explanation_grounding": "controlled claim derived from a released record; not a natural model explanation",
    }
    if changed_claim:
        result["changed_claim"] = changed_claim
    return result


def build_records(rows: list[dict[str, Any]], image_root: Path | None) -> list[dict[str, Any]]:
    au_vocabulary = sorted({
        au.upper()
        for row in rows
        for au in strings(row.get("AUs", row.get("aus")))
        if AU_PATTERN.match(au)
    })
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("id") or row.get("image") or f"row-{index:06d}")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID in annotations: {sample_id}")
        seen_ids.add(sample_id)
        recorded_aus = sorted({au.upper() for au in strings(row.get("AUs", row.get("aus"))) if AU_PATTERN.match(au)})
        recorded_emotions = sorted({canonical_emotion(label) for label in strings(row.get("labels", row.get("emotion_labels")))})
        image_path, image_exists = safe_image_path(image_root, row.get("image"))
        common = {
            "sample_id": sample_id,
            "source": row.get("source") or "unknown",
            "image_path": image_path,
            "image_exists": image_exists,
            "recorded_emotions": recorded_emotions,
            "recorded_aus": recorded_aus,
        }
        output.append({
            **common,
            "condition": "natural_prompt_source",
            "family": "generation_source",
            "generation_source": True,
            "rationale": "",
            "claimed_aus": [],
            "claimed_emotions": [],
        })

        if recorded_aus:
            reference = recorded_aus[0]
            foil = next((au for au in au_vocabulary if au not in set(recorded_aus)), None)
            if foil:
                pair_id = f"{sample_id}::au"
                output.append(control_record(
                    common, condition="au_reference", family="controlled_au_pair",
                    rationale=au_panel(reference), pair_id=pair_id, claimed_aus=[reference],
                ))
                output.append(control_record(
                    common, condition="au_foil", family="controlled_au_pair",
                    rationale=au_panel(foil), pair_id=pair_id, claimed_aus=[foil],
                    changed_claim={"replaced": reference, "with": foil},
                ))
                output.append(control_record(
                    common, condition="au_style_expanded", family="controlled_au_style",
                    rationale=au_panel(reference, "expanded"), pair_id=pair_id, claimed_aus=[reference],
                ))

        if recorded_emotions:
            reference_label = recorded_emotions[0]
            foil_label = next((label for label in EMOTIONS if label not in set(recorded_emotions)), None)
            if foil_label:
                pair_id = f"{sample_id}::label"
                output.append(control_record(
                    common, condition="label_reference", family="controlled_label_pair",
                    rationale=label_panel(reference_label), pair_id=pair_id,
                    claimed_emotions=[reference_label],
                ))
                output.append(control_record(
                    common, condition="label_foil", family="controlled_label_pair",
                    rationale=label_panel(foil_label), pair_id=pair_id,
                    claimed_emotions=[foil_label],
                    changed_claim={"replaced": reference_label, "with": foil_label},
                ))
                output.append(control_record(
                    common, condition="label_style_expanded", family="controlled_label_style",
                    rationale=label_panel(reference_label, "expanded"), pair_id=pair_id,
                    claimed_emotions=[reference_label],
                ))
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Official MFEA-Main JSONL")
    parser.add_argument("--image-root", type=Path, help="Root directory of lawfully obtained source images")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.annotations)
    records = build_records(rows, args.image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    conditions: dict[str, int] = {}
    for record in records:
        conditions[record["condition"]] = conditions.get(record["condition"], 0) + 1
    metadata = {
        "schema_version": 2,
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "image_root": str(args.image_root.resolve()) if args.image_root else None,
        "images": len(rows),
        "records": len(records),
        "conditions": conditions,
        "selection_rule": "lexicographically first recorded AU/label; first vocabulary/category value not recorded as foil",
        "important_limit": "Reference and foil are record-derived claims. Human ratings determine visual support; omitted annotations are not proof of absence.",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
