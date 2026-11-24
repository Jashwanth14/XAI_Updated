#!/usr/bin/env python3
"""Normalize saved mini-1 result.json files for a clearly labeled legacy audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_") or "unknown"


def resolve_image(result_path: Path, record: dict[str, Any], image_root: Path | None) -> Path | None:
    sibling_crop = result_path.parent / "face_crop.jpg"
    if sibling_crop.is_file():
        return sibling_crop.resolve()
    raw = str(record.get("image_path", "")).replace("\\", "/")
    candidates = []
    if raw:
        candidate = Path(raw)
        if candidate.is_absolute():
            candidates.append(candidate)
        if image_root is not None:
            candidates.append(image_root / raw)
            candidates.append(image_root / candidate.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def infer_model_and_identity(result_path: Path, input_root: Path, record: dict[str, Any]) -> tuple[str, str, str]:
    try:
        relative = result_path.relative_to(input_root)
        parts = relative.parts
    except ValueError:
        parts = result_path.parts
    model = parts[0] if len(parts) > 1 else "unknown_model"
    raw_path = str(record.get("image_path", "")).replace("\\", "/")
    if raw_path:
        identity = "/".join(Path(raw_path).parts[-2:])
    else:
        identity = "/".join(parts[1:-1]) if len(parts) > 2 else result_path.parent.name
    split = parts[1] if len(parts) > 3 and parts[1] in {"test_unseen", "test", "validation", "val"} else "legacy_demo"
    return model, split, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="mini-1 outputs directory containing result.json files")
    parser.add_argument("--output", type=Path, required=True, help="Canonical JSONL for run_qwen_vl.py --mode judge")
    parser.add_argument("--image-root", type=Path, help="Optional root for the original source images")
    args = parser.parse_args()

    root = args.input_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Input root is not a directory: {root}")
    paths = sorted(root.rglob("result.json"))
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for result_path in paths:
        try:
            raw_bytes = result_path.read_bytes()
            value = json.loads(raw_bytes)
            if not isinstance(value, dict):
                raise ValueError("top-level JSON value is not an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            skipped.append({"path": str(result_path), "reason": str(exc)})
            continue

        explanation = str(value.get("explanation", "")).strip()
        emotion_obj = value.get("emotion") if isinstance(value.get("emotion"), dict) else {}
        predicted_label = str(emotion_obj.get("label", "")).strip().lower()
        action_obj = value.get("action_units") if isinstance(value.get("action_units"), dict) else {}
        active_aus = action_obj.get("active", [])
        if isinstance(active_aus, str):
            active_aus = [active_aus]
        active_aus = sorted({str(au).upper() for au in active_aus if str(au).strip()})
        attention_obj = value.get("attention") if isinstance(value.get("attention"), dict) else {}
        visualizations = value.get("visualizations") if isinstance(value.get("visualizations"), dict) else {}
        panel = result_path.parent / "xai_panel.png"
        if not explanation:
            skipped.append({"path": str(result_path), "reason": "missing explanation"})
            continue

        model, split, identity = infer_model_and_identity(result_path, root, value)
        sample_id = f"mini1::{slug(split)}::{slug(identity)}"
        image_path = resolve_image(result_path, value, args.image_root)
        if image_path is None:
            skipped.append({"path": str(result_path), "reason": "no readable face_crop or source image found"})
            continue
        condition = f"mini1_legacy_{slug(model)}"
        records.append({
            "sample_id": sample_id,
            "image_path": str(image_path),
            "source": "mini-1 curated/demo output; not a benchmark split",
            "condition": condition,
            "family": "legacy_natural_output",
            "rationale": explanation,
            "claimed_aus": active_aus,
            "claimed_emotions": [predicted_label] if predicted_label else [],
            "generator_id": f"mini-1:{model}",
            "prompt_version": "legacy_prompt_unverified",
            "explanation_grounding": "text-only language model in the legacy pipeline; do not treat as image-conditioned prose",
            "legacy_prediction": predicted_label or None,
            "legacy_confidence": emotion_obj.get("confidence"),
            "legacy_top_predictions": emotion_obj.get("top_predictions", []),
            "legacy_action_unit_descriptions": action_obj.get("descriptions", ""),
            "legacy_attention_regions": attention_obj.get("regions", []),
            "legacy_attention_summary": attention_obj.get("summary", ""),
            "legacy_xai_panel_path": str(panel.resolve()) if panel.is_file() else None,
            "legacy_visualization_paths": visualizations,
            "legacy_active_aus_are_heuristic": True,
            "result_json": str(result_path.resolve()),
            "result_json_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "error": None,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "input_root": str(root),
        "result_json_files": len(paths),
        "normalized_records": len(records),
        "skipped": skipped,
        "interpretation": "Legacy outputs are exploratory only: text-only rationale generation and geometry-derived AU heuristics are not validated visual evidence.",
    }
    args.output.with_suffix(args.output.suffix + ".import_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"result_json_files": len(paths), "normalized_records": len(records), "skipped": len(skipped), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
