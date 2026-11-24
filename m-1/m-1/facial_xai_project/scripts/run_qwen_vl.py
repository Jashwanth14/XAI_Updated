#!/usr/bin/env python3
"""Run free, local Qwen3-VL inference over prepared MFEA audit JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENERATOR_PROMPT = """Describe only visible facial movements in the image. Do not infer a person's inner emotional state, personality, intent, health, or demographic identity. If you provide an emotion category, treat it only as a dataset-style visual label, choose one of neutral, anger, disgust, fear, happiness, sadness, surprise, contempt, and use null if uncertain. Return one JSON object with keys: emotion_label (string or null), claimed_aus (list of AU codes only when you can identify them visually), rationale (at most 60 words, specific to visible facial movement). Do not make up action units. If uncertain, use an empty AU list."""

JUDGE_PROMPT = """Rate the evidence support of this explanation for the visible face in the image. Judge only whether the stated visible facial-action cues are supported by the pixels. Do not reward fluency, length, confidence, or the emotion label. Do not infer the person's internal emotional state. A released annotation rendering is a probe and may describe recorded annotations rather than claim that you independently verified them. Return exactly one JSON object with keys: evidence_support_score (integer 0-4), dataset_label_consistency_score (integer 0-4 or null if no label is stated), claimed_AUs (array of AU codes explicitly claimed), claimed_emotions (array of dataset label terms explicitly stated), brief_reason (one short sentence). Rubric for evidence_support_score: 0=claims contradicted/unrelated; 1=mostly unsupported; 2=mixed/uncertain support; 3=most specific cue claims visible; 4=all specific cue claims supported. If the face is too small, obscured, or ambiguous, use a low score rather than guess.\n\nExplanation:\n{rationale}"""

AU_NAMES = {
    "AU1": "inner brow raiser", "AU2": "outer brow raiser", "AU4": "brow lowerer",
    "AU5": "upper lid raiser", "AU6": "cheek raiser", "AU7": "lid tightener",
    "AU9": "nose wrinkler", "AU10": "upper lip raiser", "AU12": "lip corner puller",
    "AU14": "dimpler", "AU15": "lip corner depressor", "AU17": "chin raiser",
    "AU20": "lip stretcher", "AU23": "lip tightener", "AU24": "lip presser",
    "AU25": "lips part", "AU26": "jaw drop", "AU27": "mouth stretch",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_no} is not a JSON object")
                rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_keys(path: Path, mode: str) -> set[str]:
    """Read only successful records so interrupted/error rows can be retried."""
    if not path.exists():
        return set()
    done: set[str] = set()
    if mode == "generate":
        for record in load_jsonl(path):
            if record.get("sample_id") and not record.get("error"):
                done.add(str(record["sample_id"]))
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for record in csv.DictReader(stream):
                if record.get("sample_id") and record.get("condition") and not record.get("error"):
                    done.add(f"{record['sample_id']}\t{record['condition']}")
    return done


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_response(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    # Qwen can emit a reasoning block before its final answer. Work backwards
    # so an earlier brace in that block cannot corrupt parsing of the final JSON.
    for start in reversed([index for index, char in enumerate(cleaned) if char == "{"]):
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def make_prompt(mode: str, row: dict[str, Any]) -> str:
    if mode == "generate":
        return GENERATOR_PROMPT
    aus = row.get("claimed_aus", row.get("recorded_aus", []))
    emotions = row.get("claimed_emotions", row.get("recorded_emotions", []))
    if isinstance(aus, str):
        aus = [aus]
    if isinstance(emotions, str):
        emotions = [emotions]
    au_claim_text = ", ".join(f"{str(au).upper()} ({AU_NAMES.get(str(au).upper(), 'action-unit code')})" for au in aus or [])
    emotion_claim_text = ", ".join(str(label) for label in emotions or [])
    stated_claims = f"Explicit AU claims: {au_claim_text or 'none supplied'}. Stated dataset emotion label(s): {emotion_claim_text or 'none supplied'}.\n\n"
    rationale = stated_claims + str(row.get("rationale", ""))
    return JUDGE_PROMPT.format(rationale=rationale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "judge"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Prepared JSONL from prepare_mfea_audit.py or combined study JSONL")
    parser.add_argument("--output", type=Path, required=True, help="JSONL for generation; CSV for judge mode")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--revision", help="Pinned Hugging Face commit/tag; recommended for reproducible runs")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, help="Optional cap after sample-manifest selection; does not sample randomly")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--resume", action="store_true", help="Skip successful records already in the output")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files during this run")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists() and not args.resume:
        parser.error(f"Output already exists: {args.output}. Choose a new path or pass --resume.")

    rows = load_jsonl(args.input)
    if args.mode == "generate":
        seen: set[str] = set()
        unique = []
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            is_generation_source = bool(row.get("generation_source"))
            # Backward-compatible selector for pre-v2 prepared files.
            is_legacy_source = row.get("condition") == "annotation_reference"
            if sample_id and sample_id not in seen and (is_generation_source or is_legacy_source):
                seen.add(sample_id)
                unique.append(row)
        rows = unique
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    already_done = completed_keys(args.output, args.mode) if args.resume else set()
    rows = [row for row in rows if (
        str(row.get("sample_id", "")) if args.mode == "generate"
        else f"{row.get('sample_id', '')}\t{row.get('condition', 'unknown')}"
    ) not in already_done]
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    old_metadata: dict[str, Any] = {}
    if args.resume and metadata_path.exists():
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old_metadata.get("schema_version") != 2:
            parser.error("Cannot resume: output metadata schema is from an older run script. Use a new output path.")
        if not old_metadata.get("sessions"):
            parser.error("Cannot resume: output metadata has no run session. Use a new output path.")
        previous = old_metadata["sessions"][-1]
        current_input_hash = sha256_file(args.input)
        for field, value in (("mode", args.mode), ("model_requested", args.model),
                             ("revision_requested", args.revision), ("input_sha256", current_input_hash),
                             ("max_new_tokens", args.max_new_tokens), ("seed", args.seed)):
            if previous.get(field) != value:
                parser.error(f"Cannot resume: {field} differs from the existing run metadata. Use a new output path.")
        if old_metadata.get("prompts") != {"generator": GENERATOR_PROMPT, "judge": JUDGE_PROMPT}:
            parser.error("Cannot resume: prompt text differs from the existing run metadata. Use a new output path.")
    elif args.resume and args.output.exists():
        parser.error("Cannot resume an output without its metadata JSON. Use a new output path.")
    sessions = list(old_metadata.get("sessions", []))
    session = {
        "started_utc": now_utc(), "mode": args.mode, "model_requested": args.model,
        "revision_requested": args.revision, "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input), "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "limit": args.limit,
        "python": platform.python_version(), "status": "running", "attempted": 0,
        "successes": 0, "errors": 0,
    }
    sessions.append(session)
    metadata = {"schema_version": 2, "output": str(args.output.resolve()), "sessions": sessions,
                "prompts": {"generator": GENERATOR_PROMPT, "judge": JUDGE_PROMPT}}
    save_metadata(metadata_path, metadata)
    if not rows:
        session.update({"finished_utc": now_utc(), "status": "complete", "note": "No pending records."})
        save_metadata(metadata_path, metadata)
        print(json.dumps({"pending": 0, "output": str(args.output), "resume": args.resume}, indent=2))
        return

    # Heavy imports are deferred so data preparation and adapter commands need no GPU stack.
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from_pretrained = {"dtype": "auto", "device_map": "auto", "local_files_only": args.local_files_only}
    if args.revision:
        from_pretrained["revision"] = args.revision
    model = AutoModelForImageTextToText.from_pretrained(args.model, **from_pretrained)
    processor = AutoProcessor.from_pretrained(args.model, **{
        key: value for key, value in from_pretrained.items() if key != "dtype" and key != "device_map"
    })
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
    if args.resume and old_metadata.get("sessions"):
        prior_revision = old_metadata["sessions"][-1].get("model_revision_resolved")
        if prior_revision and resolved_revision and prior_revision != resolved_revision:
            parser.error("Cannot resume: the resolved model revision differs. Pin the original revision or use a new output path.")
    session.update({
        "model_revision_resolved": resolved_revision,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "platform": platform.platform(),
        "device_map": "auto", "model_device": str(model.device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "loading_dtype": "auto", "quantization": "none",
        "processor_revision_resolved": getattr(processor, "_commit_hash", None),
    })
    save_metadata(metadata_path, metadata)

    judge_fields = [
        "sample_id", "condition", "judge_id", "judge_revision", "judge_score", "evidence_support_score",
        "dataset_label_consistency_score", "claimed_aus", "claimed_emotions", "brief_reason", "inference_seconds", "raw_output", "error",
    ]
    mode = "a" if args.resume and args.output.exists() else "w"
    if args.mode == "judge" and mode == "a":
        with args.output.open("r", encoding="utf-8-sig", newline="") as existing_stream:
            existing_fields = next(csv.reader(existing_stream), [])
        if existing_fields != judge_fields:
            parser.error("Cannot resume: judge CSV columns differ from this script version. Use a new output path.")
    with args.output.open(mode, encoding="utf-8", newline="") as stream:
        writer = None
        if args.mode == "judge":
            writer = csv.DictWriter(stream, fieldnames=judge_fields)
            if mode == "w" or args.output.stat().st_size == 0:
                writer.writeheader()
                stream.flush()
        for index, row in enumerate(rows, start=1):
            record_started = time.perf_counter()
            sample_id = str(row.get("sample_id", ""))
            image_path = row.get("image_path")
            result_text = ""
            parsed: dict[str, Any] | None = None
            error: str | None = None
            if not image_path or not Path(str(image_path)).is_file():
                error = "missing_or_unreadable_image"
            else:
                try:
                    image_uri = Path(str(image_path)).resolve().as_uri()
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image_uri},
                        {"type": "text", "text": make_prompt(args.mode, row)},
                    ]}]
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    ).to(model.device)
                    inputs.pop("token_type_ids", None)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                    generated_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
                    result_text = processor.batch_decode(
                        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    parsed = parse_response(result_text)
                    error = None if parsed is not None else "response_not_valid_json"
                except Exception as exc:  # keep a per-record journal even for isolated image/model errors
                    error = f"inference_error:{type(exc).__name__}:{str(exc)[:300]}"

            if args.mode == "generate":
                parsed = parsed or {}
                rationale = parsed.get("rationale", "")
                if not isinstance(rationale, str) or not rationale.strip():
                    error = error or "missing_or_invalid_rationale"
                record = {
                    "sample_id": sample_id, "image_path": image_path, "condition": "original_vlm",
                    "rationale": rationale if isinstance(rationale, str) and rationale.strip() else "",
                    "claimed_aus": parsed.get("claimed_aus", []),
                    "claimed_emotions": [parsed["emotion_label"]] if parsed.get("emotion_label") else [],
                    "generator_id": args.model, "generator_revision": resolved_revision,
                    "prompt_version": "generator_v1", "explanation_grounding": "image included in VLM input",
                    "inference_seconds": round(time.perf_counter() - record_started, 3),
                    "raw_output": result_text, "parsed_output": parsed or None, "error": error,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                parsed = parsed or {}
                score = parsed.get("evidence_support_score", "")
                if not isinstance(score, int) or isinstance(score, bool) or score not in range(5):
                    error = error or "missing_or_invalid_evidence_support_score"
                    score = ""
                label_score = parsed.get("dataset_label_consistency_score", "")
                if label_score is not None and label_score != "" and (
                    not isinstance(label_score, int) or isinstance(label_score, bool) or label_score not in range(5)
                ):
                    label_score = ""
                    error = error or "invalid_dataset_label_consistency_score"
                writer.writerow({
                    "sample_id": sample_id, "condition": row.get("condition", "unknown"),
                    "judge_id": args.model, "judge_revision": resolved_revision,
                    "judge_score": score, "evidence_support_score": score,
                    "dataset_label_consistency_score": "" if label_score is None else label_score,
                    "claimed_aus": json.dumps(parsed.get("claimed_AUs", []), ensure_ascii=False),
                    "claimed_emotions": json.dumps(parsed.get("claimed_emotions", []), ensure_ascii=False),
                    "brief_reason": parsed.get("brief_reason", ""),
                    "inference_seconds": round(time.perf_counter() - record_started, 3),
                    "raw_output": result_text, "error": error or "",
                })
            stream.flush()
            session["attempted"] += 1
            session["errors"] += int(bool(error))
            session["successes"] += int(not error)
            print(f"{index}/{len(rows)} {sample_id} {error or 'ok'}")

    session.update({"finished_utc": now_utc(), "status": "complete"})
    save_metadata(metadata_path, metadata)
    print(json.dumps({"attempted": session["attempted"], "successes": session["successes"],
                      "errors": session["errors"], "output": str(args.output),
                      "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
