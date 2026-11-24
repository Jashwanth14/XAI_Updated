#!/usr/bin/env python3
"""Create blinded offline rating packets for controlled pairs and natural outputs.

The default confirmatory packet has 120 matched control pairs (roughly half AU
and half label pairs) plus 60 natural VLM outputs, divided into two sessions.
Use --copy-images only when the data terms and human-review approval allow a
local copy for each rater.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


CONTROL_ARMS = (
    ("au", "au_reference", "au_foil"),
    ("label", "label_reference", "label_foil"),
)
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
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def source_for(rows: list[dict[str, Any]]) -> str:
    return str(next((row.get("source") for row in rows if row.get("source")), "unknown"))


def proportional_quotas(groups: dict[str, list[str]], requested: int) -> dict[str, int]:
    total = sum(len(ids) for ids in groups.values())
    target = min(requested, total)
    quotas = {name: target * len(ids) // total for name, ids in groups.items()}
    extra = target - sum(quotas.values())
    ordered = sorted(groups, key=lambda name: (-(target * len(groups[name]) % total), name))
    for name in ordered[:extra]:
        quotas[name] += 1
    return quotas


def source_stratified(ids: list[str], groups: dict[str, list[dict[str, Any]]], n: int, seed: int) -> list[str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        by_source[source_for(groups[sample_id])].append(sample_id)
    if not by_source or n < 1:
        return []
    quotas = proportional_quotas(by_source, n)
    rng = random.Random(seed)
    selected: list[str] = []
    for source, values in sorted(by_source.items()):
        values = sorted(values)
        rng.shuffle(values)
        selected.extend(values[:quotas[source]])
    rng.shuffle(selected)
    return selected


def conditions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        condition = str(row.get("condition", ""))
        if condition and condition not in index:
            index[condition] = row
    return index


def select_control_pairs(
    groups: dict[str, list[dict[str, Any]]], target: int, seed: int
) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    indexes = {sample_id: conditions(rows) for sample_id, rows in groups.items()}
    candidates: dict[str, list[str]] = {}
    for arm, reference, foil in CONTROL_ARMS:
        candidates[arm] = sorted(
            sample_id for sample_id, index in indexes.items() if reference in index and foil in index
        )
    if not any(candidates.values()):
        raise ValueError("No complete controlled AU or label reference/foil pairs found")
    target = min(target, len(set(candidates["au"]) | set(candidates["label"])))
    desired = {"au": target // 2, "label": target - (target // 2)}
    selected: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    used: set[str] = set()
    for offset, (arm, reference, foil) in enumerate(CONTROL_ARMS):
        ids = [sample_id for sample_id in candidates[arm] if sample_id not in used]
        for sample_id in source_stratified(ids, groups, desired[arm], seed + offset):
            selected.append((arm, sample_id, indexes[sample_id][reference], indexes[sample_id][foil]))
            used.add(sample_id)
    if len(selected) < target:
        remaining: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
        for arm, reference, foil in CONTROL_ARMS:
            for sample_id in candidates[arm]:
                if sample_id not in used:
                    remaining.append((arm, sample_id, indexes[sample_id][reference], indexes[sample_id][foil]))
        random.Random(seed + 99).shuffle(remaining)
        for candidate in remaining:
            if candidate[1] not in used and len(selected) < target:
                selected.append(candidate)
                used.add(candidate[1])
    return selected


def nonadjacent_shuffle(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Randomize while avoiding adjacent members of the same matched pair."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    for index in range(1, len(shuffled)):
        previous_pair = shuffled[index - 1].get("pair_id")
        if previous_pair and shuffled[index].get("pair_id") == previous_pair:
            replacement = next(
                (position for position in range(index + 1, len(shuffled)) if shuffled[position].get("pair_id") != previous_pair),
                None,
            )
            if replacement is not None:
                shuffled[index], shuffled[replacement] = shuffled[replacement], shuffled[index]
    return shuffled


def usable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        image_path = Path(str(row.get("image_path", ""))).expanduser()
        rationale = row.get("rationale")
        if image_path.is_file() and isinstance(rationale, str) and rationale.strip():
            output.append({**row, "image_path": str(image_path.resolve()), "rationale": rationale.strip()})
    return output


def html_page(rater_id: str, session: int, cards: list[dict[str, Any]]) -> str:
    cards_json = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Facial movement rating · {html.escape(rater_id)} · session {session}</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:20px;color:#17202a;background:#f5f7fa}}
header{{position:sticky;top:0;background:#f5f7fa;padding:12px 0;border-bottom:1px solid #ccd3dc;z-index:2}}
.card{{background:white;border:1px solid #ccd3dc;border-radius:12px;padding:18px;margin:20px 0;scroll-margin-top:100px}}
.face{{display:block;max-width:min(100%,560px);max-height:480px;object-fit:contain;margin:12px auto;background:#eee}}
.rationale{{white-space:pre-wrap;background:#f7f8fa;padding:14px;border-radius:8px}}
label{{display:block;margin:14px 0 5px;font-weight:600}} select,textarea{{font:inherit;padding:8px;width:100%;max-width:620px}}
textarea{{min-height:70px}} .small{{font-size:.9rem;color:#46515c}} button{{font:inherit;padding:9px 14px;margin-right:8px;cursor:pointer}}
</style></head><body>
<header><strong>Facial movement evidence rating</strong> · session {session} · {len(cards)} items · {html.escape(rater_id)}<br>
<span id="progress" class="small"></span><p class="small">Rate only claims about visible facial movement. Do not infer the person's inner emotion, personality, intent, health, or diagnosis. Use “cannot assess” when visibility is inadequate.</p>
<button id="next">Next unanswered</button><button id="download">Download ratings CSV</button></header>
<main id="cards"></main>
<script>
const packet={cards_json};
const rater={json.dumps(rater_id)};
const storageKey="facial-xai-rating-"+rater+"-session-{session}";
let answers={{}};try{{answers=JSON.parse(localStorage.getItem(storageKey)||"{{}}")}}catch(e){{answers={{}}}}
const root=document.getElementById("cards");
function updateProgress(){{const done=packet.filter(x=>answers[x.presentation_id]?.evidence_support_score).length;document.getElementById("progress").textContent=done+" / "+packet.length+" rated"}}
function makeSelect(label,field,options,id){{const wrap=document.createElement("div");const title=document.createElement("label");title.textContent=label;title.htmlFor=id;const sel=document.createElement("select");sel.id=id;sel.dataset.field=field;for(const [value,text] of options){{const o=document.createElement("option");o.value=value;o.textContent=text;sel.appendChild(o)}}wrap.append(title,sel);return wrap}}
for(const item of packet){{
 const card=document.createElement("section");card.className="card";card.id=item.presentation_id;
 const heading=document.createElement("h2");heading.textContent=item.presentation_id;
 const img=document.createElement("img");img.className="face";img.src=item.image_src;img.alt="Face image for visible-cue rating";
 const rationale=document.createElement("div");rationale.className="rationale";rationale.textContent=item.rationale;
 const category=document.createElement("p");category.className="small";category.textContent="Separate stated category claim: "+((item.claimed_emotions||[]).join(", ")||"none stated");
 const score=makeSelect("Visible-cue support for the specific movement claims","evidence_support_score",[["","Choose one"],["1","1 · none verifiable"],["2","2 · mostly unsupported"],["3","3 · mixed or ambiguous"],["4","4 · most cues visible"],["5","5 · all specific cues visible"],["cannot_assess","Cannot assess"]],item.presentation_id+"-score");
 const label=makeSelect("If an emotion category is stated, is it consistent with the visible facial configuration?","label_consistency",[["","Choose one"],["no_label","No emotion category stated"],["consistent","Visually consistent"],["inconsistent","Visually inconsistent"],["ambiguous","Ambiguous / cannot tell"]],item.presentation_id+"-label");
 const unreadable=makeSelect("Image readability","unreadable",[["","Choose one"],["no","Readable enough to judge"],["yes","Too small, obscured, or unusable"]],item.presentation_id+"-readability");
 const notesLabel=document.createElement("label");notesLabel.textContent="Specific movement claims that seem unsupported or ambiguous (optional)";
 const notes=document.createElement("textarea");notes.dataset.field="unsupported_claim_notes";notes.placeholder="For example: ‘lip corners raised’ is not visible; brow movement cannot be assessed.";
 const controls=[...score.querySelectorAll("select"),...label.querySelectorAll("select"),...unreadable.querySelectorAll("select"),notes];
 const claimHeading=document.createElement("h3");claimHeading.textContent="Explicit action-unit claims (if provided)";
 const claimNodes=[];
 for(const au of item.claimed_aus||[]){{const field="au_claim_support_"+au;const wrap=makeSelect(au+" · "+(item.au_names[au]||"action-unit code"),field,[["","Choose one"],["visible","Visible"],["not_visible","Not visible"],["cannot_assess","Cannot assess"]],item.presentation_id+"-"+field);claimNodes.push(wrap);controls.push(...wrap.querySelectorAll("select"))}}
 const saved=answers[item.presentation_id]||{{}};for(const control of controls)control.value=saved[control.dataset.field]||"";
 for(const control of controls){{const save=()=>{{answers[item.presentation_id]=answers[item.presentation_id]||{{}};answers[item.presentation_id][control.dataset.field]=control.value;try{{localStorage.setItem(storageKey,JSON.stringify(answers))}}catch(e){{}}}};control.addEventListener("change",()=>{{save();updateProgress()}});control.addEventListener("input",save)}}
 card.append(heading,img,rationale,category);if(claimNodes.length)card.append(claimHeading,...claimNodes);card.append(score,label,unreadable,notesLabel,notes);root.appendChild(card);
}}
updateProgress();
document.getElementById("next").onclick=()=>{{const item=packet.find(x=>!answers[x.presentation_id]?.evidence_support_score);document.getElementById(item?.presentation_id||packet[0]?.presentation_id)?.scrollIntoView({{behavior:"smooth",block:"start"}})}};
document.getElementById("download").onclick=()=>{{const fields=["rater_id","presentation_id","evidence_support_score","label_consistency","unreadable","unsupported_claim_notes","au_claim_support"];const quote=v=>'"'+String(v??"").replaceAll('"','""')+'"';const lines=[fields.join(","),...packet.map(item=>{{const answer=answers[item.presentation_id]||{{}};const claims=Object.fromEntries(Object.entries(answer).filter(([key])=>key.startsWith("au_claim_support_")));return [rater,item.presentation_id,answer.evidence_support_score||"",answer.label_consistency||"",answer.unreadable||"",answer.unsupported_claim_notes||"",JSON.stringify(claims)].map(quote).join(",")}})];const blob=new Blob([lines.join("\\r\\n")],{{type:"text/csv;charset=utf-8"}});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=rater+"-session-{session}-ratings.csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}};
</script></body></html>'''


def image_source(row: dict[str, Any], output_dir: Path, copy_images: bool, copied: dict[str, str]) -> str:
    source = Path(row["image_path"])
    if not copy_images:
        return source.as_uri()
    existing = copied.get(str(source))
    if existing:
        return existing
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    destination = output_dir / "assets" / f"{digest}{source.suffix.lower() or '.jpg'}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    relative = destination.relative_to(output_dir).as_posix()
    copied[str(source)] = relative
    return relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Combined study variants JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key", type=Path, help="Restricted condition key CSV")
    parser.add_argument("--max-pairs", type=int, default=120, help="Controlled reference/foil image pairs")
    parser.add_argument("--natural-images", type=int, default=60, help="Natural VLM outputs to rate separately")
    parser.add_argument("--raters", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--copy-images", action="store_true", help="Copy selected images into packet assets after confirming permission")
    parser.add_argument("--unpaired", action="store_true", help="Use unpaired legacy records only; exploratory Mini-1 use")
    args = parser.parse_args()
    if args.max_pairs < 1 or args.natural_images < 0 or args.raters < 1 or args.sessions < 1:
        parser.error("--max-pairs, --raters, and --sessions must be positive; --natural-images cannot be negative")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        if row.get("sample_id"):
            groups[str(row["sample_id"])].append(row)

    selected_pair_info: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    if args.unpaired:
        ids = source_stratified(sorted(groups), groups, args.max_pairs, args.seed)
        records = [row for sample_id in ids for row in groups[sample_id]]
        selected_images = set(ids)
    else:
        selected_pair_info = select_control_pairs(groups, args.max_pairs, args.seed)
        records = [row for _, _, reference, foil in selected_pair_info for row in (reference, foil)]
        selected_images = {sample_id for _, sample_id, _, _ in selected_pair_info}
        natural_candidates = [
            sample_id for sample_id, rows in groups.items()
            if sample_id not in selected_images and "natural_vlm" in conditions(rows)
        ]
        for sample_id in source_stratified(natural_candidates, groups, args.natural_images, args.seed + 43):
            records.append(conditions(groups[sample_id])["natural_vlm"])
            selected_images.add(sample_id)

    records = usable_records(records)
    if not records:
        raise ValueError("No selected records have both a readable image and non-empty rationale")
    if not args.unpaired:
        valid_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("pair_id"):
                valid_pairs[str(record["pair_id"])].append(record)
        complete_pair_ids = {
            pair_id for pair_id, rows in valid_pairs.items()
            if {row.get("condition") for row in rows} in ({"au_reference", "au_foil"}, {"label_reference", "label_foil"})
        }
        records = [record for record in records if not record.get("pair_id") or record["pair_id"] in complete_pair_ids]
    if not records:
        raise ValueError("No complete pairs remain after image/rationale validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    key_path = args.key or args.output_dir / "private" / "condition_key.csv"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists() or any(args.output_dir.glob("rater_*_session_*.html")):
        parser.error("Packet files or condition key already exist. Use a new output directory to preserve blinding.")

    copied: dict[str, str] = {}
    key_rows: list[dict[str, Any]] = []
    packet_paths: list[Path] = []
    for rater_index in range(1, args.raters + 1):
        rater_id = f"rater_{rater_index:02d}"
        randomized = nonadjacent_shuffle(records, args.seed + 1000 * rater_index)
        chunks = [randomized[index::args.sessions] for index in range(args.sessions)]
        for session_index, chunk in enumerate(chunks, 1):
            cards = []
            for position, row in enumerate(chunk, 1):
                presentation_id = f"R{rater_index:02d}-S{session_index:02d}-P{position:04d}"
                cards.append({
                    "presentation_id": presentation_id,
                    "image_src": image_source(row, args.output_dir, args.copy_images, copied),
                    "rationale": row["rationale"],
                    "claimed_aus": [str(au).upper() for au in row.get("claimed_aus", []) if str(au).strip()],
                    "claimed_emotions": [str(label) for label in row.get("claimed_emotions", []) if str(label).strip()],
                    "au_names": AU_NAMES,
                })
                key_rows.append({
                    "rater_id": rater_id, "presentation_id": presentation_id, "session": session_index,
                    "sample_id": row.get("sample_id", ""), "pair_id": row.get("pair_id", ""),
                    "condition": row.get("condition", "unknown"), "family": row.get("family", ""),
                    "source": row.get("source", ""), "generator_id": row.get("generator_id", ""),
                    "image_path": row["image_path"],
                    "claimed_aus": json.dumps(row.get("claimed_aus", []), ensure_ascii=False),
                    "claimed_emotions": json.dumps(row.get("claimed_emotions", []), ensure_ascii=False),
                    "changed_claim": json.dumps(row.get("changed_claim", {}), ensure_ascii=False),
                })
            packet_path = args.output_dir / f"{rater_id}_session_{session_index:02d}.html"
            packet_path.write_text(html_page(rater_id, session_index, cards), encoding="utf-8")
            packet_paths.append(packet_path)

    with key_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    arm_counts = defaultdict(int)
    for _, _, reference, _ in selected_pair_info:
        arm_counts[reference.get("condition", "unknown")] += 1
    metadata = {
        "schema_version": 2, "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "seed": args.seed, "raters": args.raters, "sessions_per_rater": args.sessions,
        "controlled_pairs": sum(arm_counts.values()), "controlled_pair_arms": dict(arm_counts),
        "natural_cards": sum(row.get("condition") == "natural_vlm" for row in records),
        "cards_per_rater": len(records), "copy_images": args.copy_images,
        "unpaired_exploratory_mode": args.unpaired, "condition_key": str(key_path.resolve()),
        "condition_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "packet_files": [str(path.resolve()) for path in packet_paths],
        "packet_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in packet_paths},
    }
    metadata_path = args.output_dir / "packet_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
