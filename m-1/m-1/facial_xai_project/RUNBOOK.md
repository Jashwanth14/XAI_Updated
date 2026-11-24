# End-to-end runbook

This is the executable workflow for the evaluator-audit study in [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md). The core study does not need fine-tuning. It uses local open-weight inference; the scripts save per-record outputs and run metadata. Do not fill a paper with expected or invented results.

## 0. Access, ethics, and preregistration

1. Read the MFEA paper, official repository, and terms for every source image collection. Public annotation JSONL does not grant permission to reuse or redistribute the underlying images.
2. Verify that local inference and copying data to a friend's computer comply with the relevant source terms. If access terms are unclear, do not download or process those images.
3. Before human ratings, obtain any institutional approval required by your institution. Keep the pilot separate from the confirmatory sample.
4. Finalize [configs/study_v1.json](configs/study_v1.json) and freeze the primary outcomes, eligibility rules, and sample size in a dated public or institutional protocol before confirmatory scoring. The checked-in config is only a planning template, not a preregistration. The hypotheses can fail; report negative results.

## 1. Install the free local environment

```bash
cd /path/to/facial_xai_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-audit.txt
```

Python 3.10 or newer is required. Use a PyTorch build that fits the available laptop/GPU. Model weights are downloaded from Hugging Face the first time unless already cached. Review each model's license. The inference script uses a local model and does not call a paid hosted API. Use `--local-files-only` on a disconnected machine with cached weights. The current script uses `dtype="auto"` and no quantization; the 4B judge needs more memory than the 2B generator. If the machine cannot run the 4B checkpoint, use a smaller judge and report it; if it is also the generator checkpoint, label the run self-evaluation. No fine-tuning is required.

## 2. Prepare records, then select pilot and confirmatory samples

Generate all probe records; do not take the first 50 annotation rows as a pilot because file order can be biased.

```bash
python scripts/prepare_mfea_audit.py \
  --annotations data/MFEA-Main.jsonl \
  --image-root /path/to/lawfully-obtained/MFEA-images \
  --output runs/all_audit_items.jsonl
```

Select a reproducible source-stratified pilot. The manifest and its metadata save the IDs, seed, and input hash.

```bash
python scripts/select_study_subset.py \
  --input runs/all_audit_items.jsonl --n 50 --seed 20261111 \
  --require-existing-images \
  --output-items runs/pilot_items.jsonl \
  --output-manifest runs/pilot_manifest.csv
```

Check image access, file readability, annotation fields, and model/runtime constraints using the pilot. Exclude the pilot IDs from the confirmatory sample:

```bash
python scripts/select_study_subset.py \
  --input runs/all_audit_items.jsonl --n 200 --seed 20261112 \
  --require-existing-images \
  --exclude-manifest runs/pilot_manifest.csv \
  --output-items runs/confirmatory_items.jsonl \
  --output-manifest runs/confirmatory_manifest.csv
```

The subset script samples images proportionally within the chosen source field, keeps every probe for an image together, and records a manifest. If the eligible population is smaller than the requested number, it reports the actual selected count.

The resulting annotation-reference, AU-edit, label-edit, and style rows are controlled text probes. They describe released annotations, not verified visual truth. A non-listed AU or label is not automatically false.

## 3. Generate image-conditioned explanations

Use one small vision-language model as the generator. The script saves the exact prompt, model request, resolved model revision (when provided by Transformers), package versions, input hash, runtime settings, per-item raw output, and parse/inference errors beside the output file.

```bash
python scripts/run_qwen_vl.py \
  --mode generate \
  --input runs/confirmatory_items.jsonl \
  --output runs/generator_outputs.jsonl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --seed 20261111
```

For a repeatable confirmatory run, pass a pinned Hugging Face commit with `--revision`. Use `--resume` after an interruption; it skips successful outputs and checks the run metadata before appending. Do not reuse an output path for different model, input, prompt, seed, or decoding settings. The generator receives the image but does not receive the released annotation labels.

## 4. Create matched explanation variants

```bash
python scripts/make_generated_variants.py \
  --generated runs/generator_outputs.jsonl \
  --items runs/confirmatory_items.jsonl \
  --output runs/generated_variants.jsonl
```

This keeps each natural output with a minimally substituted AU claim, a label not listed in the record, and a repeated-rationale style control when eligible. Edits are annotation-inconsistent probes; human assessment is needed before calling a cue visually unsupported. The repetition condition tests verbosity sensitivity, not general paraphrase robustness.

## 5. Score with an independent judge

```bash
python scripts/run_qwen_vl.py \
  --mode judge \
  --input runs/generated_variants.jsonl \
  --output runs/generated_judge_scores.csv \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --seed 20261111
```

The judge sees one image and one explanation at a time. It does not receive condition names, released annotations, or paired alternatives. The primary numeric score measures visible-cue support; dataset-label consistency is a separate field. Preserve raw outputs and parse errors. If the same checkpoint generates and judges, label it self-evaluation. A different open-weight judge family is a useful replication if compute permits.

For a second judge, rerun the same input with a different model and a different output path. Keep the prompt, sample, and generation outputs fixed. Pass both judge CSVs to the summary scripts; judge IDs remain separate in the output.

The prepared annotation-only probes can be scored as a separate diagnostic by using `runs/confirmatory_items.jsonl` as the judge input. Do not merge these results with natural model explanations without clearly identifying the probe type.

## 6. Human reference arm

After approval, FACS training, and a locked confirmatory sample, create randomized offline packets:

```bash
python scripts/make_human_rating_packets.py \
  --input runs/generated_variants.jsonl \
  --output-dir runs/human_packets \
  --key runs/private/condition_key.csv \
  --max-images 200 --raters 3 --seed 20261111
```

Keep the condition key private and share only each rater's HTML file and approved local image access. Raters save their completed CSV from the page. The packet includes the original explanation and one counterbalanced AU- or label-edited version for each selected image. The key is created with restrictive file permissions when supported.

Move the downloaded response files to a protected run directory and analyze them with the key:

```bash
python scripts/analyze_human_ratings.py \
  --ratings runs/human_packets/rater_01-ratings.csv \
            runs/human_packets/rater_02-ratings.csv \
            runs/human_packets/rater_03-ratings.csv \
  --key runs/private/condition_key.csv \
  --judgments runs/generated_judge_scores.csv \
  --output-dir runs/human_summary
```

For multiple judge CSVs, list each after `--judgments`.

Without the human arm, omit claims about judge-human alignment or visually verified support. Annotation agreement alone is not evidence of visual faithfulness.

## 7. Summarize automated scores

```bash
python scripts/summarize_judgments.py \
  --judgments runs/generated_judge_scores.csv \
  --items runs/confirmatory_items.jsonl \
  --output-dir runs/summary
```

The output includes score distributions by condition, paired judge-score differences with image-level bootstrap intervals, and claim-to-recorded-annotation agreement proxies. It separates AU and emotion fields. Scores and matches do not establish inner emotional state or causal feature use.

## 8. Carry `mini-1` forward as an exploratory baseline

The previous work remains useful for a legacy baseline: its classifiers, visualizations, and saved explanations can be inspected by the same local judge. Normalize the saved `result.json` files using the adapter:

```bash
python scripts/adapt_mini1_outputs.py \
  --input-root /Users/abhi/Desktop/mini-1/outputs \
  --output runs/mini1_legacy_outputs.jsonl
```

Then score those outputs separately:

```bash
python scripts/run_qwen_vl.py \
  --mode judge \
  --input runs/mini1_legacy_outputs.jsonl \
  --output runs/mini1_legacy_judgments.csv \
  --model Qwen/Qwen3-VL-4B-Instruct
```

The adapter uses a saved `face_crop.jpg` when present, preserves the legacy prediction/AU fields, and writes an import report. The mini-1 language model was text-only and its AU values came from unvalidated landmark heuristics. Its 23 curated images and saved demonstrations are not a confirmatory benchmark, and must not be pooled with the MFEA study. Optional unpaired human packets can be made with `--unpaired`, but their results remain exploratory.

## 9. Paper and reproducibility

Use [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) only after analyzing real outputs. Archive scripts/configs, subset manifests, metadata JSON, model revisions, raw outputs, exclusions, and aggregate tables. Do not share source images unless terms explicitly permit it. Recheck the nearest-paper literature before submission and narrow the claim if a closer evaluator-validation paper appears.

## Interpretation limits

- A score measures the configured judge rubric, not a person's internal emotion.
- An AU omitted from an annotation is not necessarily absent from the image.
- A fluent rationale and a saliency map do not prove what a classifier used.
- Controlled edits test a particular intervention; they are not naturalistic explanations.
- No small study can support “best among all papers” or a universal leaderboard claim.
