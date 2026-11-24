# Start here: from empty folder to paper submission

## What you are building

Your paper asks one narrow, useful question:

> When an automated vision-language judge gives a high score to a facial-expression explanation, does that score track visible facial-cue evidence or wording shortcuts?

This is an **evaluation paper**. You are not trying to beat emotion-classification accuracy with a new network. The main experiment uses an image-conditioned open model to produce explanations, a separate open model to score them, controlled claim edits, and blinded human ratings of a subset.

Do not begin by fine-tuning. Fine-tuning is only a follow-up if the audit reveals a specific repeatable failure that your current protocol cannot explain.

## The four rules that protect the paper

1. Use only images you are permitted to process and share with the computer that runs inference.
2. Never use pilot images in the final results.
3. Freeze prompts, model revisions, sample manifests, exclusions, and metrics before the final run.
4. Do not claim that a facial image proves someone's internal emotion. Write about visible facial cues and dataset labels.

## Phase 1 — Read the project before running anything

Read these in this order:

1. [README.md](README.md)
2. [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md)
3. [SOTA_NOVELTY_AUDIT.md](SOTA_NOVELTY_AUDIT.md)
4. [DATASET_ACCESS.md](DATASET_ACCESS.md)
5. [HUMAN_RATING_PROTOCOL.md](HUMAN_RATING_PROTOCOL.md)
6. [RUNBOOK.md](RUNBOOK.md)

Your working title is:

**Do Automated Judges Track Visual Evidence? Validating Explanation Scores for Facial Emotion Analysis**

Do not call it “the best FER model,” “first XAI FER,” or “clinical emotion diagnosis.” The contribution is a controlled validation of an explanation-evaluation metric.

## Phase 2 — Prepare the laptop

Use a laptop with Python 3.10+ and preferably an NVIDIA GPU. The project can run without a paid API. Your friend's laptop is acceptable only after the image-dataset terms allow you to copy the images there.

```bash
cd /path/to/facial_xai_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-audit.txt
```

Create these local folders; they are ignored by Git:

```bash
mkdir -p data runs/private
```

Keep a simple local note containing:

- Computer/GPU name and memory
- Python, PyTorch, Transformers versions
- Exact model names and revisions
- Dates of every run
- Any file that could not be read

The inference script also saves run metadata automatically.

## Phase 3 — Get data legally before processing it

The intended dataset is MFEA-Main because it links face images with recorded action units and expression labels. Get its annotation JSONL and images only through the official route and under the relevant source terms.

Before running any model, make a private license log with one row for each source dataset represented in the sample:

| Field | Record |
| --- | --- |
| Source dataset | e.g. the value in MFEA's `source` field |
| Official access page | URL |
| Date checked | Date |
| Research/inference allowed? | Yes / no / unclear |
| Transfer to friend's laptop allowed? | Yes / no / unclear |
| Evidence | License page, approval email, or terms reference |

If any selected image's terms are unclear, exclude it before the confirmatory manifest. Do not use a scraped image mirror. If MFEA images cannot be used lawfully, stop this version of the study and choose a new, fully documented dataset; do not silently substitute FER2013 because it lacks the AU labels needed for this question.

## Phase 4 — Build the complete candidate pool

Put the official annotation file at `data/MFEA-Main.jsonl`. Point `--image-root` to the lawfully obtained image root.

```bash
python scripts/prepare_mfea_audit.py \
  --annotations data/MFEA-Main.jsonl \
  --image-root /path/to/MFEA-images \
  --output runs/all_audit_items.jsonl
```

Check the output metadata file. It must show a nonzero number of samples and the expected source fields. Do not select the final sample until image paths are readable.

## Phase 5 — Run a 50-image development pilot

The pilot is for finding access problems, unreadable images, malformed outputs, and hardware limits. It is not evidence for your paper.

```bash
python scripts/select_study_subset.py \
  --input runs/all_audit_items.jsonl \
  --n 50 --seed 20261111 --require-existing-images \
  --output-items runs/pilot_items.jsonl \
  --output-manifest runs/pilot_manifest.csv
```

Run generation, variant construction, and judging on `pilot_items.jsonl` using the same commands in [RUNBOOK.md](RUNBOOK.md). Inspect at least:

- missing-image count;
- invalid JSON outputs;
- whether the VLM produces usable cue claims;
- GPU memory and runtime;
- whether the explanation wording visibly conflicts with the edited claim panel;
- whether source-specific data access is recorded.

Fix code, prompts, or protocol only during this stage. Keep the pilot output folder; label it `pilot`, and never merge it into final tables.

## Phase 6 — Freeze the final protocol

After the pilot, finalize [configs/study_v1.json](configs/study_v1.json) and save a dated copy outside the results folder. Record:

- The exact model revision for generator and judge
- Whether the judge is independent from the generator
- Eligibility and exclusion rules
- The confirmatory image count
- Primary outcomes and bootstrap seed
- The human-rater task and institutional ethics/approval determination

Do not change these after looking at confirmatory results. If a change is necessary, write down the reason and call the new run exploratory.

## Phase 7 — Select the final image set

Create a source-stratified, non-overlapping final set:

```bash
python scripts/select_study_subset.py \
  --input runs/all_audit_items.jsonl \
  --n 200 --seed 20261112 --require-existing-images \
  --exclude-manifest runs/pilot_manifest.csv \
  --output-items runs/confirmatory_items.jsonl \
  --output-manifest runs/confirmatory_manifest.csv
```

The model audit can use 200 images. For the human arm, use a predeclared source-stratified subset that raters can complete without fatigue. A practical strong target is 100–120 image pairs, each with the natural explanation and one edited explanation. Each rater then completes 200–240 cards, ideally split into two sessions. Report the actual count and all incomplete responses.

## Phase 8 — Generate the natural explanations

```bash
python scripts/run_qwen_vl.py \
  --mode generate \
  --input runs/confirmatory_items.jsonl \
  --output runs/generator_outputs.jsonl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --seed 20261111
```

For the final paper, add `--revision <pinned-commit>` after you identify the exact released model revision. If a run stops, rerun exactly the same command with `--resume`. Do not overwrite an output file with a different model or prompt.

## Phase 9 — Make controlled variants

```bash
python scripts/make_generated_variants.py \
  --generated runs/generator_outputs.jsonl \
  --items runs/confirmatory_items.jsonl \
  --output runs/generated_variants.jsonl
```

Before the final study, verify one important design property: each AU/label intervention must change the claim that judges and raters actually read, without leaving contradictory original cue wording beside it. The natural free-text rationale arm and the controlled structured-claim arm should be identified separately in the paper. This prevents a score drop caused merely by a self-contradictory explanation.

## Phase 10 — Run independent automated judges

```bash
python scripts/run_qwen_vl.py \
  --mode judge \
  --input runs/generated_variants.jsonl \
  --output runs/judge_a_scores.csv \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --seed 20261111
```

If the hardware allows it, run a second judge from a different model family with a new output file. Keep the images, variants, rubric, and decoding settings fixed. This is much more convincing than tuning one judge until it gives the expected result.

## Phase 11 — Run the blinded human-rating study

Do this only after the confirmatory images, explanations, variants, and protocol are locked. Obtain the required institutional review or determination first.

Create packets with a predeclared human subset size, for example 120:

```bash
python scripts/make_human_rating_packets.py \
  --input runs/generated_variants.jsonl \
  --output-dir runs/human_packets \
  --key runs/private/condition_key.csv \
  --max-images 120 --raters 3 --seed 20261111
```

Keep `condition_key.csv` private. Each rater receives only their own randomized HTML packet and legal local access to images. They should not see the model name, condition name, released AU list, or the paired alternative.

## Phase 12 — Analyze everything

Analyze automated judges:

```bash
python scripts/summarize_judgments.py \
  --judgments runs/judge_a_scores.csv \
  --items runs/confirmatory_items.jsonl \
  --output-dir runs/judge_summary
```

Analyze human ratings and judge alignment:

```bash
python scripts/analyze_human_ratings.py \
  --ratings runs/human_packets/rater_01-ratings.csv \
            runs/human_packets/rater_02-ratings.csv \
            runs/human_packets/rater_03-ratings.csv \
  --key runs/private/condition_key.csv \
  --judgments runs/judge_a_scores.csv \
  --output-dir runs/human_summary
```

Report every parse failure, unavailable image, rater `cannot assess`, human tie, and judge tie. Bootstrap by image, not by individual text cards.

## Phase 13 — Use your old `mini-1` work correctly

Do not throw it away. Import it as an exploratory legacy baseline:

```bash
python scripts/adapt_mini1_outputs.py \
  --input-root /Users/abhi/Desktop/mini-1/outputs \
  --output runs/mini1_legacy_outputs.jsonl
```

You may score it with the same judge and include a clearly marked qualitative appendix. Do not put its reported accuracy, landmark-derived AUs, or text-only explanations in the main result table unless you reconstruct the old data split, checkpoints, and evaluation logs.

## Phase 14 — Write the paper only from measured outputs

Use [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md). Your main result table should contain:

| Result | What it answers |
| --- | --- |
| Natural rationale ↔ human support correlation | Does the judge agree with blinded human evidence ratings? |
| Natural vs AU-claim edit | Does the judge respond when a visible-cue claim changes? |
| Natural vs label edit | Does a category label change the score independently of visual cues? |
| Original vs style control | Does wording length/repetition move the score when claims stay fixed? |
| Rater agreement and cannot-assess rate | How reliable was the human reference? |
| Failure cases | Where does the automated judge fail? |

Your conclusion must be conditional on the results. A strong positive result supports using the tested judge as a partial evaluation signal. A negative result is also publishable if it shows specific shortcut or reliability failures with proper controls.

## Submission checklist

Before submission, you must have:

- A current literature search and a revised novelty statement
- A frozen protocol and manifest hashes
- Exact model revisions, prompts, decoding parameters, environment metadata, and random seeds
- Full output and failure logs
- Data license/access records
- Human-study approval/determination and rater instructions, if human data were collected
- Tables and figures generated from the scripts, not manually invented
- A release plan that excludes restricted face images and private rating data

When every item is present, the study is a defensible research project. The result may be positive or negative; the strength comes from the controlled design and transparent evidence.
