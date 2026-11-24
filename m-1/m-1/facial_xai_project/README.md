# Facial XAI research project

## Recommended paper direction

**Do automated explanation scores for facial emotion models reward evidence, or fluent wording?**

The previous plan—calibration and heatmap reliability under dataset shift—should not be the paper's main contribution. A June 2026 FER study, UAR, already separates annotation ambiguity from distribution shift and evaluates uncertainty-based routing. The more defensible next study is an evaluation audit of explanation scoring: test whether an automated judge gives higher scores to explanations that agree with released facial-cue annotations, and whether scores change when only style, wording, or label framing changes.

This is a **candidate research gap, not a claim of guaranteed novelty or state-of-the-art performance**. The closest work is the 2026 MFEA paper: it introduces multi-cue FER and an LLM-as-judge CoT score. Your proposed contribution would validate and stress-test such scores against controlled, image-matched evidence changes. See [SOTA_NOVELTY_AUDIT.md](SOTA_NOVELTY_AUDIT.md) and [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md).

## Important boundary

The study evaluates whether explanations agree with dataset annotations and visible evidence. It does not establish a person's inner emotional state. Call the targets **dataset emotion labels** and call absent annotation values **not recorded**, unless the source paper confirms that its annotations are exhaustive. Do not claim causal truth from a dataset match.

## Start here

1. Read [SOTA_NOVELTY_AUDIT.md](SOTA_NOVELTY_AUDIT.md) for the closest papers, overlap, and limits of the novelty claim.
2. Read [PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md) for the question, hypotheses, dataset, baselines, measures, and paper claim.
3. Check [DATASET_ACCESS.md](DATASET_ACCESS.md) and the MFEA release's underlying dataset terms before obtaining images. The public annotation files do not grant redistribution rights to the source images.
4. Follow [RUNBOOK.md](RUNBOOK.md) to create a random source-stratified pilot and confirmatory sample, run open-weight models locally, and analyze the outputs.
5. Use [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) only after experiments. Results are intentionally blank until measured.

## Project status and existing code

The `src/` classifier, calibration, and Grad-CAM code was written for the superseded uncertainty-under-shift idea. Keep it as a baseline/learning artifact; do not present it as the main research contribution. The earlier review of your `mini-1` folder is in [MINI1_REVIEW.md](MINI1_REVIEW.md). Its model outputs are now importable through `scripts/adapt_mini1_outputs.py` and can be scored as a separate exploratory baseline. Its reported accuracy values and curated demos are not sufficient evidence for a benchmark claim because the original splits, checkpoints, and aggregated evaluation artifacts were not present; the language model was text-only and the AU detector used unvalidated geometry heuristics.

The new study is evaluation-first. It does not require fine-tuning to answer the core question. A small open vision-language model can produce candidate explanations; a separate open-weight judge scores them. The project now includes deterministic source-stratified sampling, resumable inference journals with model/run metadata, blinded offline human-rating packets, and analysis for ordinal agreement and judge-human alignment. Fine-tuning is an optional follow-up only if the audit identifies a repeatable failure mode.

## Free-resource plan

- Python and the scripts in this repository are free.
- Use local open-weight vision-language models and a local judge; no paid inference API is required.
- MFEA annotations are in the authors' public repository. The image archive is linked through Baidu Netdisk and aggregates images from other datasets. Obtain/use every image only under its original terms. If that route is inaccessible or the terms do not permit your use, stop before training and use an approved dataset; do not use unofficial mirrors.
- GPU time is optional for the scoring audit but useful for local VLM inference. A friend's computer is fine only if the applicable dataset terms allow copying the files there.

## Repository map

- `SOTA_NOVELTY_AUDIT.md`: literature comparison through 11 Nov 2026.
- `PROJECT_PROPOSAL.md`: final candidate and claims.
- `configs/study_v1.json`: machine-readable planning configuration; it is not a preregistration until finalized and archived.
- `RUNBOOK.md`: end-to-end sequence, sample selection, baseline use, and run commands.
- `HUMAN_RATING_PROTOCOL.md`: blinded visible-cue task and human-analysis protocol.
- `scripts/select_study_subset.py`: source-stratified, seeded image sampling with manifests and pilot exclusion.
- `scripts/prepare_mfea_audit.py`: constructs matched annotation-reference and controlled annotation-inconsistent probes from MFEA JSONL.
- `scripts/make_generated_variants.py`: builds matched AU, label, and repetition-only probes from local VLM outputs.
- `scripts/run_qwen_vl.py`: local Qwen3-VL generation and blind scoring with resumable per-record output and metadata.
- `scripts/make_human_rating_packets.py`: makes randomized, offline HTML packets and a restricted condition key.
- `scripts/analyze_human_ratings.py`: reports rating summaries, ordinal agreement, paired effects, and optional judge-human alignment.
- `scripts/adapt_mini1_outputs.py`: normalizes legacy `mini-1` result files without promoting them to confirmatory evidence.
- `scripts/summarize_judgments.py`: summarizes condition scores, paired judge-score sensitivity, and annotation agreement proxies.
- `requirements-audit.txt`: packages for the new audit scripts and optional local VLM.
- `DATASET_ACCESS.md`: dataset access and license notes.
- `PAPER_TEMPLATE.md`: fill only with real outputs.
- `src/`: legacy FER classifier/calibration/Grad-CAM work from the earlier proposal.

## What counts as a successful research result

The project is useful even if your initial hypothesis is false. A credible paper requires a fixed protocol, exact model/version records, full per-example outputs, paired statistical estimates, failure cases, and a candid account of annotation limits. This code improves reproducibility; it cannot guarantee novelty, acceptance, or that results will outperform prior work. Do not claim that it is “best among all papers” or invent improvement numbers. If results show that current judges are already robust, report that result and narrow the contribution to a benchmark audit.
