# Project proposal

## Working title

**Do Automated Judges Track Visual Evidence? Validating Explanation Scores for Facial Emotion Analysis**

This title is provisional. It must be updated to match what the experiments actually show.

## One-sentence recommendation

Build an evaluation paper, not another emotion classifier: measure whether LLM-based explanation scores for facial emotion models respond to evidence changes in a face-matched explanation pair, and whether they stay stable when the wording changes but the claims do not.

## Why this is the strongest candidate from the current search

Current work has moved beyond Grad-CAM demos and plain AU-to-text generation. Facial-R1 and TAG train AU-grounded reasoning; FACR explicitly tests AU counterfactual consistency; ReFER studies evidence-grounded video explanations and reports label-anchoring failures; XFER-AU evaluates saliency maps against AU-derived regions. Most directly, the late-2026 MFEA paper introduces a multi-cue benchmark and an LLM-as-judge score for reasoning quality. A new “we generate AU explanations” or “we compare Grad-CAM and SHAP” project would repeat existing contributions.

The defensible candidate gap is narrower: **evaluate whether automatic explanation-quality scores themselves are evidence-sensitive and resistant to wording/label shortcuts in visual facial emotion analysis.** The nearest studies provide the model/score to audit, while recent causal-fidelity work is mainly about building a reasoner or evaluating structured rationale fidelity in text. The project audits a scoring metric; it does not propose another causal counterfactual reasoner. This gap is plausible, not proven unique.

## Research questions

- **RQ1 — Evidence sensitivity:** For matched image/rationale pairs, do judges rank explanations that independent raters find visually supported higher than explanations they find unsupported?
- **RQ2 — Shortcut sensitivity:** At fixed semantic claims, how much do repetition/verbosity and label framing change a judge score?
- **RQ3 — Judge validity:** How well do open-weight judges agree with blinded human ratings of evidence support, and with released annotation fields?

## Hypotheses to preregister

- **H1:** A useful explanation judge assigns higher evidence-support scores to rationales that independent human raters find visually supported.
- **H2:** Style-only edits produce smaller score changes than evidence-changing edits.
All hypotheses may fail. Do not tune the prompt or choose the main score after examining held-out audit outcomes.

## Independent human evidence reference

The released AU lists are useful references, but an AU omitted from a record is not necessarily absent in the image. Before the confirmatory run, inspect annotation completeness by upstream source. For the strong publication version, have at least three trained raters independently judge specific visible-movement claims in matched natural/edit explanations (target 200 image pairs). Hide model identity, condition, released labels, and study hypothesis; report ordinal agreement and paired judgments. Ask about visible movement support, not inner emotion. Obtain any required institutional approval before collecting ratings. The implemented packet includes an overall support score, separate per-AU decisions when codes are available, and an optional note for unsupported claims.

If you cannot run this human arm, narrow the claim to a computational audit of score behavior against **recorded annotation agreement**. Do not say it establishes visual faithfulness or human alignment.

## Dataset and access plan

### Primary

**MFEA-Main:** 1,168 entries in the official annotation JSONL, with image path, source, emotion label(s), AU list, and demographic metadata. This is the closest released benchmark because it pairs face images with both cue and emotion annotations. Use only the fields needed for the study; do not model or report demographic groups unless a separate ethics-approved question is preregistered.

MFEA-Context is outside the implemented confirmatory protocol. Treat it as future work unless a separate, preregistered context experiment is added.

### Access and license gate

The authors publish annotation JSONL in their GitHub repository, but link images through Baidu Netdisk. The authors list many source datasets, including AffectNet, RAF-DB, CK+, BP4D, DISFA, FABA, and others. The GitHub code license does not replace each image dataset's license. Before using any image, record the originating dataset and access conditions, verify that research use and local inference are permitted, and verify whether copying data to another laptop is permitted. If that cannot be established, do not use the image archive. No unofficial mirror should be used.

### Scale

Start with an access/quality pilot of 50 seeded, source-stratified images and confirm every image has a readable path and required annotations. Exclude those IDs from the 200-image confirmatory sample. Predefine exclusions (missing/corrupt image, face not visible, unusable annotation) and report counts. Do not silently replace excluded items.

## Experimental design

### Unit of analysis

One image plus a matched set of candidate explanations. Every model/rationale variant for an image stays together in bootstrap resampling and statistical comparisons.

### Explanation conditions

Use the same face, same generation model, and same prompt family wherever possible. Produce and retain:

1. **Original rationale:** the open-weight vision-language model's response to a fixed request to describe visible facial movements, state a dataset emotion label, and cite supporting cues. This provides a natural-output subset for human and judge ratings.
2. **Annotation rendering:** a short, deterministic text rendering of the released AU and emotion annotations. This is a structured probe, not a human-written gold explanation.
3. **AU-claim edit:** minimally substitute one AU claim using a documented FACS description. Label this `annotation-inconsistent`, not “false,” unless the annotation protocol confirms exhaustive negatives.
4. **Emotion-label edit:** swap the stated label to one not recorded for that item. Again call it `annotation-inconsistent`; do not equate a released category with a person's true internal state.
5. **Style control:** repeat the rationale while keeping the same structured AU and emotion claims. This provides a simple verbosity/repetition test; it does not test paraphrase robustness.

Keep the condition names hidden from judge models and human raters. Store the exact intervention as structured metadata for analysis. Human-rate a stratified subset of original outputs and one matched AU- or label-edited version; image-matched annotation probes are a separate diagnostic experiment. The 23-image `mini-1` demo set is retained as an optional legacy baseline and never pooled into confirmatory analysis.

### Models and baselines

- **Generator:** one small open-weight vision-language model in the main run; a second, different model family as replication if compute allows. Freeze prompt, decoding settings, and model revision before scoring.
- **Judge A:** a local open-weight vision-language model given the image and candidate rationale, with a fixed rubric and a bounded numeric score.
- **Judge B:** a second local open-weight judge for cross-judge replication if available.
- **Direct annotation baseline:** deterministic match between extracted AU/emotion claims and the released annotation fields. Report AU and emotion channels separately. Call it annotation agreement, not causal faithfulness.
- **Style-only control:** scores for semantic-preserving variants. This is a required negative control.
- **Human reference:** three trained raters score a stratified subset for whether each stated visible cue is supported by the image, blinded to model identity and intervention. Use an institution-approved process where required. Without this step, do not claim that an automated judge aligns with people.

Do not use closed paid APIs in the required run. If authors' exact MFEA-CoT implementation requires a proprietary judge, report your open-weight implementation as an **MFEA-CoT-style replication**, not an exact reproduction.

## Measures

### Primary outcomes

- **Paired evidence sensitivity:** judge-score differences between the natural output and its AU edit, and between the natural output and its label edit; report image-paired bootstrap 95% intervals. Treat edits as probes, not verified false claims.
- **Human-pairwise ranking accuracy:** among human-rated original/edit pairs where mean human scores differ, the fraction where the judge orders the pair in the same direction; report human and judge ties separately.
- **Style sensitivity:** absolute score change between semantically equivalent style variants; compare with the evidence-edit score change.
- **Claim annotation-match:** precision, recall, and F1 for judge-extracted AU and emotion claims versus the released annotation fields. These are recorded-annotation proxies; do not infer true absence.
- **Judge-human agreement:** ordinal Krippendorff alpha among raters and Spearman rank correlation between condition-level judge and human means, with disagreement and image-cluster intervals reported.

### Secondary diagnostics

- Judge responses to annotation-only probes as a secondary diagnostic.
- Results by source dataset, rationale generator, and judge model, provided counts support stable estimates.
- Failure examples where a fluent unsupported rationale outranks a concise annotation-supported one.

Do not treat deletion/insertion scores or attention maps as ground truth. The main research claim is about evaluator behavior.

## Statistical and reproducibility rules

- Freeze a sample manifest and random seed before model scoring.
- Use paired comparisons at the image level. Bootstrap images (not individual text variants) for confidence intervals.
- Run the primary judge at least three stochastic generations or seeds if decoding has randomness; use deterministic decoding for the confirmatory run if appropriate.
- Keep a separate pilot subset for prompt debugging. Never use it in confirmatory estimates.
- Log model name/revision, quantization, prompt, decoding settings, software versions, device, date, and any manual exclusions.
- Save raw judge responses and parse failures. Report failures; do not drop them silently.
- Save pilot and confirmatory manifests; exclude pilot IDs from confirmatory analysis.
- Report all prespecified comparisons and negative outcomes. No unsupported “SOTA” claim is needed for this paper.

## Primary contribution claim, if supported

> We provide a controlled audit of whether open-weight automated explanation judges for facial emotion analysis respond to image-matched changes in recorded facial-cue evidence, and quantify their sensitivity to semantic-preserving style changes.

This wording claims an evaluation contribution. Do not claim to discover true emotions, establish causality, solve fairness, prove user trust, or outperform every FER paper.

## Role of the previous `mini-1` project

The earlier classifiers, CAM panels, heuristic AU features, and saved text explanations are preserved as an exploratory engineering baseline. `scripts/adapt_mini1_outputs.py` converts its result files into a common JSONL format so a local judge can score them. These outputs do not enter the confirmatory MFEA analysis: the old language model did not receive the image, the AU heuristics were not validated against FACS labels, and its 23-image demo set was curated rather than protocol-defined. Any future exact-image comparison needs a documented crosswalk, permitted image access, and a new preregistered analysis.

## What would make the paper publishable

The paper needs actual paired results, complete model/prompt disclosures, a substantial fraction of the available eligible primary data, strong controls, uncertainty intervals, and an honest novelty comparison. For a claim about visual evidence, the strong version needs blinded human ratings or an independently verified exhaustive annotation protocol. A positive result is not guaranteed. If the MFEA image archive is inaccessible under its component terms, or if the only result is a judge preferring literal annotation text over a paraphrase, pause and redesign before claiming novelty.

## Draft abstract template

Facial emotion analysis systems increasingly produce natural-language explanations, and recent benchmarks use model judges to assess explanation quality. Whether these automated scores track image-level evidence under controlled changes remains unclear. We audit [N] explanation sets on [dataset/version], comparing [number] open-weight judges against released [AU/emotion] annotations under matched evidence edits and meaning-preserving style edits. We measure [primary outcomes]. Results show [measured findings with uncertainty]. These findings [support/do not support] the use of automated scores as evidence-fidelity measures in this setting. We release [code/configs/aggregate results] subject to source dataset terms. Our study evaluates agreement with annotations and visible cues; it does not infer participants' internal emotional states.
