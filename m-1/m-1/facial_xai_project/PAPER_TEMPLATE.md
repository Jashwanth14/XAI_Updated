# Do Automated Judges Track Visual Evidence? Validating Explanation Scores for Facial Emotion Analysis

> **Do not submit this template before replacing every bracketed field with measured, source-checked information. Do not invent results.**

## Abstract

Facial emotion analysis systems increasingly generate natural-language explanations, and recent benchmarks use model judges to score their reasoning quality. We test whether those scores respond to image-matched changes in visible-cue claims and remain stable under a repetition-only style control. On [dataset/version; N eligible images], we compare [N] open-weight judges and [N] blinded raters under [conditions]. We report [paired score differences, human-pairwise ranking accuracy, judge-human correlation, style sensitivity, annotation-match proxies; intervals]. We find [actual finding]. This is [supportive/not supportive/mixed] evidence that [specific, narrow conclusion]. The study evaluates a selected rubric and visible-cue judgments; it does not infer a person's internal emotional state.

## 1. Introduction

- Describe recent use of natural-language explanation scores in facial emotion analysis.
- State the evaluation problem: a score may respond to fluency or label framing rather than visual evidence.
- Present the nearest work, especially MFEA/MFEA-CoT, FACR, ReFER, XFER-AU, and text-based structural-fidelity work.
- State research questions and contributions using measured wording.
- Do not say “best,” “first ever,” or “all previous work” without a formal, reproducible systematic review.

## 2. Related work

### AU-grounded facial reasoning

[Compare Facial-R1, TAG, FACR, ReFER, and the exact overlap with this evaluation study.]

### Explanation evaluation in facial analysis

[Compare MFEA-CoT, XFER-AU, AAMAS 2025, Emotion Rationale Verifier, and EMO-CoT. Separate natural-language scoring from saliency-map scoring and from explanation/prediction consistency.]

### Emotion ambiguity and interpretation limits

[Discuss EMONET-FACE and annotation-distribution work. Use “dataset label” and “visible facial cue” language; avoid treating an expression label as a verified internal state.]

## 3. Method

### Data and licensing

Dataset version: [ ]  
Eligible images: [ ]  
Missing/corrupt/excluded items and reasons: [ ]  
Source dataset and license handling: [ ]  
Annotation completeness assumptions: [ ]

### Generators and judges

Generator models/revisions: [ ]  
Judge models/revisions: [ ]  
Exact prompts, decoding, quantization, and hardware: [appendix/repo link]

### Matched explanation conditions

[Describe natural image-conditioned rationale, matched AU and label edits, and the repetition-only style control. State that annotation-only probes are a separate diagnostic. Explain that edits are annotation-inconsistent probes and that absence from the released set is not proof of absence.]

### Scoring rubric and comparators

[State exact judge rubric, judge-extracted claim-to-recorded-annotation proxy, and blinded human rating task. Explain that these measure different properties.]

### Statistical analysis

[Paired image-level analysis; bootstrap method; random seeds; missing-output rule; correction for multiple comparisons if used.]

## 4. Results

### Dataset and model coverage

[Table: counts by source, condition, image availability, annotation completeness, and excluded rows.]

### Evidence sensitivity

[Table/figure with paired score effects, pairwise preference, and 95% intervals.]

### Style sensitivity

[Compare style-only score changes with evidence-edit changes.]

### Direct annotation agreement and human comparison

[Report AU and emotion matches separately. Report ordinal Krippendorff alpha, human paired ratings, judge-human rank correlation, and judge pairwise ordering against non-tied human pairs. If no human study was performed, say so and do not claim human alignment.]

### Error analysis

[Include positive and negative examples; preserve anonymous IDs and dataset terms. Explain why the judge scored them as it did.]

## 5. Discussion

- What the measured judge behavior supports.
- What the study cannot conclude about causal faithfulness, emotion truth, user trust, or safety.
- Effect of annotation completeness and model choice.
- Whether results generalize to video, other cultures, open-vocabulary emotion labels, or non-MFEA datasets.
- How a follow-up evaluator could be validated without overclaiming.

## 6. Conclusion

[One short paragraph with the strongest result and its exact limit.]

## Reproducibility statement

Code commit: [ ]  
Pilot and confirmatory manifests with input hashes: [ ]  
Model revisions and prompts: [ ]  
Raw output hashes/aggregate files: [ ]  
Random seeds/environment: [ ]

## Ethics and data statement

The task evaluates dataset categories and visible action-unit annotations; it does not determine a person's internal emotional state. The study uses [dataset components] under [terms]. Human evaluation [was/was not] conducted. [Institutional review determination or approved protocol details, if applicable.] Images and personal data are [not redistributed / redistributed only if the source terms allow].
