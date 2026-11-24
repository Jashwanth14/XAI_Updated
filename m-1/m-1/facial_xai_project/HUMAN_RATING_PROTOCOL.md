# Human rating protocol

This is a research protocol draft, not ethics approval. Obtain any required institutional review before recruiting raters or collecting responses. Check the source-image terms before preparing packets or sharing image access. The packet generator is offline and stores ratings in the rater's browser until the CSV is downloaded.

## Objective and unit

Rate whether specific visible facial-movement claims are supported by the pixels. Do not judge a person's true internal emotional state, personality, intent, health, or demographic traits. The unit is an image plus one explanation. For the confirmatory human comparison, each sampled image has its natural VLM explanation and one matched AU- or label-edited explanation; the edit type is hidden and assignments are approximately balanced.

Target 200 paired image sets and three independent raters per item, subject to workload and approval. That is about 400 explanation ratings per rater. If that workload is not feasible, set a smaller number before collecting responses and report it; do not reduce the sample after looking at scores.

## Packet creation and blinding

After the pilot is excluded and the confirmatory variants are frozen, run `scripts/make_human_rating_packets.py`. It creates a separate randomized offline HTML file for each rater and a restricted condition key. The rater view omits the model name, condition, released labels, and study hypothesis. It shows the image, explanation, the model-stated category as a separate claim, and any explicit AU codes that the explanation output supplied. It never shows released annotation values.

Keep `condition_key.csv` separate from the HTML packets and rating CSVs. Share each rater only their HTML file and local access to the approved images. Do not send the key to raters. Each rater should use their own packet and download their own CSV. Do not show matched pairs side by side.

The packet script can also create unpaired packets from `mini-1` via `--unpaired`; those ratings are exploratory and must be analyzed separately from the confirmatory MFEA pair study.

## Rating form

1. **Visible-cue support, 1–5:** 1 = no specific movement claim is verifiable; 3 = claims are mixed or ambiguous; 5 = all specific movement claims are clearly visible. `Cannot assess` is available for an image that is too small, obscured, or otherwise unreadable.
2. **Explicit AU claims:** for each AU code supplied with the rationale, mark visible, not visible, or cannot assess. This is a structured cue check, not a replacement for judging the full rationale.
3. **Emotion wording:** separately mark whether any stated category is visually consistent, inconsistent, ambiguous, or absent. This is about visible configuration only, not what the person feels.
4. **Optional note:** identify a movement claim that seems unsupported or ambiguous.

Train raters with a concise FACS reference and a practice set excluded from analysis. Explain how to use “cannot assess.” Freeze instructions before confirmatory ratings and record any changes.

## Analysis

Use `scripts/analyze_human_ratings.py` to join the downloaded responses to the restricted key. It reports condition-specific distributions, missing and cannot-assess rates, ordinal Krippendorff alpha, and paired original-minus-edit score differences with image-level bootstrap intervals. With `--judgments`, it also reports Spearman alignment between the human condition means and automated-judge scores, with images resampled as clusters.

Report the number of submitted, numeric, blank, and cannot-assess responses; score distributions for each rater; condition-specific sample counts; agreement; and all paired effects. Keep label-consistency responses separate from visible-cue support. Report disagreement and ties rather than forcing a single “gold” rating.

## Limits

These ratings are judgments about visible evidence under this protocol. They do not establish true emotion, cultural universality, causal influence on a classifier, or that a classifier used the explanation's stated cues. Edited rationales are controlled probes, not naturally occurring explanations.
