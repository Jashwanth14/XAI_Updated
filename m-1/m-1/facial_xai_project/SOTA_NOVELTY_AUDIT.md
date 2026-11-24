# State-of-the-art and novelty audit

**Search cutoff:** 11 November 2026  
**Scope:** facial-expression/facial-emotion explanation and evaluation; action-unit grounding; counterfactual/faithfulness tests; multimodal emotion reasoning; ambiguity/uncertainty; recent FER evaluation benchmarks.

This is a documented, broad search of primary publisher/proceedings pages, arXiv, and official project repositories. It is **not** a formal systematic review: no complete bibliographic export was available, search-engine coverage is incomplete, and new work can appear at any time. The novelty conclusion below is therefore provisional.

## Decision

The earlier proposal, **“calibration-aware explanation reliability under dataset shift,” is not the recommended main project**. The 2026 paper UAR already separates emotion ambiguity from distribution shift and uses that decomposition for routing in FER. A paper that merely combines calibration, domain shift, and saliency stability would be too close to active work.

The strongest narrower candidate from this search is:

> **Test whether vision-language explanation judges actually use the face when scoring facial-evidence explanations.**

This candidate is distinct from proposing another classifier or explanation generator. It audits the *measurement instrument* used to evaluate model explanations. The closest direct overlap is MFEA (2026), which introduces multi-cue facial emotion analysis and an LLM-as-a-judge CoT score. Its published protocol uses DeepSeek-V3.2 for contextual-cue and reasoning-trace judgments, so a free local judge is an **adaptation**, not an exact reproduction. The crucial experiment should test visual reliance directly: the judge must respond to a face/rationale pairing and to a controlled cue edit, while remaining stable under wording changes.

**Novelty confidence: moderate and conditional.** General MLLM-as-a-judge work already validates visual judges across diverse tasks and documents score, length, position, and self-preference biases. The contribution cannot be “the first validation of a multimodal judge.” The defensible claim is a face-specific, cue-level meta-evaluation with an explicit visual-reliance intervention and blinded human cue-support ratings. The search found adjacent work that could narrow or eliminate this gap, especially FACR, ReFER, MFEA, Emotion Rationale Verifier, EMO-CoT, XFER-AU, and general MLLM-as-a-judge benchmarks. Before submission, repeat the query set and citation chase, obtain the closest full papers, and revise the contribution if an overlapping evaluator-audit study is found.

## Closest work and what it means for this project

| Work | What it already contributes | Consequence for this project |
|---|---|---|
| **Facial-R1**, AAAI 2026 | Joint facial emotion recognition, AU recognition, and AU-based natural-language reasoning; FEA-20K; training with supervised fine-tuning/reward optimization. | Do not claim that adding rationale text, AU grounding, or joint emotion/AU tasks is novel. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/39906) · [official code](https://github.com/RobitsG/Facial-R1) |
| **TAG: Thinking with Action Unit Grounding**, 2026 preprint | AU-grounded reasoning traces and AU-aware optimization for FER. | Generic AU-grounded CoT is occupied. [arXiv](https://arxiv.org/abs/2602.18763) |
| **FACR: Faithful Action-unit Causal Reasoning**, June 2026 preprint | Counterfactual AU interventions align facial-emotion predictions and rationales to an AU→emotion graph; includes a textual rationale audit. | Do not claim first AU counterfactual faithfulness or first causal facial rationale method. The remaining distinction is to audit *judge scores* on matched controlled explanations, not to introduce another trained reasoner. [arXiv](https://arxiv.org/abs/2606.15779) |
| **ReFER**, Frontiers in AI 2026 (accepted October) | Video FER pipeline with AU sequences, relevant frames, and language explanations; tests label corruption and reports anchoring to supplied predictions. | Video evidence grounding and label anchoring have been directly studied. Keep the proposed study about evaluator validity and static image benchmark scores, not generic label anchoring. [Paper](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1984257/abstract) |
| **Beyond Action Units: Multi-cue Facial Emotion Analysis (MFEA)**, Pattern Recognition 2026 | Multi-cue reasoning (AUs, demographics, context), MFEA-Bench, and an MFEA-CoT score combining objective metrics with an LLM judge. Official annotations include 1,168 MFEA-Main and 503 MFEA-Context items. | This is the nearest paper and likely the best target to audit. Do not present MFEA's metric or dataset as your contribution. Test evaluator validity under controlled paired edits and style-only controls. [Publisher page](https://www.sciencedirect.com/science/article/pii/S0031320326004851) · [official repo/data README](https://github.com/syc1336/MFEA-Multi-cue-Facial-Emotion-Recoginition/) |
| **MLLM-as-a-Judge**, 2024 | A general benchmark of 11 multimodal judges over 14 visual tasks, with human judgments for scoring, pairwise comparison, and ranking; reports disagreement and verbosity, position, and ego biases. | A facial project cannot claim the first multimodal-judge validation. Use its lesson: include pairwise evaluation, randomized order, a verbosity control, and a no-image/face-swap visual-reliance ablation. [arXiv](https://arxiv.org/abs/2402.04788) |
| **XFER-AU**, IEEE Access, published online 29 Dec 2025 / in print 2026 | Compares saliency methods (including Grad-CAM, SHAP, LIME and RISE) against AU-derived regions with a weighted score. | A Grad-CAM-vs-AU-region comparison is not novel. The proposed study audits natural-language judge scores. [DOI](https://doi.org/10.1109/ACCESS.2025.3649062) |
| **Explaining Facial Expression Recognition**, AAMAS 2025 | Textual and visual AU explanations; user study with 280 participants. | Do not claim first textual/visual FER explanation or first user evaluation. [Proceedings PDF](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1511.pdf) |
| **Emotion-Coherent Reasoning via Emotional Rationale Verifier**, AAAI 2026 | Rationale/prediction consistency method, a learned verifier, and a human evaluation of conveyed emotion. | Explanation coherence and human assessment already exist. Separate label/rationale coherence from visible cue support and directly test whether the image changes a judge’s verdict. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/37184) |
| **EMO-CoT**, IEEE Transactions on Affective Computing, July 2026 | End-to-end multimodal emotion reasoning with a framework and explanation-quality evaluation across benchmarks. | Broad “new multimodal emotion reasoning + score” is already occupied. [DOI](https://doi.org/10.1109/TAFFC.2026.3713090) |
| **Right for the Right Reasons?**, EMNLP 2026 | Human-curated causal abstractions, structural-causal similarity, and counterfactual sensitivity for LLM emotion explanations on text/social data. | Causal structure and counterfactual sensitivity in emotion explanation are no longer new in text. The proposed study is specifically the validity of automatic visual FER explanation evaluators against face-cue annotations. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1382/) |
| **MFEA-CoT / LLM-as-judge evaluation** | Already combines an LLM-based reasoning-quality score with objective cue/emotion scores. | The target is to test the score's validity and robustness, not assert that it has none. The paper's exact judge appears to use a hosted model; the free run in this project should be labeled an open-weight MFEA-CoT-style replication unless the exact implementation and model are freely accessible. [MFEA paper](https://www.sciencedirect.com/science/article/pii/S0031320326004851) |
| **UAR: Interpretable Uncertainty Routing**, June 2026 preprint | Separates annotation ambiguity from distribution shift and evaluates FER routing. | This directly overlaps the old calibration/shift idea. Do not use that as the central novelty claim. [arXiv](https://arxiv.org/abs/2606.22725) |
| **UAF: Unified Ambiguity Signal**, AACL 2026 | Uses model ambiguity as a shared control signal for calibrated multimodal emotion recognition. | Uncertainty-aware emotion work is very active; avoid a broad “we add uncertainty to emotion classification” pitch. [ACL Anthology](https://aclanthology.org/2026.aacl-long.111/) |
| **EMONET-FACE**, NeurIPS 2025 | Synthetic faces with expert ratings over a fine-grained emotion taxonomy; emphasizes rating distributions and disagreement. | Annotator ambiguity and distributional labels are established evaluation topics. This project does not claim to solve ambiguity; it uses MFEA's AU annotations for an evaluator audit. [OpenReview PDF](https://openreview.net/pdf?id=BqVIt5Dxxh) · [official repo](https://github.com/laion-ai/emonet-face) |
| **Robustness of MLLMs in FER: Human-comparative study**, available online 24 Oct 2026 | Tests semantic conflict and information scarcity and compares model behavior with human patterns. | Do not make a generic robustness-to-conflict or human-comparison claim. Keep intervention conditions narrowly tied to explanation scoring. [Publisher page](https://www.sciencedirect.com/science/article/pii/S0031320326021400) |
| **Emotional Expressions Reconsidered**, Psychological Science in the Public Interest, 2019 | Reviews evidence limiting one-to-one inference of internal emotional states from facial movements. | Use careful terminology: dataset categories and visible actions, not ground-truth inner feelings. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6640856/) |

## Recommended study design after the literature audit

The project should be framed as a **multimodal meta-evaluation**, not an annotation-match benchmark. Its main experiment should use the same generated explanation under four controlled conditions:

1. **Matched image and explanation:** the original face and its image-conditioned explanation.
2. **Cue edit:** the same face with one stated movement claim rewritten coherently in both the structured claim field and the natural-language rationale. Do not leave the original cue phrase in the rationale.
3. **Face swap:** the same explanation paired with a source-matched but annotation-divergent face. Call this an annotation-divergent swap until human raters verify visible support.
4. **Style control:** a paraphrase or length change with the same claims.

For each item, compare both a vision-language judge and a text-only judge/no-image ablation. Randomize option order in a pairwise judge task and repeat the order reversal. A valid visual evaluator should show larger change for the cue edit and face swap than for the style control, and the vision-enabled configuration should outperform its no-image version against blinded human cue-support judgments. This directly addresses the central failure mode: a nominally visual judge could score plausible words while ignoring the face.

The primary human task should use plain-language movement descriptions where possible (for example, “lip corners pulled upward”), with AU codes as secondary structured metadata. Raters should assess visible cue support, not a person's inner emotional state. The test set must be locked before model scoring; all manipulated claims should be retained regardless of outcome.

## Proposed gap as a testable statement

The new candidate is worth pursuing only if the following statement survives a further citation search:

> Recent FEA work reports LLM-based reasoning scores and general MLLM-as-a-judge work studies visual evaluator behavior, but the retrieved work does not establish whether facial-explanation judges rely on the face itself: whether they distinguish matched face/rationale pairs from annotation-divergent face swaps and coherent cue edits, while resisting style-only changes and agreeing with blinded human cue-support ratings.

This is a statement about the search conducted, not a claim that every paper has been found. It is falsifiable: if an existing paper already performs this controlled judge-validation protocol on visual FEA with comparable expert cue annotations, the project must change.

## Required comparison set in the eventual paper

- MFEA-CoT as described by its authors; exact reproduction only if its prompt, model, and data setup are available under free terms.
- A fixed open-weight vision-language judge, plus the same judge with no image or a text-only judge.
- A second open-weight judge family for replication if compute allows.
- Coherent cue edits, face-swap controls, and randomized-order pairwise tests.
- A semantic-preserving style control and direct recorded-annotation match as secondary diagnostics.
- Blinded human visible-cue ratings as the primary reference for judge validity.
- FACR and ReFER for causal/counterfactual and label-corruption distinctions; XFER-AU for spatial-map evaluation; MFEA for the nearest benchmark comparison; MLLM-as-a-Judge for evaluator methodology.

## Search record

Search performed 11 Nov 2026 using the connected web search across primary publisher/proceedings pages, arXiv, and official project pages. Query groups included:

- `facial emotion explanation faithfulness benchmark human evaluation counterfactual metrics action unit`
- `facial emotion analysis LLM as a judge explanation quality evaluation MFEA-CoT human evaluation`
- `facial expression recognition explanation faithfulness counterfactual intervention action unit causal benchmark`
- `2026 facial emotion explanation benchmark evidence faithfulness rationale image AU LLM evaluation`
- `facial emotion reasoning explanation evaluation metrics visual evidence gold standard action unit`
- `facial emotion uncertainty natural language rationale annotator disagreement ambiguity`
- `counterfactual fairness context deconfounding facial expression recognition`

The final search before submission should include at least Google Scholar or Semantic Scholar, OpenAlex/Crossref, arXiv, ACL Anthology, CVF Open Access, AAAI, IEEE Xplore, and recent citation lists of MFEA, FACR, and ReFER. Record all exact query strings, result dates, inclusion/exclusion decisions, and access limits in `LITERATURE_SEARCH_LOG.csv`.

## Sources and access caveats

- The MFEA GitHub repo publishes the annotation JSONL and links image files via Baidu Netdisk. Its code repository license does not override source dataset licenses. Confirm permissions for every component dataset and any transfer to a second computer.
- The exact hosted judge used in a recent paper may not be runnable for free. Do not silently swap in a local model and claim exact reproduction.
- Do not infer that an AU absent from a sparse annotation list is definitely absent from the face. First inspect the annotation protocol. Until completeness is confirmed, metrics are **match to recorded annotations** and controlled edits are **annotation-inconsistent**, not objectively false.
