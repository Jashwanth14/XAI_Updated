# Review of the previous `mini-1` project

Reviewed read-only on 11 November 2026. The source folder is `/Users/abhi/Desktop/mini-1`; this document and the updated implementation are in the writable workspace project.

## What is already there

- Four image classifiers are implemented: ResNet-18, EfficientNet-B4, RegNetY-800MF, and ConvNeXt-Tiny.
- Training, evaluation, face landmark extraction, a CAM/Grad-ECLIP module, a region parser, MediaPipe landmark geometry, a prompt builder, and a text-generation wrapper exist.
- The README reports validation accuracy of 74.06% for ConvNeXt-Tiny, 72.74% for RegNetY-800MF, 71.90% for ResNet-18, and 71.58% for EfficientNet-B4 on an AffectNet-8 split. It also contains two training-curve PNGs and illustrative predictions.
- The `Test` directory contains 23 images. It is not an independent, balanced, protocol-defined test set.
- There are 98 saved per-image `result.json` files from the test-folder demos. Comparing each prediction with the class directory used by the demo, 93 of 98 predictions match the folder label. These are curated demo examples, with different numbers of samples per model, so this is not a defensible benchmark accuracy.

## What the checked-in artifacts do not establish

The repository does not contain the claimed training dataset, checkpoint files, per-epoch JSON/CSV logs, split manifest, environment record, or aggregate held-out metrics. The README figures therefore cannot be independently reproduced from this folder, and the reported values are validation results, not verified test-set results. The 98 JSON demos do not provide a common image list or evaluation protocol. The recent Git commit message about “unseen data” is not itself a saved evaluation protocol or metric table.

Five of the 98 saved demo explanations say the prediction was “correctly” identified even though it disagrees with the class folder used for that demo. One saved example has an `anger` input path, a `sad` prediction, and prose calling `sad` correctly identified. The explainer therefore asserts correctness without receiving the ground-truth label. In that same output, the geometry heuristic activates many AUs at once, including AU12 (lip-corner pull) and AU15 (lip-corner depressor), then treats them as verified observations. This is not reliable AU ground truth.

## Main methodological problems to fix

1. **The text model is text-only.** `src/explainer/vlm_engine.py` loads `AutoModelForCausalLM` and tokenizes only the prompt. The `image` argument does not reach the language model. The result is a prompted verbalization of supplied features, not an image-grounded VLM explanation.
2. **The AU values are hand-built landmark heuristics.** Static distances between 2D landmarks do not implement validated FACS coding; landmark geometry alone is not sufficient to claim observed AU presence or intensity. The comments calling thresholds “calibrated” need source data, annotation targets, calibration procedure, and held-out validation before they can be used as scientific claims.
3. **The prompt is leading.** It says to explain why the predicted category is correct and includes expected emotion-to-AU features. That invites confirmation bias and fabricated support.
4. **Heatmaps are not proof of reasoning.** Grad-CAM/Grad-ECLIP shows a model sensitivity/attribution visualization. It does not establish that an AU caused an emotion prediction or that the prediction reflects a person's internal emotional state.
5. **The novelty claims are too broad.** “First” application, “clinical explanation,” “eliminates human-in-the-loop,” and “higher-quality heatmaps” need evidence and direct baselines. The current literature includes AU-based explanation evaluation and user studies.
6. **Reproducibility is incomplete.** Freeze exact data versions, authorized split IDs, preprocessing, seed, class mapping, pretrained weight source, model selection rule, and package/device versions. Do not mix validation data, hand-picked demo photos, and the held-out test set.

## How it carries into the current paper

Keep the classifiers, CAM panels, face crops, and saved `result.json` files as an exploratory legacy baseline. The current paper direction is an audit of automatic explanation judges, not the old calibration-under-shift proposal. `scripts/adapt_mini1_outputs.py` imports the saved JSON records into the common schema, and the local Qwen judge can score them separately. This makes the earlier work useful as a documented baseline while preserving its limits: its language model was text-only, its AU branch is heuristic, and its 23-image demo set is not confirmatory evidence. Do not pool it with the MFEA results or present the reported accuracy as a held-out benchmark result unless the original split, checkpoints, and logs are recovered and independently reproduced.

