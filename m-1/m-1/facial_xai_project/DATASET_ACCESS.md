# Dataset access and handling

## Recommended primary dataset: MFEA-Bench

The official MFEA repository publishes annotation JSONL files in `annotation/` and links its images through Baidu Netdisk. The MFEA-Main annotation file contains image ID/path, source dataset, label(s), AU list, and demographic metadata. The MFEA-Context subset contains context descriptions and label(s). See the [official README](https://github.com/syc1336/MFEA-Multi-cue-Facial-Emotion-Recoginition/).

The paper's benchmark aggregates images from multiple upstream datasets, including FER2013, AffectNet, ExpW, RAF-DB, CK+, KDEF, NIMH-ChEFS, WSEFEP, HECO, DISFA, BP4D, RAF-AU, and FABA. **The GitHub code license does not supersede the source image terms.** Before downloading or using an image:

1. Identify its upstream source from the released manifest.
2. Read the current source dataset agreement and confirm your research use is allowed.
3. Confirm the storage location, model inference, and transfer to a friend's laptop are allowed.
4. Do not put images, credentials, or restricted labels in GitHub or this project folder.
5. Do not use unofficial mirrors when the official route is unavailable.

The annotation file can be inspected and processed without copying images. To run the image-based judge audit, however, each image must be lawfully available locally. The official repository links the image archive through Baidu Netdisk; access may be difficult outside its intended region. If terms or access cannot be established, do not proceed with that archive.

## Other datasets

- **AffectNet / AffectNet+**: the official site requests approval through a PI, lab manager, or professor for academic research; storage and sharing conditions apply. [Official page](https://mohammadmahoor.com/pages/databases/affectnetplus/)
- **RAF-DB**: access is controlled by the dataset provider. Use the official access page and follow its conditions. [Official page](http://www.whdeng.cn/RAF/model1.html)
- **CK+ / BP4D / DISFA / FABA**: access and redistribution terms differ by dataset. Do not treat them as open-licensed merely because papers cite them or an aggregate benchmark provides pointers.
- **FER-2013/FERPlus**: can support the legacy classifier pipeline, but the source labels do not provide the same AU annotations needed for the new primary experiment. Kaggle access is free of charge but account and upstream terms apply. Do not substitute it silently for MFEA.

## Data minimization

The study does not need demographic metadata for its primary question. The preparation script omits it. Keep all source IDs and license notes in a private manifest if needed, and publish only what upstream terms allow. Avoid sharing original face images, individual demographic records, or model outputs that reproduce restricted content.

## What this workspace did

No dataset images were downloaded. The literature audit inspected the official repository README and public annotation-file structure only. You must obtain images under the source terms before running the vision-language models.
