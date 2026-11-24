"""
Automated XAI Benchmark Evaluation Script
Evaluates and compares:
  - FC-MSCAM (FACS-Contrastive Multi-Scale CAM — Proposed)
  - Grad-ECLIP (Channel + Spatial Baseline)
  - Standard Grad-CAM (Coarse 7x7 Baseline)

Three Standard Metrics (CVPR / XAI Literature):
  1. Confidence Drop at Top-5% Masking (Drop@5% ↑) — Faithfulness
  2. FACS Energy-Based Pointing Game (EBPG_FACS ↑) — Anatomical Localization
  3. Neuro-Symbolic Alignment Score (alpha_NSA ↑) — AU-Attention Consistency
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention.grad_eclip import build_attention_generator, AU_TO_LANDMARK_KEYS
from src.attention.region_parser import RegionParser, FACIAL_REGIONS
from src.emotion.model import build_model
from src.face_detection.au_extractor import AUExtractor, LANDMARKS
from src.face_detection.detector import FaceDetector


def compute_drop_at_k(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    cam: np.ndarray,
    target_class: int,
    k_percent: float = 0.05,
    device: str = "cuda",
) -> float:
    """
    Confidence Drop at Top-K% Masking.
    Masks out the top-k% brightest pixels in the attention map and measures
    the drop in target class confidence.
    """
    with torch.no_grad():
        orig_logits = model(input_tensor.to(device))
        orig_conf = torch.softmax(orig_logits, dim=1)[0, target_class].item()

    # Resize CAM to input image resolution (224, 224)
    cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    threshold = np.percentile(cam_resized, 100.0 * (1.0 - k_percent))
    mask = (cam_resized >= threshold).astype(np.float32)

    # Apply mask (set masked pixels to 0 / dataset mean)
    mask_tensor = torch.from_numpy(1.0 - mask).unsqueeze(0).unsqueeze(0).to(device)
    masked_input = input_tensor.to(device) * mask_tensor

    with torch.no_grad():
        masked_logits = model(masked_input)
        masked_conf = torch.softmax(masked_logits, dim=1)[0, target_class].item()

    drop = max(0.0, orig_conf - masked_conf)
    return drop


def compute_ebpg_facs(
    cam: np.ndarray,
    landmarks: np.ndarray,
    active_aus: List[str],
    sigma: float = 0.08,
) -> float:
    """
    FACS Energy-Based Pointing Game (EBPG_FACS).
    Measures the ratio of heatmap energy inside the active FACS Action Unit regions.
    """
    if cam.sum() <= 1e-8:
        return 0.0

    H, W = 56, 56
    cam_grid = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    total_energy = float(cam_grid.sum())
    if total_energy <= 1e-8 or landmarks is None or len(landmarks) == 0:
        return 0.0

    # Build active AU mask
    active_keys = set()
    for au in active_aus:
        active_keys.update(AU_TO_LANDMARK_KEYS.get(au, []))

    if not active_keys:
        # Fallback to all core facial landmark anchors
        active_keys = set(LANDMARKS.keys())

    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, H, dtype=np.float32),
        np.linspace(0.0, 1.0, W, dtype=np.float32),
        indexing="ij",
    )

    au_mask = np.zeros((H, W), dtype=np.float32)
    two_sig_sq = 2.0 * (sigma ** 2)

    for key in active_keys:
        idx = LANDMARKS.get(key)
        if idx is None or idx >= len(landmarks):
            continue
        lx, ly = float(landmarks[idx, 0]), float(landmarks[idx, 1])
        dist_sq = (xs - lx) ** 2 + (ys - ly) ** 2
        kernel = np.exp(-dist_sq / two_sig_sq)
        au_mask = np.maximum(au_mask, kernel)

    # Binary or weighted FACS region
    binary_facs_zone = (au_mask >= 0.25).astype(np.float32)
    facs_energy = float((cam_grid * binary_facs_zone).sum())

    return min(1.0, facs_energy / max(total_energy, 1e-8))


def compute_nsa_score(
    attention_regions: List[str],
    active_aus: List[str],
) -> float:
    """
    Neuro-Symbolic Alignment Score (alpha_NSA).
    Measures how consistently top attention regions map to active FACS Action Units.
    """
    if not attention_regions and not active_aus:
        return 1.0
    if not attention_regions or not active_aus:
        return 0.0

    # Mapping between FACS AUs and RegionParser region names
    AU_TO_REGIONS = {
        "AU1": {"left_eyebrow", "right_eyebrow", "forehead"},
        "AU2": {"left_eyebrow", "right_eyebrow", "forehead"},
        "AU4": {"left_eyebrow", "right_eyebrow", "nose"},
        "AU5": {"left_eye", "right_eye"},
        "AU6": {"left_cheek", "right_cheek", "left_eye", "right_eye"},
        "AU12": {"mouth", "left_cheek", "right_cheek"},
        "AU14": {"mouth", "left_cheek", "right_cheek"},
        "AU15": {"mouth", "chin"},
        "AU17": {"chin", "mouth"},
        "AU24": {"mouth"},
        "AU25": {"mouth"},
        "AU26": {"mouth", "chin"},
    }

    expected_regions = set()
    for au in active_aus:
        expected_regions.update(AU_TO_REGIONS.get(au, set()))

    if not expected_regions:
        return 0.5

    actual_set = set(attention_regions)
    overlap = len(actual_set & expected_regions)
    union = len(actual_set | expected_regions)

    return overlap / max(union, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate XAI Attention Generators")
    parser.add_argument("--model", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "regnet_y_800mf", "resnet18", "efficientnet_b4"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="Test")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/xai_benchmark.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 65)
    print("  XAI BENCHMARK EVALUATION")
    print("=" * 65)
    print(f"  Model       : {args.model}")
    print(f"  Data Path   : {args.data_path}")
    print(f"  Device      : {device}")
    print(f"  Max Samples : {args.max_samples}")
    print("=" * 65 + "\n")

    # Load components
    detector = FaceDetector()
    au_extractor = AUExtractor()
    region_parser = RegionParser(threshold=0.25, top_k=3)

    # Load model
    ckpt = args.checkpoint or f"checkpoints/{args.model}_best.pth"
    model = build_model(args.model, num_classes=8, pretrained=not os.path.exists(ckpt))
    if os.path.exists(ckpt):
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"[Loaded] Checkpoint from {ckpt}")
    model = model.to(device).eval()

    # Build attention generators
    methods = ["fc_mscam", "grad_eclip", "grad_cam"]
    generators = {m: build_attention_generator(model, method=m, device=device) for m in methods}

    # Collect images
    extensions = ("*.png", "*.jpg", "*.jpeg")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(args.data_path, ext)))
        images.extend(glob.glob(os.path.join(args.data_path, "**", ext), recursive=True))

    images = sorted(list(set(images)))[:args.max_samples]
    if not images:
        print(f"No images found in {args.data_path}")
        return

    print(f"Auditing {len(images)} images across {methods}...\n")

    results = {m: {"drop_at_5": [], "ebpg_facs": [], "alpha_nsa": [], "time_ms": []} for m in methods}

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for idx, img_path in enumerate(images, 1):
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            face_result = detector.detect(img_bgr)
            if not face_result.face_found or face_result.face_crop_pil is None:
                continue

            input_tensor = preprocess(face_result.face_crop_pil).unsqueeze(0).to(device)

            # AU features
            if face_result.landmarks is not None:
                au_res = au_extractor.extract(face_result.landmarks)
                active_aus = au_res.active_aus
            else:
                active_aus = []

            # Model prediction
            with torch.no_grad():
                logits = model(input_tensor)
                target_cls = logits.argmax(dim=1).item()

            for m in methods:
                gen = generators[m]
                t0 = time.time()
                cam = gen.generate(
                    input_tensor,
                    target_class=target_cls,
                    landmarks=face_result.landmarks,
                    active_aus=active_aus,
                )
                dur = (time.time() - t0) * 1000.0

                drop = compute_drop_at_k(model, input_tensor, cam, target_cls, k_percent=0.05, device=device)
                ebpg = compute_ebpg_facs(cam, face_result.landmarks, active_aus)
                regions = [r.name for r in region_parser.parse(cam)]
                nsa = compute_nsa_score(regions, active_aus)

                results[m]["drop_at_5"].append(drop)
                results[m]["ebpg_facs"].append(ebpg)
                results[m]["alpha_nsa"].append(nsa)
                results[m]["time_ms"].append(dur)

            print(f"[{idx}/{len(images)}] {os.path.basename(img_path):<15} -> Done")
        except Exception as e:
            print(f"[{idx}/{len(images)}] Error processing {img_path}: {e}")

    # Summary Table
    print("\n" + "=" * 78)
    print("  QUANTITATIVE XAI BENCHMARK RESULTS")
    print("=" * 78)
    print(f"  {'Method':<16} | {'Drop@5% (↑)':<12} | {'EBPG_FACS (↑)':<14} | {'α_NSA (↑)':<12} | {'Latency (ms)':<12}")
    print("-" * 78)

    summary_out = {}
    for m in methods:
        drop_mean = np.mean(results[m]["drop_at_5"]) if results[m]["drop_at_5"] else 0.0
        ebpg_mean = np.mean(results[m]["ebpg_facs"]) if results[m]["ebpg_facs"] else 0.0
        nsa_mean = np.mean(results[m]["alpha_nsa"]) if results[m]["alpha_nsa"] else 0.0
        time_mean = np.mean(results[m]["time_ms"]) if results[m]["time_ms"] else 0.0

        summary_out[m] = {
            "drop_at_5_mean": float(drop_mean),
            "ebpg_facs_mean": float(ebpg_mean),
            "alpha_nsa_mean": float(nsa_mean),
            "latency_ms_mean": float(time_mean),
            "n_samples": len(results[m]["drop_at_5"]),
        }
        tag = " (Ours)" if m == "fc_mscam" else ""
        print(f"  {m + tag:<16} | {drop_mean*100:>10.2f}% | {ebpg_mean*100:>12.2f}% | {nsa_mean*100:>10.2f}% | {time_mean:>10.1f} ms")

    print("=" * 78)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\n[Saved] Detailed benchmark metrics written to {args.output}\n")


if __name__ == "__main__":
    main()
