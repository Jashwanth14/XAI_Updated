from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import CLASS_NAMES, FER2013
from .calibration import calibration_metrics, fit_temperature
from .model import FERClassifier, GradCAM


def device_for_run():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def tensor_to_gray(x):
    return (x.detach().cpu().squeeze().numpy() * 0.5 + 0.5).clip(0, 1)


def deletion_curve(model, x, target, cam, device, steps=10):
    """Progressively gray-mask top-ranked square patches; returns normalized probabilities."""
    target = torch.as_tensor(target, dtype=torch.long, device=device).reshape(1)
    h, w = cam.shape[-2:]
    grid = 8
    importance = torch.nn.functional.adaptive_avg_pool2d(cam, (grid, grid)).flatten()
    order = importance.argsort(descending=True)
    x0 = x.clone()
    with torch.no_grad():
        baseline = torch.softmax(model(x0), dim=1)[0, target].item()
        probs = [baseline]
        for n in range(1, steps + 1):
            count = max(1, int((n / steps) * grid * grid))
            masked = x0.clone()
            for idx in order[:count].tolist():
                row, col = divmod(idx, grid)
                y0, y1 = round(row * h / grid), round((row + 1) * h / grid)
                x0c, x1c = round(col * w / grid), round((col + 1) * w / grid)
                masked[:, :, y0:y1, x0c:x1c] = 0.0  # gray in normalized input space
            probs.append(torch.softmax(model(masked), dim=1)[0, target].item())
    curve = np.asarray(probs, dtype=float)
    return float(np.mean(curve[0] - curve[1:])), curve


def save_overlay(image, cam, path):
    base = tensor_to_gray(image)
    heat = cam.detach().cpu().squeeze().numpy()
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    ax.imshow(heat, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def bootstrap_ci(frame, metric, replicates=1000, seed=1701):
    """Percentile 95% CI resampling test images (paired rows), not pixels."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    estimates = []
    for _ in range(replicates):
        sample = frame.iloc[rng.integers(0, n, size=n)]
        if metric == "accuracy":
            value = sample.correct.mean()
        elif metric == "macro_f1":
            value = f1_score(sample.true_label, sample.predicted_label, average="macro", zero_division=0)
        else:
            value = sample[metric].mean()
        estimates.append(float(value))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _probabilities_for_metrics(model, dataset, device, temperature=None, limit=0):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    chunks = []
    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            if temperature is not None:
                logits = logits / temperature
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
            seen += len(x)
            if limit and seen >= limit:
                break
    probabilities = np.concatenate(chunks, axis=0)
    return probabilities[:limit] if limit else probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="FER CSV or ImageFolder root containing train/ and test/")
    parser.add_argument("--calibration-data", default=None,
                        help="CSV or ImageFolder used to fit temperature on validation only; defaults to --csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 audits the full held-out test split")
    args = parser.parse_args()
    random.seed(1701)
    np.random.seed(1701)
    torch.manual_seed(1701)
    out = Path(args.output)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    device = device_for_run()
    ds = FER2013(args.csv, "test", train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "resnet18")
    model = FERClassifier(architecture, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    calibration_path = args.calibration_data or args.csv
    calibration_ds = FER2013(calibration_path, "val", train=False)
    calibration_dl = DataLoader(calibration_ds, batch_size=64, shuffle=False)
    calibration_logits, calibration_labels = [], []
    with torch.no_grad():
        for x, y in calibration_dl:
            calibration_logits.append(model(x.to(device)).cpu())
            calibration_labels.append(y.cpu())
    temperature = fit_temperature(torch.cat(calibration_logits), torch.cat(calibration_labels))

    gradcam = GradCAM(model)
    # Fixed perturbation definitions; occlusion is a stress test, not asserted label-preserving.
    perturbations = {
        "brightness": transforms.ColorJitter(brightness=0.08),
        "contrast": transforms.ColorJitter(contrast=0.08),
        "translation": transforms.RandomAffine(degrees=0, translate=(0.025, 0.025), scale=(0.98, 1.02)),
    }
    tensorize = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    records = []
    deletion_values, random_deletion_values = [], []
    total_auditing = min(len(ds), args.max_images) if args.max_images else len(ds)
    print(f"\n  Auditing {total_auditing:,} test images with Grad-CAM and perturbation analysis...")
    audit_start = time.time()
    for i, (image, label) in enumerate(dl):
        if args.max_images and i >= args.max_images:
            break
        image, label = image.to(device), int(label.item())
        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(probs.argmax().item())
            confidence = float(probs[pred].item())
            calibrated_probs = torch.softmax(logits / temperature, dim=1)[0]
            calibrated_confidence = float(calibrated_probs[pred].item())
        cam = gradcam(image, torch.tensor([pred], device=device))
        aopc, curve = deletion_curve(model, image, pred, cam, device)
        deletion_values.append(aopc)
        # Random-order deletion control; identical grid and target.
        random_cam = torch.rand_like(cam)
        random_aopc, _ = deletion_curve(model, image, pred, random_cam, device)
        random_deletion_values.append(random_aopc)

        raw = Image.fromarray((tensor_to_gray(image) * 255).astype(np.uint8), mode="L")
        perturb_result = {}
        for name, perturb in perturbations.items():
            random.seed(1701 + i)
            torch.manual_seed(1701 + i)
            pert_x = tensorize(perturb(raw)).unsqueeze(0).to(device)
            pert_cam = gradcam(pert_x, torch.tensor([pred], device=device))
            original_map = cam.detach().cpu().squeeze().numpy().ravel()
            changed_map = pert_cam.detach().cpu().squeeze().numpy().ravel()
            rho = spearmanr(original_map, changed_map).statistic
            if not np.isfinite(rho):
                rho = 0.0
            with torch.no_grad():
                pert_pred = int(model(pert_x).argmax(1).item())
            perturb_result[f"stability_{name}"] = float(rho)
            perturb_result[f"prediction_stable_{name}"] = pert_pred == pred

        # A central gray patch is an occlusion stress test; it may alter the expression cue.
        occluded = raw.copy()
        draw = ImageDraw.Draw(occluded)
        width, height = occluded.size
        x0, y0 = int(width * 0.40), int(height * 0.38)
        x1, y1 = int(width * 0.60), int(height * 0.62)
        draw.rectangle((x0, y0, x1, y1), fill=127)
        occlusion_x = tensorize(occluded).unsqueeze(0).to(device)
        occlusion_cam = gradcam(occlusion_x, torch.tensor([pred], device=device))
        occlusion_rho = spearmanr(
            cam.detach().cpu().squeeze().numpy().ravel(),
            occlusion_cam.detach().cpu().squeeze().numpy().ravel(),
        ).statistic
        if not np.isfinite(occlusion_rho):
            occlusion_rho = 0.0
        with torch.no_grad():
            occlusion_pred = int(model(occlusion_x).argmax(1).item())
        perturb_result["stability_occlusion"] = float(occlusion_rho)
        perturb_result["prediction_stable_occlusion"] = occlusion_pred == pred
        records.append({
            "index": i, "true_label": label, "predicted_label": pred,
            "true_name": CLASS_NAMES[label], "predicted_name": CLASS_NAMES[pred],
            "correct": pred == label, "confidence": confidence,
            "calibrated_confidence": calibrated_confidence,
            **perturb_result,
            "deletion_aopc": aopc, "random_deletion_aopc": random_aopc,
        })
        if i < 30:
            save_overlay(image, cam, out / "examples" / f"{i:04d}_{CLASS_NAMES[pred]}_true-{CLASS_NAMES[label]}.png")

        # Progress reporting
        if (i + 1) % 10 == 0 or (i + 1) == total_auditing:
            elapsed = time.time() - audit_start
            rate = (i + 1) / max(elapsed, 1e-5)
            eta_s = (total_auditing - (i + 1)) / max(rate, 1e-5)
            pct = 100 * (i + 1) / total_auditing
            eta_m, eta_sec = divmod(int(eta_s), 60)
            print(f"\r  Auditing: [{i+1:>4}/{total_auditing}] ({pct:>5.1f}%) | "
                  f"{rate:>4.1f} img/s | ETA: {eta_m:02d}m {eta_sec:02d}s ", end="", flush=True)

    print()
    gradcam.close()
    frame = pd.DataFrame(records)
    frame.to_csv(out / "predictions.csv", index=False)
    summaries = {
        "n": len(frame), "accuracy": float(accuracy_score(frame.true_label, frame.predicted_label)),
        "macro_f1": float(f1_score(frame.true_label, frame.predicted_label, average="macro", zero_division=0)),
        "mean_confidence": float(frame.confidence.mean()),
        "temperature_fit_on_validation": temperature,
        "raw_probability_metrics": calibration_metrics(
            _probabilities_for_metrics(model, ds, device, temperature=None, limit=args.max_images),
            frame.true_label.to_numpy(),
        ),
        "temperature_scaled_probability_metrics": calibration_metrics(
            _probabilities_for_metrics(model, ds, device, temperature=temperature, limit=args.max_images),
            frame.true_label.to_numpy(),
        ),
        "mean_deletion_aopc": float(np.mean(deletion_values)),
        "mean_random_deletion_aopc": float(np.mean(random_deletion_values)),
        "confidence_bins": {},
        "notes": "Grad-CAM and deletion are diagnostics, not causal proof. FER-2013 has no subject IDs.",
    }
    summaries["perturbation_summary"] = {}
    for name in ["brightness", "contrast", "translation", "occlusion"]:
        stability_col = f"stability_{name}"
        prediction_col = f"prediction_stable_{name}"
        summaries["perturbation_summary"][name] = {
            "mean_map_spearman": float(frame[stability_col].mean()),
            "prediction_consistency": float(frame[prediction_col].mean()),
            "correct_only_map_spearman": float(frame.loc[frame.correct, stability_col].mean()) if frame.correct.any() else None,
            "incorrect_only_map_spearman": float(frame.loc[~frame.correct, stability_col].mean()) if (~frame.correct).any() else None,
        }
    summaries["bootstrap_95pct_ci_by_image"] = {
        "accuracy": bootstrap_ci(frame, "accuracy"),
        "macro_f1": bootstrap_ci(frame, "macro_f1"),
        "brightness_stability": bootstrap_ci(frame, "stability_brightness"),
        "contrast_stability": bootstrap_ci(frame, "stability_contrast"),
        "translation_stability": bootstrap_ci(frame, "stability_translation"),
        "deletion_aopc": bootstrap_ci(frame, "deletion_aopc"),
    }
    frame["confidence_bin"] = pd.cut(frame.calibrated_confidence, [0, .5, .7, .85, 1.000001], include_lowest=True)
    for key, group in frame.groupby("confidence_bin", observed=True):
        summaries["confidence_bins"][str(key)] = {
            "n": int(len(group)), "accuracy": float(group.correct.mean()),
            "brightness_stability": float(group.stability_brightness.mean()),
            "mean_calibrated_confidence": float(group.calibrated_confidence.mean()),
        }
    (out / "metrics.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

