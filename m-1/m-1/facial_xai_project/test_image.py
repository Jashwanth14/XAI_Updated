"""
Single Image Testing & Visual Explanation CLI for m-1 Facial XAI.

Usage:
  .venv\Scripts\python.exe -m m-1.m-1.facial_xai_project.test_image --image path/to/face.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_NAMES
from src.model import FERClassifier, GradCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensorize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return pil_img, False
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.15)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(rgb.shape[1], x + w + pad_w)
    y1 = min(rgb.shape[0], y + h + pad_h)
    return pil_img.crop((x0, y0, x1, y1)), True

def test_image(image_path: str, checkpoint_path: str, output_path: str = "xai_result.png"):
    img_p = Path(image_path)
    if not img_p.exists():
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        sys.exit(1)

    print(f"\nLoading model: {ckpt_p.name} on {DEVICE}...")
    checkpoint = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
    arch = checkpoint.get("architecture", "convnext_tiny")
    model = FERClassifier(arch, pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    gradcam = GradCAM(model)

    raw_img = Image.open(img_p).convert("RGB")
    cropped_img, detected = detect_and_crop(raw_img)
    if detected:
        print("  Face automatically detected and cropped.")
    else:
        print("  No bounding box found; using direct image crop.")

    gray_img = cropped_img.convert("L")
    x = tensorize(gray_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    # Grad-CAM heatmap
    cam = gradcam(x, torch.tensor([pred_idx], device=DEVICE))
    heat = cam.detach().cpu().squeeze().numpy()
    base_gray = (x.detach().cpu().squeeze().numpy() * 0.5 + 0.5).clip(0, 1)

    # Save visual comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=150)
    axes[0].imshow(raw_img)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(base_gray, cmap="gray")
    axes[1].set_title("Processed Face (224×224)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(base_gray, cmap="gray")
    im = axes[2].imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title(f"Grad-CAM: {pred_label.upper()} ({conf*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    gradcam.close()

    print("\n" + "═" * 50)
    print(f"  PREDICTED EMOTION :  {pred_label.upper()}")
    print(f"  CONFIDENCE        :  {conf * 100:.2f}%")
    print("═" * 50)
    print("  Class Probability Distribution:")
    sorted_probs = sorted([(CLASS_NAMES[i], probs[i]) for i in range(len(CLASS_NAMES))], key=lambda item: item[1], reverse=True)
    for name, p in sorted_probs:
        bar = "█" * int(25 * p)
        print(f"    {name:<10} : {p * 100:>5.1f}%  {bar}")
    print("─" * 50)
    print(f"  Saved Visual Explanation to: {Path(output_path).resolve()}")
    print("═" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an image with m-1 Facial XAI model")
    parser.add_argument("--image", required=True, help="Path to input image (jpg/png)")
    parser.add_argument("--checkpoint", default="m-1/m-1/facial_xai_project/runs/convnext_soap_fer2013/best.pt",
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--output", default="m-1/m-1/facial_xai_project/xai_result.png",
                        help="Path to save output visualization image")
    args = parser.parse_args()
    test_image(args.image, args.checkpoint, args.output)

