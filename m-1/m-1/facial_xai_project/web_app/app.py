"""
Standalone Web Application for m-1 Facial XAI Project.
Features:
- Dynamic Model Selection across runs (SOAP 50-epoch, AdamW 20-epoch, etc.)
- Customizable Grad-CAM Heatmap Colormap (jet, turbo, viridis, inferno, plasma)
- Heatmap Opacity Slider (10% - 90%)
- Target Class Explanation (explain predicted class OR counterfactual classes)
- Automatic Face Detection & Cropping toggle
- Calibrated Confidence via Temperature Scaling

Usage:
  python m-1/m-1/facial_xai_project/web_app/app.py
"""

import io
import json
import os
import sys
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_NAMES
from src.model import FERClassifier, GradCAM

PORT = 5050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = PROJECT_ROOT / "runs"
TEST_DATA_DIR = PROJECT_ROOT / "fer-2013-dataset" / "test"

# Global state
loaded_models = {}  # model_id -> {'model': FERClassifier, 'gradcam': GradCAM, 'temperature': float, 'meta': dict}
current_model_id = None

def get_available_models():
    """Discover all trained checkpoints in runs/ directory."""
    models_info = []
    if not RUNS_DIR.exists():
        return models_info

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            continue

        model_id = run_dir.name
        meta = {
            "id": model_id,
            "architecture": "convnext_tiny",
            "optimizer": "adamw",
            "epochs": 20,
            "val_acc": None,
            "val_f1": None,
            "display_name": model_id,
            "is_default": False,
        }

        # Read run.json if present
        run_json = run_dir / "run.json"
        if run_json.exists():
            try:
                with open(run_json, "r") as f:
                    rdata = json.load(f)
                    meta["architecture"] = rdata.get("model", "convnext_tiny")
                    meta["optimizer"] = rdata.get("optimizer", "adamw")
                    meta["epochs"] = rdata.get("epochs", 20)
                    meta["val_f1"] = rdata.get("best_val_f1", None)
            except Exception:
                pass

        # Read history.json if present for accuracy
        hist_json = run_dir / "history.json"
        if hist_json.exists():
            try:
                with open(hist_json, "r") as f:
                    hdata = json.load(f)
                    if hdata:
                        best_acc = max((row.get("val_acc", row.get("val_accuracy", 0)) for row in hdata), default=0)
                        meta["val_acc"] = best_acc
            except Exception:
                pass

        # Build clean user-facing title
        arch_title = meta["architecture"].replace("_", "-").title()
        opt_title = meta["optimizer"].upper()
        ep_title = f"{meta['epochs']}ep"
        acc_str = f" · {meta['val_acc']*100:.1f}% Acc" if meta["val_acc"] else ""
        meta["display_name"] = f"{arch_title} ({opt_title} · {ep_title}{acc_str})"

        if "soap" in model_id.lower():
            meta["is_default"] = True

        models_info.append(meta)

    # Sort so default/SOAP comes first
    models_info.sort(key=lambda m: (not m["is_default"], m["id"]))
    return models_info

def get_or_load_model(model_id: str):
    global loaded_models, current_model_id
    if model_id in loaded_models:
        current_model_id = model_id
        return loaded_models[model_id]

    ckpt_path = RUNS_DIR / model_id / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"[m-1 Web App] Loading model '{model_id}' on {DEVICE}...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    arch = checkpoint.get("architecture", "convnext_tiny")

    clf = FERClassifier(arch, pretrained=False).to(DEVICE)
    clf.load_state_dict(checkpoint["model"])
    clf.eval()

    cam = GradCAM(clf)

    # Temperature from audit or run
    temp = 2.276
    metrics_path = RUNS_DIR / "audit_convnext" / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                temp = float(json.load(f).get("temperature_fit_on_validation", 2.276))
        except Exception:
            pass

    loaded_models[model_id] = {
        "model": clf,
        "gradcam": cam,
        "temperature": temp,
        "arch": arch,
    }
    current_model_id = model_id
    print(f"[m-1 Web App] Model '{model_id}' ready!")
    return loaded_models[model_id]

# Transform pipeline matching training
tensorize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face(pil_img: Image.Image):
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

def run_inference(
    pil_img: Image.Image,
    model_id: str = None,
    colormap: str = "jet",
    alpha: float = 0.45,
    auto_crop: bool = True,
    target_class: str = "auto"
):
    available = get_available_models()
    if not model_id:
        model_id = available[0]["id"] if available else "convnext_soap_fer2013"

    sys_bundle = get_or_load_model(model_id)
    clf = sys_bundle["model"]
    cam_gen = sys_bundle["gradcam"]
    temp = sys_bundle["temperature"]

    face_detected = False
    if auto_crop:
        cropped_img, face_detected = detect_and_crop_face(pil_img)
    else:
        cropped_img = pil_img

    gray_img = cropped_img.convert("L")
    input_tensor = tensorize(gray_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = clf(input_tensor)
        raw_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        calib_probs = torch.softmax(logits / temp, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(raw_probs))
    pred_label = CLASS_NAMES[pred_idx]
    raw_conf = float(raw_probs[pred_idx])
    calib_conf = float(calib_probs[pred_idx])

    # Target class for Grad-CAM
    cam_target_idx = pred_idx
    if target_class != "auto" and target_class in CLASS_NAMES:
        cam_target_idx = CLASS_NAMES.index(target_class)

    cam = cam_gen(input_tensor, torch.tensor([cam_target_idx], device=DEVICE))
    heat = cam.detach().cpu().squeeze().numpy()

    # Generate colored overlay
    base_gray = (input_tensor.detach().cpu().squeeze().numpy() * 0.5 + 0.5).clip(0, 1)

    valid_cmaps = {"jet", "turbo", "viridis", "inferno", "plasma", "magma"}
    chosen_cmap = colormap if colormap in valid_cmaps else "jet"
    chosen_alpha = min(max(float(alpha), 0.1), 0.95)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=140)
    ax.imshow(base_gray, cmap="gray", vmin=0, vmax=1)
    ax.imshow(heat, cmap=chosen_cmap, alpha=chosen_alpha, vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    cam_base64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    buf_crop = io.BytesIO()
    gray_img.resize((224, 224)).save(buf_crop, format="PNG")
    crop_base64 = "data:image/png;base64," + base64.b64encode(buf_crop.getvalue()).decode("utf-8")

    prob_list = []
    for i, name in enumerate(CLASS_NAMES):
        prob_list.append({
            "name": name,
            "raw": float(raw_probs[i]),
            "calibrated": float(calib_probs[i]),
        })
    prob_list = sorted(prob_list, key=lambda x: x["calibrated"], reverse=True)

    return {
        "model_id": model_id,
        "predicted_emotion": pred_label,
        "raw_confidence": raw_conf,
        "calibrated_confidence": calib_conf,
        "cam_target": CLASS_NAMES[cam_target_idx],
        "temperature": temp,
        "face_detected": face_detected,
        "probabilities": prob_list,
        "cam_overlay": cam_base64,
        "cropped_face": crop_base64,
    }

class XAIServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_path = Path(__file__).parent / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
            return

        if path == "/api/models":
            models = get_available_models()
            resp = json.dumps({"models": models, "active": current_model_id or (models[0]["id"] if models else None)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        if path == "/api/samples":
            samples = []
            if TEST_DATA_DIR.exists():
                for emo in CLASS_NAMES:
                    emo_dir = TEST_DATA_DIR / emo
                    if emo_dir.is_dir():
                        imgs = sorted(list(emo_dir.glob("*.jpg")) + list(emo_dir.glob("*.png")))
                        for img_p in imgs[:3]:
                            rel = img_p.relative_to(PROJECT_ROOT).as_posix()
                            samples.append({
                                "emotion": emo,
                                "filename": img_p.name,
                                "path": rel
                            })
            resp = json.dumps(samples).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        if path == "/api/sample-image":
            qs = urllib.parse.parse_qs(parsed.query)
            rel_path = qs.get("path", [None])[0]
            if rel_path:
                full_path = (PROJECT_ROOT / rel_path).resolve()
                if str(full_path).startswith(str(PROJECT_ROOT)) and full_path.exists():
                    data = full_path.read_bytes()
                    mime = "image/png" if full_path.suffix.lower() == ".png" else "image/jpeg"
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_error(404, "Image not found")
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            try:
                data = json.loads(body.decode("utf-8"))
                img_data_str = data.get("image", "")
                model_id = data.get("model_id", None)
                colormap = data.get("colormap", "jet")
                alpha = float(data.get("alpha", 0.45))
                auto_crop = bool(data.get("auto_crop", True))
                target_class = data.get("target_class", "auto")

                if "," in img_data_str:
                    img_data_str = img_data_str.split(",", 1)[1]

                img_bytes = base64.b64decode(img_data_str)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                result = run_inference(
                    pil_img=pil_img,
                    model_id=model_id,
                    colormap=colormap,
                    alpha=alpha,
                    auto_crop=auto_crop,
                    target_class=target_class
                )
                resp = json.dumps(result).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            except Exception as e:
                import traceback
                traceback.print_exc()
                err = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            return

        self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        pass

def run():
    available = get_available_models()
    if available:
        default_id = available[0]["id"]
        get_or_load_model(default_id)
    else:
        print("[m-1 Web App] Warning: No models found in runs/ directory!")

    server_address = ("0.0.0.0", PORT)
    httpd = HTTPServer(server_address, XAIServerHandler)
    print("\n" + "="*60)
    print("  🚀  m-1 XAI Visual Evidence & Multi-Model Testing Panel")
    print("="*60)
    print(f"  URL: http://127.0.0.1:{PORT}")
    print(f"  Available models: {[m['id'] for m in available]}")
    print("="*60 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[m-1 Web App] Shutting down.")

if __name__ == "__main__":
    run()
