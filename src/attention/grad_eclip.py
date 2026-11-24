"""
Attention Map Generators for ConvNeXt-Tiny and RegNetY-800MF:
  1. FCMSCAM (FACS-Contrastive Multi-Scale CAM) — Primary Novel Contribution
  2. GradECLIP — Channel + Spatial Importance Baseline
  3. StandardGradCAM — Standard Grad-CAM Baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
import cv2

from src.face_detection.au_extractor import LANDMARKS


# Maps each AU code from AUExtractor to its MediaPipe landmark keys
AU_TO_LANDMARK_KEYS: Dict[str, List[str]] = {
    "AU1":  ["left_inner_brow", "right_inner_brow"],
    "AU2":  ["left_outer_brow", "right_outer_brow"],
    "AU4":  ["left_inner_brow", "right_inner_brow", "left_brow_mid", "right_brow_mid"],
    "AU5":  ["left_eye_upper", "left_eye_lower", "right_eye_upper", "right_eye_lower"],
    "AU6":  ["left_cheek", "right_cheek", "left_eye_lower", "right_eye_lower"],
    "AU12": ["mouth_left", "mouth_right", "left_cheek_low", "right_cheek_low"],
    "AU14": ["mouth_left", "mouth_right"],
    "AU15": ["mouth_left", "mouth_right", "lower_lip_center"],
    "AU17": ["lower_lip_center", "chin"],
    "AU24": ["upper_lip_top", "lower_lip_bottom", "upper_lip_center"],
    "AU25": ["upper_lip_top", "lower_lip_bottom"],
    "AU26": ["upper_lip_top", "lower_lip_bottom", "chin"],
}


class FCMSCAM:
    """
    FACS-Contrastive Multi-Scale Class Activation Mapping (FC-MSCAM).

    Works directly on trained checkpoints without retraining:
      - ConvNeXtTinyFER: Coarse Stage 4 = features[7] (7x7), Fine Stage 3 = features[5] (14x14)
      - RegNetY800MF_FER: Coarse Stage 4 = block4 (7x7), Fine Stage 3 = block3 (14x14)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        fine_layer: Optional[nn.Module] = None,
        device: str = "cuda",
        gamma: float = 0.60,
        beta: float = 0.70,
        alpha_hybrid: float = 0.50,
        bg_floor: float = 0.15,
        gaussian_sigma: float = 0.08,
        output_size: int = 56,
    ):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.alpha_hybrid = alpha_hybrid
        self.bg_floor = bg_floor
        self.gaussian_sigma = gaussian_sigma
        self.output_size = output_size

        coarse_auto, fine_auto = self._auto_detect_multiscale_layers(model)
        self.coarse_layer = target_layer if target_layer is not None else coarse_auto
        self.fine_layer = fine_layer if fine_layer is not None else fine_auto

        self.coarse_activations: Optional[torch.Tensor] = None
        self.coarse_gradients: Optional[torch.Tensor] = None
        self.fine_activations: Optional[torch.Tensor] = None
        self.fine_gradients: Optional[torch.Tensor] = None

        self.last_metadata: Dict = {}
        self._register_hooks()

    @staticmethod
    def _auto_detect_multiscale_layers(model: nn.Module) -> Tuple[nn.Module, Optional[nn.Module]]:
        """
        Resolves (Stage 4 Coarse 7x7, Stage 3 Fine 14x14) layers for
        ConvNeXtTinyFER and RegNetY800MF_FER.
        """
        cls_name = model.__class__.__name__

        # 1. ConvNeXt-Tiny: features[7] is 7x7 (768ch), features[5] is 14x14 (384ch)
        if cls_name == "ConvNeXtTinyFER" or (
            hasattr(model, "backbone") and hasattr(model.backbone, "features")
        ):
            return model.backbone.features[7], model.backbone.features[5]

        # 2. RegNetY-800MF: block4 is 7x7 (784ch), block3 is 14x14 (320ch)
        if cls_name == "RegNetY800MF_FER" or (
            hasattr(model, "backbone") and hasattr(model.backbone, "trunk_output")
        ):
            trunk = model.backbone.trunk_output
            return trunk.block4, trunk.block3

        # Fallback for models with get_target_layer
        if hasattr(model, "get_target_layer"):
            return model.get_target_layer(), None

        raise ValueError(
            f"Unsupported model '{cls_name}'. FC-MSCAM is configured for ConvNeXtTinyFER and RegNetY800MF_FER."
        )

    def _register_hooks(self):
        """Register forward and backward hooks on both coarse (7x7) and fine (14x14) stages."""

        def coarse_fwd(module, inp, out):
            self.coarse_activations = out.detach()

        def coarse_bwd(module, grad_in, grad_out):
            self.coarse_gradients = grad_out[0].detach()

        self.coarse_layer.register_forward_hook(coarse_fwd)
        self.coarse_layer.register_full_backward_hook(coarse_bwd)

        if self.fine_layer is not None:
            def fine_fwd(module, inp, out):
                self.fine_activations = out.detach()

            def fine_bwd(module, grad_in, grad_out):
                self.fine_gradients = grad_out[0].detach()

            self.fine_layer.register_forward_hook(fine_fwd)
            self.fine_layer.register_full_backward_hook(fine_bwd)

    def _compute_scale_saliency(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 2: Hybrid Element-Wise + Channel-Spatial Attribution at a single scale."""
        # 1. Global channel importance
        channel_weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # 2. Element-wise positive contrastive gradient
        pos_gradients = F.relu(gradients)  # (1, C, H, W)

        # 3. Spatial importance gate
        spatial_gate = pos_gradients.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        spatial_gate = spatial_gate / (spatial_gate.amax(dim=[2, 3], keepdim=True) + 1e-8)

        # 4. Hybrid attribution
        effective_weights = (
            self.alpha_hybrid * channel_weights + (1.0 - self.alpha_hybrid) * pos_gradients
        )
        cam = (effective_weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam) * (0.5 + 0.5 * spatial_gate)

        cam_max = cam.amax(dim=[2, 3], keepdim=True)
        if cam_max.item() > 0:
            cam = cam / (cam_max + 1e-8)

        return cam

    def _build_facs_landmark_prior(
        self,
        landmarks: Optional[np.ndarray],
        active_aus: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Build (56, 56) Gaussian anatomical prior M_FACS from 468 MediaPipe landmarks."""
        H = W = self.output_size
        if landmarks is None or landmarks.size == 0 or np.all(landmarks == 0):
            return np.ones((H, W), dtype=np.float32)

        ys, xs = np.meshgrid(
            np.linspace(0.0, 1.0, H, dtype=np.float32),
            np.linspace(0.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )

        active_keys = set()
        if active_aus:
            for au in active_aus:
                active_keys.update(AU_TO_LANDMARK_KEYS.get(au, []))

        facs_anchor_keys = [
            "left_inner_brow", "right_inner_brow", "left_outer_brow", "right_outer_brow",
            "left_brow_mid", "right_brow_mid", "left_eye_upper", "left_eye_lower",
            "right_eye_upper", "right_eye_lower", "nose_tip", "nose_bridge",
            "upper_lip_top", "lower_lip_bottom", "mouth_left", "mouth_right",
            "upper_lip_center", "lower_lip_center", "left_cheek", "right_cheek",
            "left_cheek_low", "right_cheek_low", "chin",
        ]

        gaussian_field = np.zeros((H, W), dtype=np.float32)
        two_sigma_sq = 2.0 * (self.gaussian_sigma ** 2)

        for key in facs_anchor_keys:
            idx = LANDMARKS.get(key)
            if idx is None or idx >= len(landmarks):
                continue
            lx, ly = float(landmarks[idx, 0]), float(landmarks[idx, 1])
            if not (0.0 <= lx <= 1.0 and 0.0 <= ly <= 1.0):
                continue

            weight = 1.40 if key in active_keys else 0.85
            dist_sq = (xs - lx) ** 2 + (ys - ly) ** 2
            kernel = weight * np.exp(-dist_sq / two_sigma_sq)
            gaussian_field = np.maximum(gaussian_field, kernel)

        if gaussian_field.max() > 0:
            gaussian_field = gaussian_field / gaussian_field.max()

        prior_mask = self.bg_floor + (1.0 - self.bg_floor) * gaussian_field
        return prior_mask.astype(np.float32)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        active_aus: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Generate FC-MSCAM attention map (56, 56) normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)

        logits = output[0]
        if target_class is None:
            _, top2_idx = logits.topk(2)
            c1 = top2_idx[0].item()
            c2 = top2_idx[1].item()
        else:
            c1 = target_class
            masked_logits = logits.clone()
            masked_logits[c1] = -1e9
            c2 = masked_logits.argmax().item()

        # STAGE 1: Contrastive Emotion Margin Backpropagation
        self.model.zero_grad()
        contrastive_score = output[0, c1] - self.gamma * output[0, c2]
        contrastive_score.backward(retain_graph=True)

        if self.coarse_activations is None or self.coarse_gradients is None:
            raise RuntimeError("FC-MSCAM hooks did not capture coarse stage activations/gradients.")

        # STAGE 2: Multi-Scale Attribution (7x7 Coarse + 14x14 Fine)
        s_coarse = self._compute_scale_saliency(self.coarse_activations, self.coarse_gradients)
        s_coarse_up = F.interpolate(
            s_coarse,
            size=(self.output_size, self.output_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(min=0.0)

        if self.fine_layer is not None and self.fine_activations is not None and self.fine_gradients is not None:
            s_fine = self._compute_scale_saliency(self.fine_activations, self.fine_gradients)
            s_fine_up = F.interpolate(
                s_fine,
                size=(self.output_size, self.output_size),
                mode="bicubic",
                align_corners=False,
            ).clamp(min=0.0)

            # STAGE 3A: Multi-Scale Spatial-Semantic Harmonic Fusion
            fused = s_coarse_up * torch.pow(s_fine_up + 0.15, self.beta)
        else:
            fused = s_coarse_up

        cam_np = fused.squeeze().cpu().numpy()

        # STAGE 3B: Dynamic 468-Landmark FACS Anatomical Prior Gate
        facs_prior = self._build_facs_landmark_prior(landmarks, active_aus)
        cam_np = cam_np * facs_prior

        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        self.last_metadata = {
            "target_class": int(c1),
            "confounder_class": int(c2),
            "contrastive_margin": float((output[0, c1] - output[0, c2]).item()),
            "gamma": self.gamma,
            "beta": self.beta,
            "used_landmarks": landmarks is not None,
        }

        return cam_np.astype(np.float32)

    def generate_for_image(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        active_aus: Optional[List[str]] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> tuple:
        raw_cam = self.generate(
            input_tensor,
            target_class=target_class,
            landmarks=landmarks,
            active_aus=active_aus,
        )

        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(raw_cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)

        return overlay, raw_cam


class GradECLIP:
    """Grad-ECLIP attention map generator (Single-scale 7x7 baseline)."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer"):
            self.target_layer = model.get_target_layer()
        else:
            raise ValueError("Model must have get_target_layer() or specify one.")
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        active_aus: Optional[List[str]] = None,
    ) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        channel_weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        spatial_weights = self.gradients.abs().mean(dim=1, keepdim=True)
        cam = (channel_weights * self.activations * spatial_weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def generate_for_image(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        active_aus: Optional[List[str]] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> tuple:
        raw_cam = self.generate(input_tensor, target_class, landmarks, active_aus)
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(raw_cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        return overlay, raw_cam


class StandardGradCAM:
    """Standard Grad-CAM (Single-scale 7x7 baseline)."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_target_layer"):
            self.target_layer = model.get_target_layer()
        else:
            raise ValueError("Model must have get_target_layer() or specify one.")
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(m, i, o):
            self.activations = o.detach()

        def bwd_hook(m, gi, go):
            self.gradients = go[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        active_aus: Optional[List[str]] = None,
    ) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def generate_for_image(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        landmarks: Optional[np.ndarray] = None,
        active_aus: Optional[List[str]] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> tuple:
        raw_cam = self.generate(input_tensor, target_class, landmarks, active_aus)
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(raw_cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        return overlay, raw_cam


def build_attention_generator(
    model: nn.Module,
    method: str = "fc_mscam",
    target_layer: Optional[nn.Module] = None,
    device: str = "cuda",
):
    """
    Factory function to build an attention map generator.
    """
    method_clean = method.lower().replace("-", "_")
    if method_clean in ("fc_mscam", "fcmscam"):
        return FCMSCAM(model, target_layer=target_layer, device=device)
    elif method_clean == "grad_eclip":
        return GradECLIP(model, target_layer, device)
    elif method_clean in ("grad_cam", "gradcam"):
        return StandardGradCAM(model, target_layer, device)
    else:
        raise ValueError(f"Unknown attention method: {method}. Choose 'fc_mscam', 'grad_eclip', or 'grad_cam'.")
