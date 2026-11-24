from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    RegNet_Y_800MF_Weights,
    ResNet18_Weights,
    convnext_tiny,
    regnet_y_800mf,
    resnet18,
)


class FERClassifier(nn.Module):
    """Small model registry for the prior work's ResNet/RegNet/ConvNeXt baselines."""

    def __init__(self, architecture: str = "convnext_tiny", num_classes: int = 7,
                 pretrained: bool = True):
        super().__init__()
        self.architecture = architecture
        if architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.net = resnet18(weights=weights)
            old = self.net.conv1
            self.net.conv1 = self._gray_conv(old)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
            self.cam_layer = self.net.layer4[-1]
        elif architecture == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.net = convnext_tiny(weights=weights)
            old = self.net.features[0][0]
            self.net.features[0][0] = self._gray_conv(old)
            self.net.classifier[2] = nn.Linear(self.net.classifier[2].in_features, num_classes)
            self.cam_layer = self.net.features[-1]
        elif architecture == "regnet_y_800mf":
            weights = RegNet_Y_800MF_Weights.DEFAULT if pretrained else None
            self.net = regnet_y_800mf(weights=weights)
            old = self.net.stem[0]
            self.net.stem[0] = self._gray_conv(old)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
            self.cam_layer = self.net.trunk_output
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    @staticmethod
    def _gray_conv(old: nn.Conv2d) -> nn.Conv2d:
        new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride,
                        old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FERResNet18(FERClassifier):
    """Backward-compatible ResNet-18 alias."""

    def __init__(self, num_classes: int = 7):
        super().__init__("resnet18", num_classes=num_classes, pretrained=False)


class GradCAM:
    """Minimal Grad-CAM for the final residual block."""

    def __init__(self, model: FERClassifier):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = model.cam_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __call__(self, images: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        if target is None:
            target = logits.argmax(dim=1)
        score = logits.gather(1, target.view(-1, 1)).sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        return cam.detach()

    def close(self):
        self.handle.remove()

