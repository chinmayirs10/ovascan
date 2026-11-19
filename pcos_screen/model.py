"""Model definition, preprocessing and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMG_SIZE = 224
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Ultrasound is greyscale; convert to 3 channels so ImageNet weights apply.
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def build_model(pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with a single-logit head (P(PCOS-like))."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, 1)
    return net


def build_feature_extractor() -> nn.Module:
    """Pretrained backbone returning pooled embeddings (used for the leakage audit)."""
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier = nn.Identity()
    return net.eval()


# ---- Result bands -------------------------------------------------------
# A single image cannot diagnose PCOS. The app therefore reports a band,
# and a wide "inconclusive" zone instead of forcing a yes/no answer.

@dataclass(frozen=True)
class Band:
    key: str
    title: str
    explanation: str


BANDS = {
    "normal": Band(
        "normal",
        "No PCOS-like ovarian features detected",
        "The model did not find the multi-follicle pattern it associates with "
        "polycystic ovarian morphology in this image.",
    ),
    "inconclusive": Band(
        "inconclusive",
        "Inconclusive",
        "The model is not confident either way. This can mean the image is unclear, "
        "shows a different structure, or is unlike the training data.",
    ),
    "pcos_like": Band(
        "pcos_like",
        "PCOS-like ovarian features detected",
        "The model found a pattern similar to polycystic ovarian morphology "
        "(many small follicles). On its own this does not mean you have PCOS.",
    ),
}

LOW, HIGH = 0.30, 0.70


def band_for(prob_pcos: float, low: float = LOW, high: float = HIGH) -> Band:
    if not 0.0 <= prob_pcos <= 1.0:
        raise ValueError(f"probability out of range: {prob_pcos}")
    if prob_pcos < low:
        return BANDS["normal"]
    if prob_pcos > high:
        return BANDS["pcos_like"]
    return BANDS["inconclusive"]


class Predictor:
    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.device = device
        self.model = build_model(pretrained=False)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(device)

    @torch.no_grad()
    def predict_proba(self, image: Image.Image) -> float:
        x = eval_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return torch.sigmoid(self.model(x)).item()
