import torch
import torch.nn as nn
from torchvision import models
from src.logger import logger

def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    logger.info(f"Model created: ResNet18 | Classes: {num_classes} | Pretrained: {pretrained}")
    return model

def load_model(
    model_path: str,
    num_classes: int = 2,
    device: torch.device = None
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Model loaded from: {model_path}")
    return model