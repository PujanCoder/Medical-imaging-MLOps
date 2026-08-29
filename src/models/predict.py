import torch
from PIL import Image
from pathlib import Path
from typing import Union, Dict
from src.logger import logger
from src.models.model import load_model
from src.features.image_feature import get_image_transforms

def predict_image(
    image_path: Union[str, Path],
    model_path: str,
    class_names: list,
    device: torch.device = None,
    image_size: int = 224
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, num_classes=len(class_names), device=device)
    _, transform = get_image_transforms(image_size)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)

    predicted_class = class_names[pred_idx.item()]
    confidence_score = confidence.item()

    result = {
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "confidence": round(confidence_score, 4),
        "probabilities": {
            class_names[i]: round(probabilities[0][i].item(), 4)
            for i in range(len(class_names))
        }
    }

    logger.info(f"Prediction: {predicted_class} ({confidence_score:.4f})")
    return result

def predict_batch(
    image_paths: list,
    model_path: str,
    class_names: list,
    device: torch.device = None
) -> list:
    results = []
    for path in image_paths:
        result = predict_image(path, model_path, class_names, device)
        results.append(result)
    return results