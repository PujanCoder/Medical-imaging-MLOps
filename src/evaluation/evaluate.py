import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from src.logger import logger

def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list,
    criterion: torch.nn.Module = None
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "loss": float(running_loss / total_samples) if criterion is not None else None,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")
    if criterion is not None:
        logger.info(f"Loss     : {metrics['loss']:.4f}")
    logger.info("=" * 50)

    return metrics, all_preds, all_labels

def save_evaluation_results(
    metrics: Dict,
    save_dir: str = "artifacts/evaluation"
) -> str:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results_file = save_path / "evaluation_metrics.json"

    import json
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Evaluation metrics saved to: {results_file}")
    return str(results_file)

def print_confusion_matrix(
    cm: np.ndarray,
    class_names: list
) -> None:
    logger.info("Confusion Matrix:")
    header = " " * 15 + "  ".join([f"{name:>12}" for name in class_names])
    logger.info(header)

    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>15}" + "  ".join([f"{val:>12}" for val in row])
        logger.info(row_str)