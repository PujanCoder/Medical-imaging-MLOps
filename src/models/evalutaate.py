import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.logger import logger


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate the trained model on the test dataset.
    """

    logger.info("Starting model evaluation...")

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }