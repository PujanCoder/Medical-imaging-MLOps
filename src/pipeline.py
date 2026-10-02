from src.logger import logger

from src.data.ingestion import ingest_data
from src.data.preprocessing import create_data_loaders
import torch
from src.models.model import create_model
from src.models.train import train_model

from src.evaluation.evaluate import evaluate_model
def main():

    logger.info("========================================")
    logger.info("Starting Medical Imaging ML Pipeline")
    logger.info("========================================")

    # ============================================================
    # 1. DATA INGESTION
    # ============================================================

    logger.info("Step 1/5: Data ingestion")

    ingest_data(
    )

    # ============================================================
    # 2. DATA PREPROCESSING
    # ============================================================

    logger.info("Step 2/5: Data preprocessing")

    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
    data_dir="data/raw/chest",
        batch_size=16,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        image_size=224,
        num_workers=0,
        seed=42
    )

    logger.info(f"Dataset information: {dataset_info}")

    # ============================================================
    # 3. DEVICE
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Using device: {device}")

    # ============================================================
    # 4. MODEL CREATION
    # ============================================================

    logger.info("Step 3/5: Creating model")

    model = create_model(
        num_classes=2,
        pretrained=True
    )

    model = model.to(device)

    # ============================================================
    # 5. TRAINING
    # ============================================================

    logger.info("Step 4/5: Training model")

    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=5,
        learning_rate=0.0001,
        class_weights=None,
        save_dir="artifacts/models"
    )

    logger.info(f"Training results: {training_results}")

    # ============================================================
    # 6. EVALUATION
    # ============================================================

    logger.info("Step 5/5: Evaluating model")

    class_names = [
        "Normal",
        "Tuberculosis"
    ]

    metrics, predictions, targets = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=class_names
    )

    logger.info(f"Evaluation metrics: {metrics}")

    logger.info("========================================")
    logger.info("Medical Imaging ML Pipeline Completed")
    logger.info("========================================")

    return metrics


if __name__ == "__main__":
    main()