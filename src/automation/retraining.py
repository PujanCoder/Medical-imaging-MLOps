import torch
from pathlib import Path

from src.logger import logger
from src.data.ingestion import ingest_data
from src.data.preprocessing import create_data_loaders
from src.models.model import create_model, load_model
from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model, save_evaluation_results


def run_retraining_pipeline(
    data_dir: str = "data/raw/chest",
    batch_size: int = 16,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    model_save_dir: str = "artifacts/retrained_models",
    eval_save_dir: str = "artifacts/retrained_evaluation",
):
    logger.info("=" * 60)
    logger.info("Starting Automated Retraining Pipeline")
    logger.info("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Using device: {device}")

    # --------------------------------------------------
    # 1. DATA VERIFICATION
    # --------------------------------------------------

    logger.info("Step 1: Dataset Verification")

    ingest_data()

    # --------------------------------------------------
    # 2. DATA PREPROCESSING
    # --------------------------------------------------

    logger.info("Step 2: Data Preprocessing")

    train_loader, val_loader, test_loader, info = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        image_size=image_size,
        num_workers=0,
        seed=42,
    )

    logger.info(f"Dataset information: {info}")

    class_names = info["classes"]
    num_classes = len(class_names)

    # --------------------------------------------------
    # 3. CREATE NEW MODEL
    # --------------------------------------------------

    logger.info("Step 3: Creating NEW model")

    model = create_model(
        num_classes=num_classes,
        pretrained=True,
    )

    model = model.to(device)

    # --------------------------------------------------
    # 4. RETRAIN
    # --------------------------------------------------

    logger.info("Step 4: Retraining model")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        class_weights=None,
        save_dir=model_save_dir,
    )

    # --------------------------------------------------
    # 5. LOAD BEST MODEL
    # --------------------------------------------------

    best_model_path = (
        Path(model_save_dir) / "best_model.pth"
    )

    logger.info(
        f"Loading best retrained model: {best_model_path}"
    )

    best_model = load_model(
        str(best_model_path),
        num_classes=num_classes,
        device=device,
    )

    # --------------------------------------------------
    # 6. EVALUATION
    # --------------------------------------------------

    logger.info("Step 5: Evaluating retrained model")

    metrics, predictions, targets = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    logger.info(
        f"Retrained model metrics: {metrics}"
    )

    save_evaluation_results(
        metrics,
        save_dir=eval_save_dir,
    )

    # --------------------------------------------------
    # 7. COMPLETE
    # --------------------------------------------------

    logger.info("=" * 60)
    logger.info("Automated Retraining Pipeline Completed")
    logger.info(
        f"Best model saved at: {best_model_path}"
    )
    logger.info(
        f"Test Accuracy: {metrics['accuracy']:.4f}"
    )
    logger.info("=" * 60)

    return {
        "history": history,
        "metrics": metrics,
        "model_path": str(best_model_path),
        "class_names": class_names,
    }


if __name__ == "__main__":

    run_retraining_pipeline(
        data_dir="data/raw/chest",
        epochs=1,
        batch_size=16,
    )