import torch
from pathlib import Path
from src.logger import logger
from src.data.ingestion import ingest_data
from src.data.preprocessing import create_data_loaders
from src.models.model import create_model
from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model, save_evaluation_results

def run_retraining_pipeline(
    source_data_dir: str,
    raw_data_dir: str = "data/raw",
    batch_size: int = 16,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    model_save_dir: str = "artifacts/models",
    eval_save_dir: str = "artifacts/evaluation"
):
    logger.info("=" * 60)
    logger.info("Starting Automated Retraining Pipeline")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Step 1: Data Ingestion")
    ingest_data(source_dir=source_data_dir, destination_dir=raw_data_dir)

    logger.info("Step 2: Data Preprocessing & Loaders")
    train_loader, val_loader, test_loader, info = create_data_loaders(
        data_dir=raw_data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    class_names = info["classes"]
    num_classes = len(class_names)

    logger.info("Step 3: Model Creation")
    model = create_model(num_classes=num_classes, pretrained=True)

    logger.info("Step 4: Training")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=model_save_dir
    )

    best_model_path = Path(model_save_dir) / "best_model.pth"

    logger.info("Step 5: Evaluation on Test Set")
    from src.models.model import load_model
    best_model = load_model(str(best_model_path), num_classes=num_classes, device=device)

    metrics, preds, labels = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        device=device,
        class_names=class_names
    )

    save_evaluation_results(metrics, save_dir=eval_save_dir)

    logger.info("=" * 60)
    logger.info("Retraining Pipeline Completed Successfully")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info("=" * 60)

    return {
        "history": history,
        "metrics": metrics,
        "model_path": str(best_model_path),
        "class_names": class_names
    }

if __name__ == "__main__":
    run_retraining_pipeline(
        source_data_dir="path/to/your/new_data",
        epochs=5,
        batch_size=16
    )
    