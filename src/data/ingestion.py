from pathlib import Path
from src.logger import logger


DATASET_PATH = Path("data/raw")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def ingest_data():
    """Verify the local medical image dataset."""

    logger.info("Starting dataset verification...")

    if not DATASET_PATH.exists():
        logger.error(f"Dataset directory not found: {DATASET_PATH}")
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_PATH}"
        )

    class_dirs = [
        folder for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    ]

    if not class_dirs:
        raise ValueError("No class directories found.")

    total_images = 0

    for class_dir in class_dirs:
        images = [
            file
            for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]

        logger.info(
            f"{class_dir.name}: {len(images)} images"
        )

        total_images += len(images)

    if total_images == 0:
        raise ValueError("No valid medical images found.")

    logger.info(
        f"Dataset verification completed. "
        f"Total images: {total_images}"
    )

    return True