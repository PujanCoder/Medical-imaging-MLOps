import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from typing import Dict

from src.logger import logger


def freeze_backbone(model):
    """Freeze pretrained ResNet layers and train only the classifier."""

    for param in model.parameters():
        param.requires_grad = False

    # ResNet18 final classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    logger.info("ResNet18 backbone frozen. Training only final classifier.")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(loader, desc="Val", leave=False):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    class_weights: torch.Tensor = None,
    save_dir: str = "artifacts/models"
) -> Dict:

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Freeze ResNet18 backbone
    freeze_backbone(model)

    # Loss
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    optimizer = Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = -1.0

    best_model_path = save_path / "best_model.pth"

    logger.info(
        f"Starting training for {epochs} epoch(s) on {device}"
    )

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc
                },
                best_model_path
            )

            logger.info(
                f"Best model saved | "
                f"Val Acc: {best_val_acc:.4f}"
            )

    logger.info(
        f"Training finished. "
        f"Best Val Acc: {best_val_acc:.4f}"
    )

    return history