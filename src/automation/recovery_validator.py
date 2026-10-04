from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RecoveryValidation:
    success: bool
    message: str
    metrics: Optional[dict] = None

    def to_dict(self):
        return asdict(self)


class RecoveryValidator:
    """
    Validates the result of a recovery operation.
    """

    def validate_stage(self, result) -> RecoveryValidation:

        if result is None:
            return RecoveryValidation(
                success=False,
                message="Recovery returned no result."
            )

        if isinstance(result, bool):
            return RecoveryValidation(
                success=result,
                message=(
                    "Recovery completed successfully."
                    if result
                    else "Recovery failed."
                )
            )

        if isinstance(result, dict):

            success = result.get("success", False)

            return RecoveryValidation(
                success=success,
                message=result.get(
                    "message",
                    "Recovery completed."
                    if success
                    else "Recovery failed."
                ),
                metrics=result.get("metrics"),
            )

        return RecoveryValidation(
            success=True,
            message="Recovery function completed without raising an exception."
        )

    def validate_model(
        self,
        metrics: dict,
        minimum_accuracy: float = 0.70,
    ) -> RecoveryValidation:

        accuracy = metrics.get("accuracy")

        if accuracy is None:
            return RecoveryValidation(
                success=False,
                message="Accuracy was not provided."
            )

        if accuracy >= minimum_accuracy:
            return RecoveryValidation(
                success=True,
                message=f"Model passed validation with accuracy={accuracy:.4f}.",
                metrics=metrics,
            )

        return RecoveryValidation(
            success=False,
            message=(
                f"Model failed validation. "
                f"Accuracy={accuracy:.4f}, "
                f"required={minimum_accuracy:.4f}."
            ),
            metrics=metrics,
        )