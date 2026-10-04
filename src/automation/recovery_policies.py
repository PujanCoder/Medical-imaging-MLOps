from dataclasses import dataclass
from typing import Dict


@dataclass
class RecoveryPolicy:
    action: str
    max_retries: int
    retry_delay_seconds: int
    description: str


class RecoveryPolicyManager:
    """
    Maps recovery actions to controlled recovery policies.
    """

    POLICIES: Dict[str, RecoveryPolicy] = {

        # ------------------------------------------------------
        # DATA
        # ------------------------------------------------------

        "RETRY_DATA_INGESTION": RecoveryPolicy(
            action="RETRY_DATA_INGESTION",
            max_retries=3,
            retry_delay_seconds=5,
            description="Retry dataset ingestion and validation.",
        ),

        # ------------------------------------------------------
        # PREPROCESSING
        # ------------------------------------------------------

        "RETRY_PREPROCESSING": RecoveryPolicy(
            action="RETRY_PREPROCESSING",
            max_retries=2,
            retry_delay_seconds=5,
            description="Retry dataset preprocessing.",
        ),

        # ------------------------------------------------------
        # MODEL CREATION
        # ------------------------------------------------------

        "RETRY_MODEL_CREATION": RecoveryPolicy(
            action="RETRY_MODEL_CREATION",
            max_retries=2,
            retry_delay_seconds=5,
            description="Retry model creation.",
        ),

        # ------------------------------------------------------
        # TRAINING
        # ------------------------------------------------------

        "RETRY_TRAINING": RecoveryPolicy(
            action="RETRY_TRAINING",
            max_retries=3,
            retry_delay_seconds=10,
            description="Retry model training.",
        ),

        "REDUCE_RESOURCE_USAGE": RecoveryPolicy(
            action="REDUCE_RESOURCE_USAGE",
            max_retries=2,
            retry_delay_seconds=10,
            description="Reduce resource usage and retry training.",
        ),

        # ------------------------------------------------------
        # EVALUATION
        # ------------------------------------------------------

        "RETRY_EVALUATION": RecoveryPolicy(
            action="RETRY_EVALUATION",
            max_retries=2,
            retry_delay_seconds=5,
            description="Retry model evaluation.",
        ),

        # ------------------------------------------------------
        # MODEL REGISTRATION
        # ------------------------------------------------------

        "RETRY_MODEL_REGISTRATION": RecoveryPolicy(
            action="RETRY_MODEL_REGISTRATION",
            max_retries=3,
            retry_delay_seconds=5,
            description="Retry MLflow model registration.",
        ),

        "RECOVER_MODEL_ARTIFACT": RecoveryPolicy(
            action="RECOVER_MODEL_ARTIFACT",
            max_retries=2,
            retry_delay_seconds=5,
            description="Attempt to recover the required model artifact.",
        ),

        # ------------------------------------------------------
        # GENERIC RECOVERY
        # ------------------------------------------------------

        "RETRY_STAGE": RecoveryPolicy(
            action="RETRY_STAGE",
            max_retries=3,
            retry_delay_seconds=5,
            description="Retry the failed pipeline stage.",
        ),

        "RETRY_CONNECTION": RecoveryPolicy(
            action="RETRY_CONNECTION",
            max_retries=3,
            retry_delay_seconds=5,
            description="Retry the external service connection.",
        ),

        # ------------------------------------------------------
        # MANUAL INTERVENTION
        # ------------------------------------------------------

        "REVIEW_CONFIGURATION": RecoveryPolicy(
            action="REVIEW_CONFIGURATION",
            max_retries=0,
            retry_delay_seconds=0,
            description="Configuration requires manual intervention.",
        ),

        "INSTALL_DEPENDENCY": RecoveryPolicy(
            action="INSTALL_DEPENDENCY",
            max_retries=0,
            retry_delay_seconds=0,
            description="Dependency issue requires environment correction.",
        ),

        # ------------------------------------------------------
        # ESCALATION
        # ------------------------------------------------------

        "ESCALATE": RecoveryPolicy(
            action="ESCALATE",
            max_retries=0,
            retry_delay_seconds=0,
            description="Failure cannot safely be recovered automatically.",
        ),
    }

    @classmethod
    def get_policy(cls, action: str) -> RecoveryPolicy:
        """
        Return the policy for the requested recovery action.

        Unknown actions are automatically escalated.
        """

        return cls.POLICIES.get(
            action,
            cls.POLICIES["ESCALATE"],
        )