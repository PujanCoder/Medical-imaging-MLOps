import time
import logging
from typing import Optional

from src.automation.failure_detector import FailureEvent
from src.automation.failure_analyzer import FailureAnalysis
from src.automation.recovery_policies import RecoveryPolicyManager
from src.automation.recovery_validator import RecoveryValidator


logger = logging.getLogger(__name__)


class RecoveryEngine:
    """
    Executes automated recovery actions with controlled retries.
    """

    def __init__(self):
        self.validator = RecoveryValidator()

    def recover(
        self,
        failure: FailureEvent,
        analysis: FailureAnalysis,
        recovery_functions: Optional[dict] = None,
    ):

        policy = RecoveryPolicyManager.get_policy(
            analysis.recommended_action
        )

        logger.warning(
            "Recovery started: action=%s max_retries=%s",
            policy.action,
            policy.max_retries,
        )

        # ------------------------------------------------------
        # FAILURE CANNOT BE AUTOMATICALLY RECOVERED
        # ------------------------------------------------------

        if not analysis.recoverable:

            logger.error(
                "Failure is not automatically recoverable: %s",
                analysis.root_cause,
            )

            return {
                "success": False,
                "action": "ESCALATE",
                "attempts": 0,
                "message": "Failure requires manual intervention.",
            }

        recovery_functions = recovery_functions or {}

        recovery_function = recovery_functions.get(
            policy.action
        )

        # ------------------------------------------------------
        # NO RECOVERY FUNCTION
        # ------------------------------------------------------

        if recovery_function is None:

            logger.error(
                "No recovery function registered for action=%s",
                policy.action,
            )

            return {
                "success": False,
                "action": policy.action,
                "attempts": 0,
                "message": "No recovery function is registered.",
            }

        # ------------------------------------------------------
        # RETRY LOOP
        # ------------------------------------------------------

        for attempt in range(
            1,
            policy.max_retries + 1,
        ):

            logger.info(
                "Recovery attempt %s/%s: %s",
                attempt,
                policy.max_retries,
                policy.action,
            )

            try:

                result = recovery_function()

                validation = self.validator.validate_stage(
                    result
                )

                # --------------------------------------------------
                # RECOVERY SUCCESS
                # --------------------------------------------------

                if validation.success:

                    logger.info(
                        "Recovery successful on attempt %s.",
                        attempt,
                    )

                    return {
                        "success": True,
                        "action": policy.action,
                        "attempts": attempt,
                        "message": validation.message,
                        "metrics": validation.metrics,
                        "result": result,
                    }

                # --------------------------------------------------
                # RECOVERY ATTEMPT FAILED
                # --------------------------------------------------

                logger.warning(
                    "Recovery attempt %s failed: %s",
                    attempt,
                    validation.message,
                )

            except Exception as recovery_error:

                logger.exception(
                    "Recovery attempt %s raised an exception.",
                    attempt,
                )

                # If this was the final attempt,
                # return failure immediately.
                if attempt == policy.max_retries:

                    return {
                        "success": False,
                        "action": policy.action,
                        "attempts": attempt,
                        "message": str(
                            recovery_error
                        ),
                    }

            # --------------------------------------------------
            # WAIT BEFORE NEXT ATTEMPT
            # --------------------------------------------------

            if attempt < policy.max_retries:

                time.sleep(
                    policy.retry_delay_seconds
                )

        # ------------------------------------------------------
        # ALL RETRIES FAILED
        # ------------------------------------------------------

        logger.error(
            "Maximum recovery attempts exceeded: action=%s attempts=%s",
            policy.action,
            policy.max_retries,
        )

        return {
            "success": False,
            "action": policy.action,
            "attempts": policy.max_retries,
            "message": "Maximum recovery attempts exceeded.",
        }