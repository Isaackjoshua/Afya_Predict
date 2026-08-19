"""Intervention feedback loops (shortcoming #15).

Existing systems do not learn from what happened after an alert. If a forecast
triggers bed-net distribution and malaria then falls, nothing records whether
the fall was the intervention working or the season turning — so the model
never improves, and neither does the response playbook.

This package closes that loop: log what was done, compare the observed
trajectory against the counterfactual the model had already forecast, and feed
the estimated effect back into future planning.
"""

from src.intervention_tracking.feedback_loop import FeedbackLoop  # noqa: F401
from src.intervention_tracking.impact_estimator import ImpactEstimate, ImpactEstimator  # noqa: F401
from src.intervention_tracking.intervention_logger import InterventionLogger  # noqa: F401
