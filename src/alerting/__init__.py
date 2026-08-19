"""Alerting and decision support (shortcoming #12).

An alert that says "cholera risk HIGH in Mwanza" and stops there pushes the
hard part back onto the reader. Every alert this package emits names the
action, the quantity, the deadline and the owner.
"""

from src.alerting.alert_generator import AlertGenerator, build_alert  # noqa: F401
from src.alerting.recommendation_engine import RecommendationEngine  # noqa: F401
from src.alerting.risk_classifier import RiskClassifier  # noqa: F401
