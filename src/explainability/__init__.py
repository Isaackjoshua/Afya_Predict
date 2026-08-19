"""Explainability (shortcoming #4).

No prediction leaves this system without an explanation. Health officials did
not act on black-box scores, and they should not be asked to. Every forecast
carries SHAP attributions, a plain-language summary written in the mechanism's
own terms, and at least one counterfactual.
"""

from src.explainability.counterfactuals import (  # noqa: F401
    Counterfactual,
    generate_counterfactuals,
    headline_counterfactual,
)
from src.explainability.feature_importance import (  # noqa: F401
    concentration_index,
    global_importance,
    proxy_importance,
    source_importance,
)
from src.explainability.natural_language import (  # noqa: F401
    describe_driver,
    explain_prediction,
    sms_summary,
    summarise_drivers,
)
from src.explainability.shap_explainer import (  # noqa: F401
    ExplanationResult,
    ShapExplainer,
    explain_predictions,
)
