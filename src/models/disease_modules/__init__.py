"""Plugin disease modules (shortcoming #11).

Each module is a thin subclass of `StandardDiseaseModule` that encodes what is
genuinely specific to that disease. Everything else — ingestion, features,
training, explanation, alerting — comes from the shared engine.
"""

from src.models.disease_modules.cholera_module import CholeraModule  # noqa: F401
from src.models.disease_modules.hiv_module import HIVModule  # noqa: F401
from src.models.disease_modules.malaria_module import MalariaModule  # noqa: F401
from src.models.disease_modules.respiratory_module import RespiratoryModule  # noqa: F401
from src.models.disease_modules.tb_module import TuberculosisModule  # noqa: F401

__all__ = [
    "CholeraModule",
    "HIVModule",
    "MalariaModule",
    "RespiratoryModule",
    "TuberculosisModule",
]
