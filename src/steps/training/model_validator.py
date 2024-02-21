import os
from zenml import step

from src.config.settings import (
    ACCURACY_THRESHOLD,
    F1_THRESHOLD,
    PRECISION_THRESHOLD,
    RECALL_THRESHOLD,
    AUC_THRESHOLD,
)


@step
def model_validation(
    metrics: dict,
) -> str:
    """
    Validation du modèle :
    le modèle est adapté au déploiement, car ses performances prédictives
    sont supérieures à une certaine référence.

    Args:
        metrics: Les métriques de validation.

    Returns:
        True si le modèle est adapté au déploiement,
        False sinon.
    """

    if (
        metrics["accuracy"] < ACCURACY_THRESHOLD
        or metrics["precision"] < PRECISION_THRESHOLD
        or metrics["recall"] < RECALL_THRESHOLD
        or metrics["f1"] < F1_THRESHOLD
        or metrics["auc"] < AUC_THRESHOLD
    ):
        return False
    return True
