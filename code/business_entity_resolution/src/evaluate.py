"""
Evaluation module for Business Entity Resolution.
Calculates macro-averaged Precision, Recall, and F0.5 per Source 1 entity.
"""

from typing import Dict, Set
import pandas as pd


def compute_f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Calculate F-beta score given precision and recall.
    Formula: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    """
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


def compute_macro_f05(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute macro-averaged precision, recall, and F0.5 across all Source 1 entities.

    Parameters:
        ground_truth_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
        predictions_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']

    Returns:
        Dict with 'precision', 'recall', 'f05'.
    """
    raise NotImplementedError("Evaluation logic will be implemented in subsequent phases.")
