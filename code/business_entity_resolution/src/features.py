"""
Feature engineering module for Business Entity Resolution.
Placeholder functions for pairwise matching feature computation.
"""

from typing import List
import pandas as pd


def compute_string_similarities(str_a: str, str_b: str) -> dict:
    """
    Compute string-level similarities (Levenshtein, Jaro-Winkler, token sort, etc.).
    Placeholder for feature engineering phase.
    """
    raise NotImplementedError("String similarity computation will be implemented in subsequent phases.")


def compute_tfidf_similarities(corpus_a: List[str], corpus_b: List[str]) -> List[float]:
    """
    Compute TF-IDF cosine similarities.
    Placeholder for feature engineering phase.
    """
    raise NotImplementedError("TF-IDF similarity computation will be implemented in subsequent phases.")


def extract_pairwise_features(
    candidate_pairs_df: pd.DataFrame,
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract comprehensive pairwise features for candidate pairs.
    Features will include name similarity, address similarity, country agreement,
    token similarity, character similarity, TF-IDF similarity, etc.
    Placeholder for feature engineering phase.
    """
    raise NotImplementedError("Pairwise feature extraction will be implemented in subsequent phases.")
