"""
Data loader module for Business Entity Resolution dataset.
Provides clean and reusable loading functions for TSV files without modifying source data.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from .config import (
    DELIMITER,
    ENCODING,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
)


def load_tsv(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    dtype: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load a tab-separated TSV file into a pandas DataFrame.

    Parameters:
        file_path: Path to the TSV file.
        nrows: Number of rows to read (useful for quick EDA/testing).
        dtype: Optional dictionary specifying column data types.

    Returns:
        pd.DataFrame containing the loaded data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        sep=DELIMITER,
        encoding=ENCODING,
        nrows=nrows,
        dtype=dtype,
    )


def load_train_source1(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load Source 1 training data (deduplicated reference entities)."""
    return load_tsv(TRAIN_SOURCE1_PATH, nrows=nrows)


def load_train_source2(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load Source 2 training data."""
    return load_tsv(TRAIN_SOURCE2_PATH, nrows=nrows)


def load_train_source3(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load Source 3 training data."""
    return load_tsv(TRAIN_SOURCE3_PATH, nrows=nrows)


def load_train_ground_truth(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load ground truth mappings for training data."""
    return load_tsv(TRAIN_GROUND_TRUTH_PATH, nrows=nrows)


def load_test_source1(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load Source 1 test data."""
    return load_tsv(TEST_SOURCE1_PATH, nrows=nrows)


def load_test_source2(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load Source 2 test data."""
    return load_tsv(TEST_SOURCE2_PATH, nrows=nrows)


def load_test_source3(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load Source 3 test data."""
    return load_tsv(TEST_SOURCE3_PATH, nrows=nrows)
