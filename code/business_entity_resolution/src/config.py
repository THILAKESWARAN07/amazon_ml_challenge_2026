"""
Configuration and central path definitions for Amazon ML Challenge 2026: Business Entity Resolution.
All paths are resolved dynamically relative to the student_resource directory.
"""

from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SRC_DIR.parent
CODE_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

# Student Resource & Dataset Paths
STUDENT_RESOURCE_DIR = PROJECT_ROOT / "student_resource"
DATASET_DIR = STUDENT_RESOURCE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Train Files
TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

# Test Files
TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

# Artifact and Output Paths (located at root)
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
NOTEBOOKS_DIR = EXPERIMENTS_DIR / "notebooks"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Standard Output Filenames
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
EDA_REPORT_PATH = RESULTS_DIR / "eda_report.md"

# Delimiter and encoding
DELIMITER = "\t"
ENCODING = "utf-8"

# Evaluation Metric Parameters
F_BETA = 0.5

# Ensure output directories exist
for directory in [MODELS_DIR, NOTEBOOKS_DIR, RESULTS_DIR, LOGS_DIR, PLOTS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

