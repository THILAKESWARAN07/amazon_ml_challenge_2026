# Business Entity Resolution — Amazon ML Challenge 2026

## Overview
This repository contains the modular solution architecture for the **Amazon ML Challenge 2026: Business Entity Resolution**.

The objective is to match business entities across disparate datasets (`Source 1`, `Source 2`, `Source 3`), using `Source 1` as the deduplicated reference source. The evaluation metric is the **macro-averaged $F_{0.5}$ score** per Source 1 entity, prioritizing high precision while maintaining strong recall.

---

## Planned Architecture & Pipeline

The resolution pipeline is designed across the following stages:

1. **Configuration & Data Ingestion (`config.py`, `data_loader.py`)**:
   - Centralized, relative-path management ensuring machine-independent reproducibility.
   - Robust TSV loaders with proper delimiter handling (`\t`) and memory-aware batching.

2. **Exploratory Data Analysis (`eda.py`)** *(Phase 1 Focus)*:
   - Comprehensive analysis of dataset schemas, missing values, duplicate frequencies, string lengths, open-set country distributions (e.g., US, India, France), and ground truth match distribution.

3. **Data Normalization (`normalization.py`)**:
   - Rule-based business name cleaning (legal suffix standardization, punctuation removal, case folding) and address normalization.

4. **Blocking & Candidate Generation (`blocking.py`)**:
   - High-recall search space reduction to generate high-quality candidate pairs for `candidate_pairs.tsv`.

5. **Feature Engineering (`features.py`)**:
   - Multidimensional pairwise feature calculation:
     - String similarity metrics (Levenshtein, Jaro-Winkler, token sorting)
     - Character & token n-gram TF-IDF cosine similarities
     - Country and geographical agreement indicators

6. **Entity Matching Model (`model.py`, `train.py`, `evaluate.py`)**:
   - Lightweight classifier (e.g., GBDT) optimized for the $F_{0.5}$ metric with optimal decision threshold tuning.
   - Evaluation adhering strictly to macro-averaged $F_{0.5}$ per Source 1 entity.

7. **Inference & Prediction (`predict.py`, `pipeline.py`)**:
   - End-to-end inference producing valid `matching_results.tsv` and `candidate_pairs.tsv` adhering to all submission requirements.

---

## Current Status

> **Current Phase: Phase 1 — Dataset EDA**
> 
> The project structure and placeholder modules have been initialized. All downstream components (normalization, blocking, feature extraction, ML models, and inference) are set up as clean stubs awaiting Phase 1 exploration and subsequent phase implementations.

---

## Directory Structure

```text
student_resource/
├── dataset/
│   ├── train/                 # Original training TSV files & ground truth
│   └── test/                  # Original test TSV files
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py          # Central configuration & paths
│       │   ├── data_loader.py     # TSV loading utilities
│       │   ├── eda.py             # Phase 1: EDA analysis module
│       │   ├── normalization.py   # Normalization placeholder
│       │   ├── blocking.py        # Candidate generation placeholder
│       │   ├── features.py        # Feature engineering placeholder
│       │   ├── model.py           # Model architecture placeholder
│       │   ├── train.py           # Training workflow placeholder
│       │   ├── predict.py         # Prediction pipeline placeholder
│       │   ├── evaluate.py        # Macro F0.5 evaluation placeholder
│       │   └── pipeline.py        # End-to-end pipeline placeholder
│       ├── README.md
│       └── requirements.txt
├── experiments/
│   ├── notebooks/             # Exploratory and experimental notebooks
│   ├── results/               # Validation results and metric logs
│   └── logs/                  # Execution and training logs
├── models/                    # Saved model artifacts
└── output/
    ├── matching_results.tsv   # Formatted final prediction output
    └── candidate_pairs.tsv    # Candidate pairs from blocking stage
```

---

## Key Challenge Constraints & Rules

- **Closed Dataset**: Strictly no external business databases, APIs, geocoding lookups, or external business datasets.
- **Reference Source**: Source 1 is deduplicated. Each Source 1 entity may have 0, 1, or multiple matches in Source 2 / Source 3.
- **Open-Set Fields**: Countries and text fields are treated dynamically (supporting unseen countries in test data such as France).
- **Candidate Integrity**: Final matches must be a subset of candidate pairs in `candidate_pairs.tsv`.
- **Model Constraints**: Compliant with MIT/Apache 2.0 licenses and $\le 8\text{B}$ parameter limitations.
