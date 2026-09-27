# Amazon ML Challenge 2026 — Business Entity Resolution

## Project Structure Overview

```text
amazon_ml_challenge_2026/
│
├── student_resource/                      # Official challenge resource folder (Unmodified)
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv
│   │   │   ├── train_source2.tsv
│   │   │   ├── train_source3.tsv
│   │   │   └── train_ground_truth.tsv
│   │   │
│   │   └── test/
│   │       ├── test_source1.tsv
│   │       ├── test_source2.tsv
│   │       └── test_source3.tsv
│   │
│   ├── utils/
│   │   └── validate_submission.py
│   │
│   ├── Documentation_template.md
│   └── README.md
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py                  # Centralized path configuration
│       │   ├── data_loader.py             # Reusable TSV loader utilities
│       │   ├── eda.py                     # Phase 1: Exploratory data analysis
│       │   ├── normalization.py           # Text & entity normalization
│       │   ├── blocking.py                # Candidate generation
│       │   ├── features.py                # Pairwise feature engineering
│       │   ├── model.py                   # Matching classifier architecture
│       │   ├── train.py                   # Model training workflow
│       │   ├── predict.py                 # Test inference script
│       │   ├── evaluate.py                # Macro F0.5 evaluation metric
│       │   └── pipeline.py                # End-to-end pipeline orchestration
│       │
│       ├── README.md
│       └── requirements.txt
│
├── experiments/
│   ├── notebooks/                         # Exploratory and experimental notebooks
│   ├── results/                           # Experiment outputs and metric logs
│   └── logs/                              # Execution logs
│
├── models/                                # Trained model artifacts
│
├── output/
│   ├── matching_results.tsv               # Final submission predictions
│   └── candidate_pairs.tsv                # Candidate pairs from blocking stage
│
└── README.md
```

## Current Phase
- **Phase 1 — Dataset EDA**: Project structure initialized with modular stubs.
