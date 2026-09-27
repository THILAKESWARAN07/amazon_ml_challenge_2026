
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
│       │   ├── config.py
│       │   ├── data_loader.py
│       │   ├── eda.py
│       │   ├── normalization.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   ├── model.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── evaluate.py
│       │   └── pipeline.py
│       │
│       ├── README.md
│       └── requirements.txt
│
├── experiments/
│   ├── notebooks/
│   ├── results/
│   └── logs/
│
├── models/
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
└── README.md
```

## Current Phase

- **Phase 1 — Dataset EDA**: Project structure initialized with modular stubs.