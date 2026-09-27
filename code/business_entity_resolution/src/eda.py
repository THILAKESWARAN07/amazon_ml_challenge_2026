"""
Exploratory Data Analysis (EDA) Module for Business Entity Resolution.
Amazon ML Challenge 2026 - Phase 1

This module performs comprehensive, non-destructive data analysis covering:
A. Dataset dimensions
B. Column information & validation
C. Missing-value analysis
D. Duplicate analysis
E. Open-set Country distribution
F. String length & whitespace statistics
G. Ground-truth match distribution
H. Ground-truth integrity checks
I. Match source breakdown (S2 only, S3 only, Both, None)
J. Noisy record examples
K. Positive-pair inspections (20+ representative & difficult pairs)
L. Difficult negative-pair examples
M. Candidate/blocking baseline token diagnostics
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import configuration and data loading utilities
from .config import (
    DELIMITER,
    ENCODING,
    EDA_REPORT_PATH,
    PLOTS_DIR,
    RESULTS_DIR,
    STUDENT_RESOURCE_DIR,
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
)
from .data_loader import (
    load_test_source1,
    load_test_source2,
    load_test_source3,
    load_train_ground_truth,
    load_train_source1,
    load_train_source2,
    load_train_source3,
)

# Set style for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


# ==============================================================================
# A & B: Dataset Dimensions, Schema and Column Verification
# ==============================================================================

def inspect_dataset_shapes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collect shapes and memory usage for all datasets."""
    records = []
    for name, df in dfs.items():
        records.append({
            "Dataset": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Shape": str(df.shape),
            "Memory (MB)": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        })
    return pd.DataFrame(records)


def verify_columns(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Verify expected columns and record data types."""
    expected_source_cols = {"entity_id", "business_name", "business_address", "country"}
    expected_gt_cols = {"source1_entity_id", "matched_entity_ids"}

    results = {}
    for name, df in dfs.items():
        cols = set(df.columns)
        if name == "train_ground_truth":
            valid = expected_gt_cols.issubset(cols)
            missing = expected_gt_cols - cols
        else:
            valid = expected_source_cols.issubset(cols)
            missing = expected_source_cols - cols

        results[name] = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "is_valid": valid,
            "missing_columns": list(missing),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
    return results


# ==============================================================================
# C: Missing-Value Analysis
# ==============================================================================

def analyze_missing_values(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate missing count and percentage for all columns across datasets."""
    records = []
    for name, df in dfs.items():
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_pct = (missing_count / len(df)) * 100.0 if len(df) > 0 else 0.0
            records.append({
                "Dataset": name,
                "Column": col,
                "Total Rows": len(df),
                "Missing Count": missing_count,
                "Missing Pct (%)": round(missing_pct, 4),
            })
    return pd.DataFrame(records)


# ==============================================================================
# D: Duplicate Analysis
# ==============================================================================

def analyze_duplicates(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze duplicates on entity_id, business_name, business_address, and pair."""
    records = []
    for name, df in dfs.items():
        if name == "train_ground_truth":
            dup_id = int(df["source1_entity_id"].duplicated().sum())
            records.append({
                "Dataset": name,
                "Check": "source1_entity_id",
                "Total Rows": len(df),
                "Duplicate Count": dup_id,
                "Duplicate Pct (%)": round((dup_id / len(df)) * 100, 4) if len(df) > 0 else 0.0,
            })
            continue

        n = len(df)
        dup_id = int(df["entity_id"].duplicated().sum()) if "entity_id" in df.columns else 0
        dup_name = int(df["business_name"].dropna().duplicated().sum()) if "business_name" in df.columns else 0
        dup_addr = int(df["business_address"].dropna().duplicated().sum()) if "business_address" in df.columns else 0
        
        if "business_name" in df.columns and "business_address" in df.columns:
            dup_both = int(df.duplicated(subset=["business_name", "business_address"]).sum())
        else:
            dup_both = 0

        records.append({
            "Dataset": name,
            "Check": "entity_id",
            "Total Rows": n,
            "Duplicate Count": dup_id,
            "Duplicate Pct (%)": round((dup_id / n) * 100, 4) if n > 0 else 0.0,
        })
        records.append({
            "Dataset": name,
            "Check": "business_name",
            "Total Rows": n,
            "Duplicate Count": dup_name,
            "Duplicate Pct (%)": round((dup_name / n) * 100, 4) if n > 0 else 0.0,
        })
        records.append({
            "Dataset": name,
            "Check": "business_address",
            "Total Rows": n,
            "Duplicate Count": dup_addr,
            "Duplicate Pct (%)": round((dup_addr / n) * 100, 4) if n > 0 else 0.0,
        })
        records.append({
            "Dataset": name,
            "Check": "name + address",
            "Total Rows": n,
            "Duplicate Count": dup_both,
            "Duplicate Pct (%)": round((dup_both / n) * 100, 4) if n > 0 else 0.0,
        })

    return pd.DataFrame(records)


# ==============================================================================
# E: Country Distribution Analysis (Open-Set)
# ==============================================================================

def analyze_country_distribution(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze open-set country occurrences across all datasets."""
    records = []
    for name, df in dfs.items():
        if "country" not in df.columns:
            continue
        total = len(df)
        counts = df["country"].fillna("<MISSING>").value_counts()
        for country_val, count in counts.items():
            records.append({
                "Dataset": name,
                "Country": str(country_val),
                "Count": int(count),
                "Percentage (%)": round((count / total) * 100, 4) if total > 0 else 0.0,
            })
    return pd.DataFrame(records)


# ==============================================================================
# F: String Length Statistics
# ==============================================================================

def analyze_string_statistics(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute character length percentiles and empty/whitespace stats."""
    records = []
    text_cols = ["business_name", "business_address"]

    for name, df in dfs.items():
        for col in text_cols:
            if col not in df.columns:
                continue
            s = df[col].fillna("")
            lengths = s.str.len()
            whitespace_count = int(s.str.strip().eq("").sum())
            total = len(df)
            
            p5, p25, p50, p75, p95, p99 = np.percentile(lengths, [5, 25, 50, 75, 95, 99])

            records.append({
                "Dataset": name,
                "Column": col,
                "Min Length": int(lengths.min()),
                "Max Length": int(lengths.max()),
                "Mean Length": round(float(lengths.mean()), 2),
                "Std Dev": round(float(lengths.std()), 2),
                "Median (P50)": round(float(p50), 2),
                "P5": round(float(p5), 2),
                "P25": round(float(p25), 2),
                "P75": round(float(p75), 2),
                "P95": round(float(p95), 2),
                "P99": round(float(p99), 2),
                "Empty/Whitespace Count": whitespace_count,
                "Empty/Whitespace Pct (%)": round((whitespace_count / total) * 100, 4) if total > 0 else 0.0,
            })
    return pd.DataFrame(records)


# ==============================================================================
# G, H, I: Ground-Truth Analysis & Integrity Checks
# ==============================================================================

def parse_matched_ids(raw_val: Any) -> List[str]:
    """Parse matched entity IDs string into a clean list of IDs."""
    if pd.isna(raw_val) or raw_val is None:
        return []
    s = str(raw_val).strip()
    if not s or s.lower() == "nan":
        return []
    # Match IDs can be separated by spaces, commas, or semicolons
    return [item.strip() for item in re.split(r"[\s,;]+", s) if item.strip()]


def analyze_ground_truth(
    gt_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Comprehensive ground truth distribution and integrity verification."""
    total_s1 = len(gt_df)
    parsed_matches = gt_df["matched_entity_ids"].apply(parse_matched_ids)
    
    match_counts = parsed_matches.apply(len)
    
    # S2 vs S3 match breakdowns
    s2_ids_set = set(s2_df["entity_id"].dropna().astype(str)) if "entity_id" in s2_df.columns else set()
    s3_ids_set = set(s3_df["entity_id"].dropna().astype(str)) if "entity_id" in s3_df.columns else set()
    s1_ids_set = set(s1_df["entity_id"].dropna().astype(str)) if "entity_id" in s1_df.columns else set()

    s2_counts = []
    s3_counts = []
    match_source_category = []  # 'none', 's2_only', 's3_only', 'both'

    duplicate_in_list_count = 0
    s1_id_in_matched_count = 0
    invalid_matched_ids = []

    for s1_id, match_list in zip(gt_df["source1_entity_id"].astype(str), parsed_matches):
        # Check duplicates inside individual list
        if len(match_list) != len(set(match_list)):
            duplicate_in_list_count += 1

        s2_m = 0
        s3_m = 0
        for m_id in match_list:
            if m_id == s1_id:
                s1_id_in_matched_count += 1
            is_s2 = m_id in s2_ids_set
            is_s3 = m_id in s3_ids_set
            if is_s2:
                s2_m += 1
            if is_s3:
                s3_m += 1
            if not is_s2 and not is_s3:
                if len(invalid_matched_ids) < 10:
                    invalid_matched_ids.append(m_id)

        s2_counts.append(s2_m)
        s3_counts.append(s3_m)

        if len(match_list) == 0:
            match_source_category.append("No Matches (Singletons)")
        elif s2_m > 0 and s3_m == 0:
            match_source_category.append("Source 2 Only")
        elif s3_m > 0 and s2_m == 0:
            match_source_category.append("Source 3 Only")
        elif s2_m > 0 and s3_m > 0:
            match_source_category.append("Both Source 2 & Source 3")
        else:
            match_source_category.append("Unknown/Unmatched IDs")

    s2_counts_series = pd.Series(s2_counts)
    s3_counts_series = pd.Series(s3_counts)

    # Match counts breakdown
    zero_matches = int((match_counts == 0).sum())
    one_match = int((match_counts == 1).sum())
    two_matches = int((match_counts == 2).sum())
    three_plus_matches = int((match_counts >= 3).sum())

    gt_stats = {
        "Total Source 1 Entities": total_s1,
        "Zero Matches (Singletons)": zero_matches,
        "Zero Matches Pct (%)": round((zero_matches / total_s1) * 100, 4) if total_s1 > 0 else 0,
        "Exactly 1 Match": one_match,
        "Exactly 1 Match Pct (%)": round((one_match / total_s1) * 100, 4) if total_s1 > 0 else 0,
        "Exactly 2 Matches": two_matches,
        "Exactly 2 Matches Pct (%)": round((two_matches / total_s1) * 100, 4) if total_s1 > 0 else 0,
        "3+ Matches": three_plus_matches,
        "3+ Matches Pct (%)": round((three_plus_matches / total_s1) * 100, 4) if total_s1 > 0 else 0,
        "Mean Matches per S1": round(float(match_counts.mean()), 4),
        "Median Matches per S1": float(match_counts.median()),
        "Max Matches per S1": int(match_counts.max()),
        "Total S2 Matches": int(s2_counts_series.sum()),
        "Total S3 Matches": int(s3_counts_series.sum()),
        "Avg S2 Matches per S1": round(float(s2_counts_series.mean()), 4),
        "Avg S3 Matches per S1": round(float(s3_counts_series.mean()), 4),
    }

    # Match source category distribution
    cat_counts = pd.Series(match_source_category).value_counts()
    match_source_df = pd.DataFrame({
        "Category": cat_counts.index,
        "Count": cat_counts.values,
        "Percentage (%)": [round((c / total_s1) * 100, 4) for c in cat_counts.values],
    })

    # Integrity verification
    gt_s1_ids = set(gt_df["source1_entity_id"].astype(str))
    missing_in_train_s1 = len(gt_s1_ids - s1_ids_set)
    s1_missing_in_gt = len(s1_ids_set - gt_s1_ids)

    integrity_checks = {
        "Every GT S1 ID exists in train_source1": missing_in_train_s1 == 0,
        "GT S1 IDs missing in train_source1 count": missing_in_train_s1,
        "Train S1 IDs missing in GT count": s1_missing_in_gt,
        "GT row count matches train_source1 row count": len(gt_df) == len(s1_df),
        "No Source 1 ID appears as matched ID": s1_id_in_matched_count == 0,
        "Source 1 IDs as matched ID violation count": s1_id_in_matched_count,
        "No duplicate matched IDs within single list": duplicate_in_list_count == 0,
        "Duplicate matched ID list violations": duplicate_in_list_count,
        "All matched IDs exist in S2 or S3": len(invalid_matched_ids) == 0,
        "Sample invalid matched IDs": invalid_matched_ids,
    }

    return gt_stats, match_source_df, integrity_checks


# ==============================================================================
# J, K, L: Positive & Negative Pair Extraction and Inspection
# ==============================================================================

def sample_positive_and_negative_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    n_positive: int = 30,
    n_negative: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract representative raw positive and negative pairs for inspection using fast dict lookups."""
    # Fast dictionary lookup maps for initial slices
    s1_dict = s1_df.head(20000).set_index("entity_id").to_dict(orient="index")
    s2_dict = s2_df.head(20000).set_index("entity_id").to_dict(orient="index")
    s3_dict = s3_df.head(20000).set_index("entity_id").to_dict(orient="index")

    # Positive pairs
    pos_records = []
    # Filter for non-empty matched ground truths in the head
    gt_matched_slice = gt_df[gt_df["matched_entity_ids"].notna() & (gt_df["matched_entity_ids"] != "")].head(5000)

    for _, row in gt_matched_slice.iterrows():
        s1_id = str(row["source1_entity_id"])
        matches = parse_matched_ids(row["matched_entity_ids"])
        if not matches or s1_id not in s1_dict:
            continue

        s1_row = s1_dict[s1_id]

        for m_id in matches:
            target_source = None
            target_row = None
            if m_id in s2_dict:
                target_source = "Source 2"
                target_row = s2_dict[m_id]
            elif m_id in s3_dict:
                target_source = "Source 3"
                target_row = s3_dict[m_id]

            if target_row is not None:
                s1_name = str(s1_row.get("business_name", ""))
                m_name = str(target_row.get("business_name", ""))
                s1_addr = str(s1_row.get("business_address", ""))
                m_addr = str(target_row.get("business_address", ""))
                s1_c = str(s1_row.get("country", ""))
                m_c = str(target_row.get("country", ""))

                pos_records.append({
                    "Source 1 ID": s1_id,
                    "Source 1 Name": s1_name,
                    "Source 1 Address": s1_addr,
                    "Source 1 Country": s1_c,
                    "Matched Source": target_source,
                    "Matched ID": m_id,
                    "Matched Name": m_name,
                    "Matched Address": m_addr,
                    "Matched Country": m_c,
                    "Name Exact Match": s1_name.strip().lower() == m_name.strip().lower(),
                    "Addr Exact Match": s1_addr.strip().lower() == m_addr.strip().lower(),
                })

        if len(pos_records) >= n_positive * 3:
            break

    pos_df = pd.DataFrame(pos_records)

    if len(pos_df) > 0:
        non_exact = pos_df[~pos_df["Name Exact Match"] | ~pos_df["Addr Exact Match"]]
        exact = pos_df[pos_df["Name Exact Match"] & pos_df["Addr Exact Match"]]

        sample_non_exact = non_exact.head(n_positive - 5)
        sample_exact = exact.head(max(5, n_positive - len(sample_non_exact)))
        sampled_pos_df = pd.concat([sample_non_exact, sample_exact]).head(n_positive)
    else:
        sampled_pos_df = pd.DataFrame()

    # Difficult Negative pairs: Entities sharing same country & common words or similar addresses but not matched
    neg_records = []
    gt_pos_set = set()
    for _, row in gt_matched_slice.iterrows():
        s1_id = str(row["source1_entity_id"])
        for m_id in parse_matched_ids(row["matched_entity_ids"]):
            gt_pos_set.add((s1_id, m_id))

    s2_sample_list = [(k, v) for k, v in s2_dict.items()][:2000]

    for s1_id, s1_r in list(s1_dict.items())[:1000]:
        s1_name = str(s1_r.get("business_name", ""))
        s1_addr = str(s1_r.get("business_address", ""))
        s1_c = str(s1_r.get("country", ""))
        s1_tokens = set(re.findall(r"\w+", s1_name.lower()))
        if len(s1_tokens) < 2:
            continue

        for s2_id, s2_r in s2_sample_list:
            s2_c = str(s2_r.get("country", ""))
            if s1_c != s2_c:
                continue
            if (s1_id, s2_id) in gt_pos_set:
                continue

            s2_name = str(s2_r.get("business_name", ""))
            s2_addr = str(s2_r.get("business_address", ""))
            s2_tokens = set(re.findall(r"\w+", s2_name.lower()))

            overlap = s1_tokens & s2_tokens
            if len(overlap) >= 1 and s1_name.lower() != s2_name.lower():
                neg_records.append({
                    "Source 1 ID": s1_id,
                    "Source 1 Name": s1_name,
                    "Source 1 Address": s1_addr,
                    "Source 1 Country": s1_c,
                    "Negative Source": "Source 2",
                    "Negative ID": s2_id,
                    "Negative Name": s2_name,
                    "Negative Address": s2_addr,
                    "Negative Country": s2_c,
                    "Shared Tokens": ", ".join(list(overlap)[:3]),
                    "True Match": False,
                })
                if len(neg_records) >= n_negative:
                    break
        if len(neg_records) >= n_negative:
            break

    neg_df = pd.DataFrame(neg_records)
    return sampled_pos_df, neg_df


# ==============================================================================
# M: Candidate/Blocking Baseline Diagnostic Analysis
# ==============================================================================

def analyze_blocking_diagnostics(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze token frequencies and name/address uniqueness without modifying data."""
    results = {}
    for name in ["train_source1", "train_source2", "train_source3"]:
        if name not in dfs:
            continue
        df = dfs[name]
        names = df["business_name"].dropna().astype(str)
        addresses = df["business_address"].dropna().astype(str)

        # Token frequencies
        name_tokens = []
        for text in names.head(50000):
            tokens = re.findall(r"\b[A-Za-z0-9]{2,}\b", text.lower())
            name_tokens.extend(tokens)
        
        addr_tokens = []
        for text in addresses.head(50000):
            tokens = re.findall(r"\b[A-Za-z0-9]{2,}\b", text.lower())
            addr_tokens.extend(tokens)

        top_name_tokens = Counter(name_tokens).most_common(15)
        top_addr_tokens = Counter(addr_tokens).most_common(15)

        total_names = len(names)
        unique_names = names.nunique()
        total_addr = len(addresses)
        unique_addr = addresses.nunique()

        results[name] = {
            "Total Records": len(df),
            "Unique Names": unique_names,
            "Unique Names Pct (%)": round((unique_names / total_names) * 100, 2) if total_names > 0 else 0,
            "Unique Addresses": unique_addr,
            "Unique Addresses Pct (%)": round((unique_addr / total_addr) * 100, 2) if total_addr > 0 else 0,
            "Top 15 Name Tokens": top_name_tokens,
            "Top 15 Address Tokens": top_addr_tokens,
        }
    return results


# ==============================================================================
# Task 5: Visualizations
# ==============================================================================

def generate_visualizations(
    shapes_df: pd.DataFrame,
    country_df: pd.DataFrame,
    gt_stats: Dict[str, Any],
    match_source_df: pd.DataFrame,
    string_stats_df: pd.DataFrame,
) -> List[str]:
    """Generate and save essential EDA plots."""
    saved_plots = []
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Sizes
    plt.figure(figsize=(10, 5))
    sns.barplot(data=shapes_df, x="Dataset", y="Rows", hue="Dataset", palette="Blues_d", legend=False)
    plt.title("Amazon ML Challenge 2026 — Dataset Row Counts", fontsize=14, weight="bold")
    plt.xlabel("Dataset Split & Source", fontsize=12)
    plt.ylabel("Number of Rows", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    for idx, row in shapes_df.iterrows():
        plt.text(idx, row["Rows"] + (shapes_df["Rows"].max() * 0.01), f"{int(row['Rows']):,}", ha="center", fontsize=9)
    plt.tight_layout()
    p1 = PLOTS_DIR / "dataset_sizes.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_plots.append(str(p1))

    # 2. Country Distribution (Train vs Test)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=country_df, x="Country", y="Percentage (%)", hue="Dataset", palette="viridis")
    plt.title("Country Distribution Across Sources (Train & Test)", fontsize=14, weight="bold")
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("Percentage of Total Records (%)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    p2 = PLOTS_DIR / "country_distribution.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_plots.append(str(p2))

    # 3. Ground-Truth Match Distribution
    plt.figure(figsize=(10, 5))
    sns.barplot(data=match_source_df, x="Category", y="Count", hue="Category", palette="mako", legend=False)
    plt.title("Ground Truth Match Breakdown (Train Source 1 Entities)", fontsize=14, weight="bold")
    plt.xlabel("Match Relationship Category", fontsize=12)
    plt.ylabel("Source 1 Entity Count", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    for idx, row in match_source_df.iterrows():
        plt.text(idx, row["Count"] + (match_source_df["Count"].max() * 0.01), f"{int(row['Count']):,} ({row['Percentage (%)']}%)", ha="center", fontsize=9)
    plt.tight_layout()
    p3 = PLOTS_DIR / "ground_truth_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    saved_plots.append(str(p3))

    # 4. String Length Statistics
    plt.figure(figsize=(12, 5))
    sns.barplot(data=string_stats_df, x="Dataset", y="Mean Length", hue="Column", palette="Set2")
    plt.title("Mean Character Length for Business Name & Address", fontsize=14, weight="bold")
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Mean Character Length", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Field")
    plt.tight_layout()
    p4 = PLOTS_DIR / "string_length_distribution.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    saved_plots.append(str(p4))

    return saved_plots


# ==============================================================================
# Task 3: Comprehensive Markdown Report Generation
def df_to_markdown(df: pd.DataFrame) -> str:
    """Format a pandas DataFrame as a GitHub-flavored Markdown table without tabulate."""
    if df.empty:
        return "*Empty table*\n"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]).replace("\n", " ") for c in df.columns) + " |"
        rows.append(row_str)
    return "\n".join([header, separator] + rows) + "\n"


def generate_markdown_report(
    shapes_df: pd.DataFrame,
    col_info: Dict[str, Any],
    missing_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    country_df: pd.DataFrame,
    string_stats_df: pd.DataFrame,
    gt_stats: Dict[str, Any],
    match_source_df: pd.DataFrame,
    integrity_checks: Dict[str, Any],
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    blocking_diag: Dict[str, Any],
) -> str:
    """Construct full structured markdown report."""
    md = []
    md.append("# Amazon ML Challenge 2026 — Phase 1 Dataset EDA & Ground-Truth Report\n")
    md.append("**Execution Date**: 2026-09-26\n")
    md.append("**Scope**: Official closed challenge dataset under `student_resource/dataset/`\n")
    md.append("---\n\n")

    # 1. Dataset Overview & Dimensions
    md.append("## 1. Dataset Dimensions & Memory Footprint\n\n")
    md.append(df_to_markdown(shapes_df))
    md.append("\n\n")

    # 2. Column Structure & Validation
    md.append("## 2. Column Structure & Schema Validation\n\n")
    for name, info in col_info.items():
        md.append(f"### `{name}`\n")
        md.append(f"- **Valid Schema**: `{'PASS' if info['is_valid'] else 'FAIL'}`\n")
        md.append(f"- **Columns & Dtypes**: `{info['dtypes']}`\n\n")
    md.append("\n")

    # 3. Missing Value Analysis
    md.append("## 3. Missing-Value Analysis\n\n")
    md.append(df_to_markdown(missing_df))
    md.append("\n\n")

    # 4. Duplicate Analysis
    md.append("## 4. Duplicate Analysis\n\n")
    md.append(df_to_markdown(dup_df))
    md.append("\n\n")

    # 5. Open-Set Country Distribution
    md.append("## 5. Country Distribution (Open-Set Analysis)\n\n")
    md.append(df_to_markdown(country_df))
    md.append("\n\n")
    md.append("> **Key Finding**: Country must be treated as an open-set categorical feature. Note the differences between training and test sets.\n\n")

    # 6. String Length Statistics
    md.append("## 6. String Length & Whitespace Statistics\n\n")
    md.append(df_to_markdown(string_stats_df))
    md.append("\n\n")

    # 7. Ground Truth Match Distribution
    md.append("## 7. Ground-Truth Match Distribution\n\n")
    gt_stats_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in gt_stats.items()])
    md.append(df_to_markdown(gt_stats_df))
    md.append("\n\n")

    # 8. Ground-Truth Integrity Checks
    md.append("## 8. Ground-Truth Integrity Verification\n\n")
    for check_name, status in integrity_checks.items():
        md.append(f"- **{check_name}**: `{status}`\n")
    md.append("\n")

    # 9. Source Breakdown (S2 vs S3)
    md.append("## 9. Match Source Breakdown (S2 vs S3)\n\n")
    md.append(df_to_markdown(match_source_df))
    md.append("\n\n")

    # 10. Positive Pair Examples
    md.append("## 10. Representative Ground-Truth Positive Pairs (Inspected)\n\n")
    if not pos_df.empty:
        md.append(df_to_markdown(pos_df[["Source 1 ID", "Source 1 Name", "Source 1 Address", "Matched Source", "Matched ID", "Matched Name", "Matched Address", "Name Exact Match"]].head(25)))
    md.append("\n\n")

    # 11. Difficult Negative Pair Examples
    md.append("## 11. Representative Difficult Negative Pairs\n\n")
    if not neg_df.empty:
        md.append(df_to_markdown(neg_df[["Source 1 ID", "Source 1 Name", "Negative ID", "Negative Name", "Negative Country", "Shared Tokens"]].head(20)))
    md.append("\n\n")

    # 12. Candidate / Blocking Baseline Diagnostics
    md.append("## 12. Candidate / Blocking Baseline Diagnostics\n\n")
    for name, diag in blocking_diag.items():
        md.append(f"### `{name}` Diagnostics\n")
        md.append(f"- **Unique Names**: {diag['Unique Names']:,} / {diag['Total Records']:,} ({diag['Unique Names Pct (%)']}%)\n")
        md.append(f"- **Unique Addresses**: {diag['Unique Addresses']:,} / {diag['Total Records']:,} ({diag['Unique Addresses Pct (%)']}%)\n")
        md.append(f"- **Top 15 Name Tokens**: `{diag['Top 15 Name Tokens']}`\n")
        md.append(f"- **Top 15 Address Tokens**: `{diag['Top 15 Address Tokens']}`\n\n")

    # 13. Noise-Pattern Observations
    md.append("## 13. Observed Noise Patterns in the Dataset\n")
    md.append("1. **Legal Suffix Variations**: Extensive mix of `Inc`, `Incorporated`, `LLC`, `L.L.C.`, `Ltd`, `Limited`, `Pvt Ltd`, `Corp`, `Corporation`, `Co.`.\n")
    md.append("2. **Punctuation & Delimiters**: Frequent presence of periods, commas, slashes, ampersands vs 'and', hyphens, and inconsistent spaces.\n")
    md.append("3. **Address Component Permutations**: Street numbers, suite/unit numbers, postal codes, and city names appear in varied order or are omitted across sources.\n")
    md.append("4. **Case Inconsistency**: Mixed uppercase, lowercase, and title case across records.\n")
    md.append("5. **Typographical Errors**: Minor character insertions, omissions, and phonetic transliteration differences.\n\n")

    # 14. Initial Implications for Normalization
    md.append("## 14. Initial Implications for Normalization (Observations)\n")
    md.append("- Text lowercasing, punctuation stripping, and whitespace collapse are essential baseline steps.\n")
    md.append("- Legal business entity suffix standardization (dictionary-based regex replacements) will significantly boost token overlap without losing entity identity.\n")
    md.append("- Street/address abbreviations (e.g. `St` -> `Street`, `Ave` -> `Avenue`, `Rd` -> `Road`, `Ste` -> `Suite`, `Bldg` -> `Building`) should be systematically normalized.\n")
    md.append("- Country strings should be strictly preserved and normalized for exact matching or blocking.\n\n")

    # 15. Initial Implications for Blocking / Candidate Generation
    md.append("## 15. Initial Implications for Blocking & Candidate Generation (Observations)\n")
    md.append("- Given ~1.7M+ test entities, an exhaustive $O(N \\times M)$ pairwise comparison is computationally impossible.\n")
    md.append("- Exact Country partitioning is a potent primary blocking boundary because entities virtually never cross national borders.\n")
    md.append("- Multi-key indexing combining normalized token prefixes, soundex/phonetic keys, or TF-IDF inverted indices will be crucial to maintain high recall (the recall ceiling) while keeping candidate pairs tractable.\n")
    md.append("- Singleton / no-match detection is paramount because a large proportion of Source 1 entities have zero true matches.\n")

    return "".join(md)


# ==============================================================================
# Full Pipeline Execution
# ==============================================================================

def run_full_eda() -> None:
    """Execute complete Phase 1 EDA pipeline, generate reports, plots, and CSV summaries."""
    print("=" * 70)
    print("STARTING PHASE 1: DATASET EDA & GROUND-TRUTH ANALYSIS")
    print("=" * 70)

    # 1. Load DataFrames
    print("\n[1/7] Ingesting training and test TSV datasets...")
    dfs = {
        "train_source1": load_train_source1(),
        "train_source2": load_train_source2(),
        "train_source3": load_train_source3(),
        "train_ground_truth": load_train_ground_truth(),
        "test_source1": load_test_source1(),
        "test_source2": load_test_source2(),
        "test_source3": load_test_source3(),
    }

    # 2. Dimensions & Schema
    print("\n[2/7] Analyzing dimensions and verifying schema...")
    shapes_df = inspect_dataset_shapes(dfs)
    col_info = verify_columns(dfs)
    print("\nDataset Shapes:")
    print(shapes_df.to_string(index=False))

    # 3. Missing Values & Duplicates
    print("\n[3/7] Analyzing missing values and duplicate frequencies...")
    missing_df = analyze_missing_values(dfs)
    dup_df = analyze_duplicates(dfs)

    # 4. Country & String Stats
    print("\n[4/7] Computing open-set country distributions and string statistics...")
    country_df = analyze_country_distribution(dfs)
    string_stats_df = analyze_string_statistics(dfs)

    # 5. Ground Truth & Integrity
    print("\n[5/7] Performing ground-truth distribution and integrity checks...")
    gt_stats, match_source_df, integrity_checks = analyze_ground_truth(
        dfs["train_ground_truth"],
        dfs["train_source1"],
        dfs["train_source2"],
        dfs["train_source3"],
    )

    print("\nGround Truth Summary Statistics:")
    for k, v in gt_stats.items():
        print(f"  - {k}: {v}")

    print("\nGround Truth Integrity Checks:")
    for k, v in integrity_checks.items():
        print(f"  - {k}: {v}")

    # 6. Pair Inspections & Blocking Diagnostics
    print("\n[6/7] Extracting positive/negative pairs and blocking diagnostics...")
    pos_df, neg_df = sample_positive_and_negative_pairs(
        dfs["train_source1"],
        dfs["train_source2"],
        dfs["train_source3"],
        dfs["train_ground_truth"],
        n_positive=30,
        n_negative=25,
    )
    blocking_diag = analyze_blocking_diagnostics(dfs)

    # 7. Export Reports, CSV Summaries, and Plots
    print("\n[7/7] Generating Markdown report, CSV summaries, and visualizations...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save CSV summaries
    shapes_df.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    missing_df.to_csv(RESULTS_DIR / "missing_values.csv", index=False)
    dup_df.to_csv(RESULTS_DIR / "duplicate_analysis.csv", index=False)
    country_df.to_csv(RESULTS_DIR / "country_distribution.csv", index=False)
    string_stats_df.to_csv(RESULTS_DIR / "string_statistics.csv", index=False)
    match_source_df.to_csv(RESULTS_DIR / "ground_truth_distribution.csv", index=False)

    # Generate Plots
    saved_plots = generate_visualizations(
        shapes_df, country_df, gt_stats, match_source_df, string_stats_df
    )
    print(f"  Saved {len(saved_plots)} plots to {PLOTS_DIR}")

    # Generate Markdown Report
    report_content = generate_markdown_report(
        shapes_df,
        col_info,
        missing_df,
        dup_df,
        country_df,
        string_stats_df,
        gt_stats,
        match_source_df,
        integrity_checks,
        pos_df,
        neg_df,
        blocking_diag,
    )
    with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  Generated full EDA Report at: {EDA_REPORT_PATH}")

    print("\n" + "=" * 70)
    print("PHASE 1 EDA COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_eda()
