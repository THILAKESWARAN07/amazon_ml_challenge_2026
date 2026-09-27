"""
Comprehensive Blocking Strategy Benchmarking Script for Amazon ML Challenge 2026.
Evaluates:
- Strategy 1: Exact Normalized & Compact Keys
- Strategy 2: Informative Name Token Inverted Index
- Strategy 3: Character N-Grams / Prefixes (Typo Robustness)
- Strategy 4: Address Token & Numeric Key Blocking
- Strategy 5: Open-Set Country Partitioning
- Strategy Combined: Multi-Stage Union + Deduplication + Pruning
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import polars as pl

# Insert package path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution"))

from src.blocking import BlockingEngine, evaluate_blocking_performance
from src.normalization import (
    compact_string,
    normalize_address,
    normalize_business_name,
    normalize_country,
)


def run_blocking_benchmarks(sample_size_s1: int = 50000):
    print("=" * 70)
    print("PHASE 3 — BLOCKING STRATEGIES EVALUATION & BENCHMARKING")
    print(f"Evaluating on {sample_size_s1:,} Source 1 records against full S2/S3 targets")
    print("=" * 70)

    # 1. Load data
    t0 = time.time()
    print("Loading datasets...")
    s1_pl = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
    s2_pl = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
    s3_pl = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")
    gt_pl = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")

    print(f"Datasets loaded in {time.time() - t0:.2f}s")
    print(f"Total rows - S1: {len(s1_pl):,}, S2: {len(s2_pl):,}, S3: {len(s3_pl):,}, GT: {len(gt_pl):,}")

    # Sample S1 and get matching GT
    s1_sample_pl = s1_pl.head(sample_size_s1)
    s1_sample_df = s1_sample_pl.to_pandas()
    s1_eval_ids = set(s1_sample_df["entity_id"])
    
    # Filter targets to countries present in the sample
    countries_in_sample = set(s1_sample_df["country"].dropna().unique())
    print(f"Countries in sample: {countries_in_sample}")
    
    s2_sample_df = s2_pl.filter(pl.col("country").is_in(countries_in_sample)).to_pandas()
    s3_sample_df = s3_pl.filter(pl.col("country").is_in(countries_in_sample)).to_pandas()
    gt_df = gt_pl.to_pandas()

    print(f"Target pool sizes for countries {countries_in_sample}:")
    print(f"S2: {len(s2_sample_df):,}, S3: {len(s3_sample_df):,}")

    # Benchmark individual strategies and combined engine
    strategies_to_test = [
        ("Strategy 1: Exact Normalized & Compact Keys", ["exact_compact_name", "exact_norm_name", "exact_compact_addr"]),
        ("Strategy 2: Informative Name Tokens (IDF Inverted Index)", ["name_tokens"]),
        ("Strategy 3: Character N-Grams / Prefixes (Typo Robustness)", ["name_ngrams"]),
        ("Strategy 4: Address Tokens & Numbers", ["addr_numeric_tokens"]),
        ("Combined Multi-Stage Pipeline (Union + Pruning @ 100)", ["all"]),
        ("Combined Multi-Stage Pipeline (Union + Pruning @ 50)", ["all_top50"]),
    ]

    benchmark_results = []

    for strat_name, channels in strategies_to_test:
        print("\n" + "-" * 70)
        print(f"Running: {strat_name}")
        t_start = time.time()

        max_cand = 50 if "top50" in channels else 100
        engine = BlockingEngine(max_candidates_per_s1=max_cand, max_bucket_size=300)
        
        # Fit index
        engine.fit_and_index_targets(s2_sample_df, s3_sample_df)

        # Generate candidates according to selected channels
        results = []
        for row in s1_sample_df.itertuples(index=False):
            s1_id = str(row.entity_id)
            c = normalize_country(row.country)
            if c not in engine.country_indexes:
                continue
            indexes = engine.country_indexes[c]
            
            raw_name = str(row.business_name or "")
            raw_addr = str(row.business_address or "")
            norm_name = normalize_business_name(raw_name)
            compact_name_val = compact_string(raw_name)
            norm_addr = normalize_address(raw_addr)
            compact_addr_val = compact_string(raw_addr)

            candidate_scores = {}

            if "all" in channels or "all_top50" in channels or "exact_compact_name" in channels:
                if compact_name_val:
                    for cid in indexes["exact_compact_name"].query(compact_name_val):
                        candidate_scores[cid] = candidate_scores.get(cid, 0.0) + 10.0
                if norm_name:
                    for cid in indexes["exact_norm_name"].query(norm_name):
                        candidate_scores[cid] = candidate_scores.get(cid, 0.0) + 8.0
                if compact_addr_val and len(compact_addr_val) >= 6:
                    for cid in indexes["exact_compact_addr"].query(compact_addr_val):
                        candidate_scores[cid] = candidate_scores.get(cid, 0.0) + 7.0

            if "all" in channels or "all_top50" in channels or "name_tokens" in channels:
                name_tokens = engine._get_name_tokens(norm_name)
                if name_tokens:
                    name_tokens.sort(key=lambda t: engine.token_doc_freq[c].get(t, 0))
                    for tok in name_tokens[:2]:
                        for cid in indexes["name_tokens"].query(tok):
                            candidate_scores[cid] = candidate_scores.get(cid, 0.0) + 4.0

            if "all" in channels or "all_top50" in channels or "name_ngrams" in channels:
                if compact_name_val and len(compact_name_val) >= 4:
                    for cid in indexes["name_ngrams"].query(f"pre_{compact_name_val[:4]}"):
                        candidate_scores[cid] = candidate_scores.get(cid, 0.0) + 2.0
                    if len(compact_name_val) >= 6:
                        for cid in indexes["name_ngrams"].query(f"pre_{compact_name_val[:6]}"):
                            candidate_scores[cid] = candidate_scores.get(cid, 0.0) + 3.0

            if "all" in channels or "all_top50" in channels or "addr_numeric_tokens" in channels:
                if norm_addr:
                    for a_key in engine._get_address_numeric_keys(norm_addr):
                        for cid in indexes["addr_numeric_tokens"].query(a_key):
                            candidate_scores[cid] = candidate_scores.get(cid, 0.0) + 3.5

            sorted_cand = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:max_cand]
            for cid, sc in sorted_cand:
                results.append({"source1_entity_id": s1_id, "candidate_entity_id": cid, "score": sc})

        cand_df = pd.DataFrame(results)
        elapsed = time.time() - t_start

        # Evaluate metrics against ground truth
        metrics = evaluate_blocking_performance(cand_df, gt_df, s1_ids=s1_eval_ids)
        metrics["Strategy"] = strat_name
        metrics["Execution Time (s)"] = round(elapsed, 2)
        metrics["Throughput (S1/sec)"] = round(len(s1_sample_df) / max(elapsed, 0.001), 1)

        benchmark_results.append(metrics)
        
        print(f"Time: {elapsed:.2f}s | Throughput: {metrics['Throughput (S1/sec)']:.1f} S1/sec")
        print(f"Overall Recall: {metrics['Overall Candidate Recall (%)']}%")
        print(f"  - Source 2 Recall: {metrics['Source 2 Candidate Recall (%)']}%")
        print(f"  - Source 3 Recall: {metrics['Source 3 Candidate Recall (%)']}%")
        print(f"Candidate Stats per S1 -> Mean: {metrics['Average Candidates per S1']}, Median: {metrics['Median Candidates per S1']}, P95: {metrics['P95 Candidates per S1']}, Max: {metrics['Max Candidates per S1']}")
        print(f"Total candidate pairs: {metrics['Total Retrieved Candidate Pairs']:,}")

    # Summary table
    summary_df = pd.DataFrame(benchmark_results)
    print("\n" + "=" * 70)
    print("FINAL BLOCKING BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    cols_to_show = [
        "Strategy", "Overall Candidate Recall (%)", "Source 2 Candidate Recall (%)", 
        "Source 3 Candidate Recall (%)", "Average Candidates per S1", "P95 Candidates per S1", 
        "Max Candidates per S1", "Execution Time (s)"
    ]
    print(summary_df[cols_to_show].to_string(index=False))
    
    # Save results to experiments/results
    out_csv = Path("experiments/results/blocking_benchmark_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to {out_csv}")


if __name__ == "__main__":
    run_blocking_benchmarks(sample_size_s1=20000)
