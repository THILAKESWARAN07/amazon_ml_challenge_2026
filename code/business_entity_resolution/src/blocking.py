"""
Candidate Generation and Blocking Module for Business Entity Resolution.
Amazon ML Challenge 2026 - Phase 3

Implements scalable, multi-stage candidate generation strategies:
- Strategy 1: Exact Normalized & Compact Key Blocking
- Strategy 2: Informative Name Token Inverted Index Blocking
- Strategy 3: Character N-Gram / Substring Blocking (Typo Tolerance)
- Strategy 4: Address Token & Numeric Key Blocking
- Strategy 5: Open-Set Country Partitioning
- Multi-Stage Candidate Union, Deduplication, and Dynamic Pruning
"""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from .normalization import (
    compact_string,
    extract_numeric_tokens,
    normalize_address,
    normalize_business_name,
    normalize_country,
    normalize_text,
)


class InvertedIndex:
    """
    High-performance in-memory inverted index mapping blocking keys to target entity IDs.
    """
    def __init__(self, max_bucket_size: int = 500):
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.max_bucket_size = max_bucket_size

    def add(self, key: str, entity_id: str) -> None:
        """Add a single (key, entity_id) entry."""
        if key:
            self.index[key].append(entity_id)

    def query(self, key: str) -> List[str]:
        """Query entity IDs associated with the key if bucket size is within threshold."""
        if not key:
            return []
        bucket = self.index.get(key, [])
        if len(bucket) > self.max_bucket_size:
            # Overly broad / unselective bucket: skip or truncate to prevent candidate explosion
            return []
        return bucket

    def size(self) -> int:
        return len(self.index)


class BlockingEngine:
    """
    Multi-stage scalable blocking engine for Business Entity Resolution.
    Combines exact matching, token inverted indexes, character n-grams, and address tokens
    under open-set country partitioning.
    """

    def __init__(
        self,
        max_candidates_per_s1: int = 100,
        max_bucket_size: int = 300,
        min_token_len: int = 3,
        ngram_n: int = 3,
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.max_bucket_size = max_bucket_size
        self.min_token_len = min_token_len
        self.ngram_n = ngram_n

        # Inverted index tables partitioned by country
        # Structure: country -> blocker_type -> InvertedIndex
        self.country_indexes: Dict[str, Dict[str, InvertedIndex]] = defaultdict(
            lambda: {
                "exact_compact_name": InvertedIndex(max_bucket_size=max_bucket_size),
                "exact_norm_name": InvertedIndex(max_bucket_size=max_bucket_size),
                "exact_compact_addr": InvertedIndex(max_bucket_size=max_bucket_size),
                "name_tokens": InvertedIndex(max_bucket_size=max_bucket_size),
                "name_ngrams": InvertedIndex(max_bucket_size=max_bucket_size),
                "addr_numeric_tokens": InvertedIndex(max_bucket_size=max_bucket_size),
                "addr_token_pairs": InvertedIndex(max_bucket_size=max_bucket_size),
            }
        )
        
        # Token frequency statistics for IDF-based filtering
        self.token_doc_freq: Dict[str, Counter] = defaultdict(Counter)
        self.total_docs: Dict[str, int] = defaultdict(int)

    def _get_name_tokens(self, name: str) -> List[str]:
        """Extract clean informative tokens from a business name."""
        tokens = re.findall(r"[a-z0-9]+", name.lower())
        return [t for t in tokens if len(t) >= self.min_token_len]

    def _get_char_ngrams(self, compact_name: str) -> List[str]:
        """Generate character n-grams from compact business name."""
        if len(compact_name) < self.ngram_n:
            return [compact_name] if compact_name else []
        return [compact_name[i : i + self.ngram_n] for i in range(len(compact_name) - self.ngram_n + 1)]

    def _get_address_numeric_keys(self, norm_addr: str) -> List[str]:
        """Extract building numbers, zip codes, and key street tokens."""
        numbers = re.findall(r"\b\d+\b", norm_addr)
        words = [w for w in re.findall(r"\b[a-z]{3,}\b", norm_addr) if w not in {"street", "road", "avenue", "lane", "drive"}]
        
        keys = []
        # Pair each number with the first 2 significant street words
        for num in numbers[:2]:
            for word in words[:2]:
                keys.append(f"{num}_{word}")
        return keys

    def fit_and_index_targets(
        self,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
    ) -> None:
        """
        Build inverted indexes for Source 2 and Source 3 target records partitioned by country.
        """
        # Combine target dataframes with source tagging
        targets = []
        for df, src_name in [(source2_df, "S2"), (source3_df, "S3")]:
            if df is not None and not df.empty:
                df_copy = df[["entity_id", "business_name", "business_address", "country"]].copy()
                targets.append(df_copy)
        
        if not targets:
            return

        combined_targets = pd.concat(targets, ignore_index=True)
        
        # 1. Compute token frequencies for IDF weighting per country
        for row in combined_targets.itertuples(index=False):
            country = normalize_country(row.country)
            self.total_docs[country] += 1
            name_tokens = set(self._get_name_tokens(str(row.business_name or "")))
            for tok in name_tokens:
                self.token_doc_freq[country][tok] += 1

        # 2. Populate Inverted Indexes
        for row in combined_targets.itertuples(index=False):
            e_id = str(row.entity_id)
            raw_name = str(row.business_name or "")
            raw_addr = str(row.business_address or "")
            country = normalize_country(row.country)
            indexes = self.country_indexes[country]

            norm_name = normalize_business_name(raw_name)
            compact_name_val = compact_string(raw_name)
            norm_addr = normalize_address(raw_addr)
            compact_addr_val = compact_string(raw_addr)

            # Strategy 1: Exact Keys
            if compact_name_val:
                indexes["exact_compact_name"].add(compact_name_val, e_id)
            if norm_name:
                indexes["exact_norm_name"].add(norm_name, e_id)
            if compact_addr_val and len(compact_addr_val) >= 6:
                indexes["exact_compact_addr"].add(compact_addr_val, e_id)

            # Strategy 2: Name Token Inverted Index (IDF weighted / rare tokens)
            name_tokens = self._get_name_tokens(norm_name)
            if name_tokens:
                # Rank tokens by rarity (lowest doc frequency)
                name_tokens.sort(key=lambda t: self.token_doc_freq[country].get(t, 0))
                # Index the top 3 most informative tokens
                for tok in name_tokens[:3]:
                    freq = self.token_doc_freq[country].get(tok, 0)
                    # Index if token appears in fewer than 2% of documents or fewer than max_bucket_size
                    if freq < max(50, self.total_docs[country] * 0.02):
                        indexes["name_tokens"].add(tok, e_id)

            # Strategy 3: Character N-grams for robust typo / transliteration retrieval
            if compact_name_val and len(compact_name_val) >= 4:
                ngrams = self._get_char_ngrams(compact_name_val)
                # Index prefixes and selective n-grams
                prefix_4 = compact_name_val[:4]
                indexes["name_ngrams"].add(f"pre_{prefix_4}", e_id)
                if len(compact_name_val) >= 6:
                    indexes["name_ngrams"].add(f"pre_{compact_name_val[:6]}", e_id)

            # Strategy 4: Address Token & Numeric Key Blocking
            if norm_addr:
                addr_keys = self._get_address_numeric_keys(norm_addr)
                for a_key in addr_keys:
                    indexes["addr_numeric_tokens"].add(a_key, e_id)

    def generate_candidates_for_record(
        self,
        s1_id: str,
        business_name: Optional[str],
        business_address: Optional[str],
        country: Optional[str],
    ) -> List[Tuple[str, str, float]]:
        """
        Generate candidate pairs for a single Source 1 record.
        Returns a list of (s1_id, candidate_target_id, priority_score).
        """
        norm_c = normalize_country(country)
        if norm_c not in self.country_indexes:
            return []

        indexes = self.country_indexes[norm_c]
        raw_name = str(business_name or "")
        raw_addr = str(business_address or "")

        norm_name = normalize_business_name(raw_name)
        compact_name_val = compact_string(raw_name)
        norm_addr = normalize_address(raw_addr)
        compact_addr_val = compact_string(raw_addr)

        candidate_scores: Dict[str, float] = defaultdict(float)

        # 1. Exact Compact & Normalized Name (Highest confidence channels)
        if compact_name_val:
            for c_id in indexes["exact_compact_name"].query(compact_name_val):
                candidate_scores[c_id] += 10.0
        if norm_name:
            for c_id in indexes["exact_norm_name"].query(norm_name):
                candidate_scores[c_id] += 8.0

        # 2. Exact Compact Address
        if compact_addr_val and len(compact_addr_val) >= 6:
            for c_id in indexes["exact_compact_addr"].query(compact_addr_val):
                candidate_scores[c_id] += 7.0

        # 3. Informative Name Tokens
        name_tokens = self._get_name_tokens(norm_name)
        if name_tokens:
            name_tokens.sort(key=lambda t: self.token_doc_freq[norm_c].get(t, 0))
            for tok in name_tokens[:2]:
                freq = self.token_doc_freq[norm_c].get(tok, 1)
                idf_weight = math.log((self.total_docs[norm_c] + 1.0) / (freq + 1.0))
                for c_id in indexes["name_tokens"].query(tok):
                    candidate_scores[c_id] += min(5.0, idf_weight)

        # 4. Character N-Gram / Prefix Blocking (Typo coverage)
        if compact_name_val and len(compact_name_val) >= 4:
            prefix_4 = compact_name_val[:4]
            for c_id in indexes["name_ngrams"].query(f"pre_{prefix_4}"):
                candidate_scores[c_id] += 2.0
            if len(compact_name_val) >= 6:
                prefix_6 = compact_name_val[:6]
                for c_id in indexes["name_ngrams"].query(f"pre_{prefix_6}"):
                    candidate_scores[c_id] += 3.0

        # 5. Address Numeric Tokens
        if norm_addr:
            addr_keys = self._get_address_numeric_keys(norm_addr)
            for a_key in addr_keys:
                for c_id in indexes["addr_numeric_tokens"].query(a_key):
                    candidate_scores[c_id] += 3.5

        if not candidate_scores:
            return []

        # Candidate Deduplication & Pruning
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        pruned_candidates = sorted_candidates[: self.max_candidates_per_s1]

        return [(s1_id, c_id, score) for c_id, score in pruned_candidates]

    def build_candidate_pairs(
        self,
        source1_df: pd.DataFrame,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate candidate pairs for all Source 1 records against Source 2 and Source 3.
        Returns a DataFrame in candidate_pairs.tsv standard format.
        """
        # Fit index on target sources
        self.fit_and_index_targets(source2_df, source3_df)

        results = []
        for row in source1_df.itertuples(index=False):
            s1_id = str(row.entity_id)
            name = row.business_name
            addr = row.business_address
            country = row.country

            candidates = self.generate_candidates_for_record(s1_id, name, addr, country)
            for _, c_id, score in candidates:
                results.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": c_id,
                    "blocking_score": round(score, 2),
                })

        return pd.DataFrame(results)


def evaluate_blocking_performance(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive blocking metrics:
    - Candidate Recall (Ground Truth Recall) for S2, S3, and Overall
    - Total candidate pairs generated
    - Candidate count distribution per Source 1 entity (Mean, Median, P90, P95, P99, Max)
    - Reduction ratio vs Cartesian space
    """
    # Filter ground truth to evaluated S1 subset if specified
    if s1_ids is not None:
        gt_filtered = ground_truth_df[ground_truth_df["source1_entity_id"].isin(s1_ids)].copy()
    else:
        gt_filtered = ground_truth_df.copy()

    # Build true pairs set
    true_pairs_all = set()
    true_pairs_s2 = set()
    true_pairs_s3 = set()

    for row in gt_filtered.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        matched_str = str(row.matched_entity_ids or "")
        if not matched_str or matched_str == "nan":
            continue
        for m_id in matched_str.split(","):
            m_id = m_id.strip()
            if not m_id:
                continue
            pair = (s1_id, m_id)
            true_pairs_all.add(pair)
            if m_id.startswith("S2"):
                true_pairs_s2.add(pair)
            elif m_id.startswith("S3"):
                true_pairs_s3.add(pair)

    # Build retrieved candidate pairs set
    retrieved_pairs = set()
    candidates_per_s1 = defaultdict(int)

    if not candidate_pairs_df.empty:
        for row in candidate_pairs_df.itertuples(index=False):
            s1_id = str(row.source1_entity_id)
            c_id = str(row.candidate_entity_id)
            retrieved_pairs.add((s1_id, c_id))
            candidates_per_s1[s1_id] += 1

    # Add 0 candidates for S1 entities that retrieved 0
    all_eval_s1 = set(gt_filtered["source1_entity_id"])
    for s1_id in all_eval_s1:
        if s1_id not in candidates_per_s1:
            candidates_per_s1[s1_id] = 0

    candidate_counts = list(candidates_per_s1.values())

    # Recall calculation
    total_true = len(true_pairs_all)
    captured_true = len(true_pairs_all.intersection(retrieved_pairs))
    overall_recall = (captured_true / total_true * 100.0) if total_true > 0 else 0.0

    captured_s2 = len(true_pairs_s2.intersection(retrieved_pairs))
    s2_recall = (captured_s2 / len(true_pairs_s2) * 100.0) if true_pairs_s2 else 0.0

    captured_s3 = len(true_pairs_s3.intersection(retrieved_pairs))
    s3_recall = (captured_s3 / len(true_pairs_s3) * 100.0) if true_pairs_s3 else 0.0

    return {
        "Total Evaluated S1 Entities": len(all_eval_s1),
        "Total True Ground-Truth Pairs": total_true,
        "Total Retrieved Candidate Pairs": len(retrieved_pairs),
        "Captured True Pairs": captured_true,
        "Overall Candidate Recall (%)": round(overall_recall, 4),
        "Source 2 Candidate Recall (%)": round(s2_recall, 4),
        "Source 3 Candidate Recall (%)": round(s3_recall, 4),
        "Average Candidates per S1": round(float(np.mean(candidate_counts)), 2),
        "Median Candidates per S1": float(np.median(candidate_counts)),
        "P90 Candidates per S1": float(np.percentile(candidate_counts, 90)),
        "P95 Candidates per S1": float(np.percentile(candidate_counts, 95)),
        "P99 Candidates per S1": float(np.percentile(candidate_counts, 99)),
        "Max Candidates per S1": int(np.max(candidate_counts)) if candidate_counts else 0,
        "Zero Candidate S1 Count": sum(1 for c in candidate_counts if c == 0),
    }
