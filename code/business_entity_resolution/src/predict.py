#!/usr/bin/env python3
"""
Run inference on the test set and write the two required output files.

Usage (from code/business_entity_resolution/src):
    python3 predict.py --test-dir ../../../dataset/test \
                        --model-path ../models/model.joblib \
                        --output-dir ../../../output
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from blocking import generate_candidates
from features import compute_features, prep_frame
from io_utils import load_source, write_id_list_tsv
from model import load_model, predict_proba
from pipeline import build_tfidf_indices, candidates_to_pair_table, group_matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model-path", default="models/model.joblib")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--max-candidates", type=int, default=100)
    args = ap.parse_args()

    s1 = load_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(args.test_dir, "test_source3.tsv"))
    print(f"Loaded test S1={len(s1)} S2={len(s2)} S3={len(s3)}")
    all_s1_ids = list(s1["entity_id"])

    candidates = generate_candidates(s1, s2, s3, max_candidates_per_s1=args.max_candidates)
    pairs = candidates_to_pair_table(candidates)
    print(f"Generated {len(pairs)} candidate pairs")

    name_index, addr_index, name_token_index, addr_token_index = build_tfidf_indices(s1, s2, s3)
    s1_prepped = prep_frame(s1)
    other_prepped = prep_frame(pd.concat([s2, s3], ignore_index=True))

    clf, threshold, _ = load_model(args.model_path)
    if len(pairs):
        X = compute_features(pairs, s1_prepped, other_prepped, name_index, addr_index, name_token_index, addr_token_index)
        probs = pd.Series(predict_proba(clf, X), index=pairs.index)
    else:
        probs = pd.Series([], dtype=float)
    print(f"Using threshold={threshold}")

    candidate_map = group_matches(pairs, pd.Series([True] * len(pairs), index=pairs.index), all_s1_ids)
    match_map = group_matches(pairs, probs >= threshold, all_s1_ids) if len(pairs) else {
        eid: set() for eid in all_s1_ids
    }

    os.makedirs(args.output_dir, exist_ok=True)
    write_id_list_tsv(
        match_map, "source1_entity_id", "matched_entity_ids",
        os.path.join(args.output_dir, "matching_results.tsv"),
    )
    write_id_list_tsv(
        candidate_map, "source1_entity_id", "candidate_entity_ids",
        os.path.join(args.output_dir, "candidate_pairs.tsv"),
    )
    n_matched = sum(1 for v in match_map.values() if v)
    print(f"Wrote outputs. {n_matched}/{len(all_s1_ids)} S1 entities got >=1 match.")


if __name__ == "__main__":
    main()
