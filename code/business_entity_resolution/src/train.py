#!/usr/bin/env python3
"""
Train the entity-matching classifier on the labeled training data.

Usage (from code/business_entity_resolution/src):
    python3 train.py --train-dir ../../../dataset/train --model-out ../models/model.joblib

Since no test-set labels are provided, the decision threshold has to be
chosen from training data alone — but tuning it on a single small holdout
split is unstable (a threshold that looks best on ~15% of entities can be
noise, and it silently fails to generalize). This script instead does
K-fold cross-validation over the Source-1 training entities: for each fold
it trains on the other folds and scores the held-out fold, so every
training entity contributes exactly one out-of-fold prediction. The
threshold is tuned once against this full out-of-fold set (much larger and
far less noisy than one holdout slice), and the final model shipped for
inference is then refit on *all* training data at that threshold.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from blocking import generate_candidates
from evaluate import macro_f_half
from features import compute_features, prep_frame
from io_utils import load_ground_truth, load_source
from model import predict_proba, save_model, train_model
from pipeline import build_tfidf_indices, candidates_to_pair_table, group_matches, label_pairs


def parse_ground_truth_dict(gt_df: pd.DataFrame):
    out = {}
    for _, row in gt_df.iterrows():
        cell = row["matched_entity_ids"]
        out[row["source1_entity_id"]] = (
            {x.strip() for x in cell.split(",") if x.strip()} if cell else set()
        )
    return out


def tune_threshold(pairs, probs, ground_truth, eval_ids):
    """Grid-search the probability threshold that maximizes macro F_0.5."""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.02):
        preds = group_matches(pairs, probs >= t, eval_ids)
        f = macro_f_half(preds, {k: v for k, v in ground_truth.items() if k in eval_ids})
        if f > best_f:
            best_f, best_t = f, t
    return round(float(best_t), 2), best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model-out", default="models/model.joblib")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(args.train_dir, "train_source3.tsv"))
    gt_df = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    ground_truth = parse_ground_truth_dict(gt_df)

    print(f"Loaded S1={len(s1)} S2={len(s2)} S3={len(s3)} ground_truth={len(gt_df)}")

    # Candidate generation and TF-cosine features run once over the *full*
    # training pool; folds only decide which S1 ids' pairs are held out when.
    candidates = generate_candidates(s1, s2, s3, max_candidates_per_s1=args.max_candidates)
    pairs_all = candidates_to_pair_table(candidates)

    # Diagnostic: recall ceiling imposed by blocking (how many true matches
    # were even reachable as candidates) — the PDF calls this out explicitly.
    all_true_pairs = {(s1id, m) for s1id, ms in ground_truth.items() for m in ms}
    reachable = set(zip(pairs_all["s1_id"], pairs_all["cand_id"]))
    n_true = len(all_true_pairs)
    n_reachable_true = len(all_true_pairs & reachable)
    print(
        f"Blocking recall ceiling: {n_reachable_true}/{n_true} true matches "
        f"reachable as candidates ({0 if n_true == 0 else 100*n_reachable_true/n_true:.1f}%)"
    )

    name_index, addr_index, name_token_index, addr_token_index = build_tfidf_indices(s1, s2, s3)
    s1_prepped = prep_frame(s1)
    other_prepped = prep_frame(pd.concat([s2, s3], ignore_index=True))

    s1_ids = np.array(list(s1["entity_id"]))
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof_frames, oof_probs = [], []
    for fold, (train_idx, held_idx) in enumerate(kf.split(s1_ids)):
        train_ids, held_ids = set(s1_ids[train_idx]), set(s1_ids[held_idx])
        fold_train_pairs = pairs_all[pairs_all["s1_id"].isin(train_ids)].reset_index(drop=True)
        fold_held_pairs = pairs_all[pairs_all["s1_id"].isin(held_ids)].reset_index(drop=True)
        if fold_train_pairs.empty or fold_held_pairs.empty:
            continue

        X_train = compute_features(fold_train_pairs, s1_prepped, other_prepped, name_index, addr_index, name_token_index, addr_token_index)
        y_train = label_pairs(fold_train_pairs, ground_truth)
        fold_clf = train_model(X_train, y_train, random_state=args.seed)

        X_held = compute_features(fold_held_pairs, s1_prepped, other_prepped, name_index, addr_index, name_token_index, addr_token_index)
        held_probs = predict_proba(fold_clf, X_held)

        print(f"Fold {fold + 1}/{args.n_folds}: train_pairs={len(fold_train_pairs)} "
              f"held_pairs={len(fold_held_pairs)} held_positives={int(y_train.sum() > 0)}")
        oof_frames.append(fold_held_pairs)
        oof_probs.append(held_probs)

    oof_pairs = pd.concat(oof_frames, ignore_index=True)
    oof_probs = np.concatenate(oof_probs)
    threshold, oof_f = tune_threshold(oof_pairs, oof_probs, ground_truth, set(s1_ids))
    print(f"Out-of-fold macro F_0.5 across all {len(s1_ids)} training entities: {oof_f:.4f}")
    print(f"Chosen threshold={threshold}")

    # Refit the final model on ALL training pairs at the chosen threshold.
    X_all = compute_features(pairs_all, s1_prepped, other_prepped, name_index, addr_index, name_token_index, addr_token_index)
    y_all = label_pairs(pairs_all, ground_truth)
    final_clf = train_model(X_all, y_all, random_state=args.seed)

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    save_model(final_clf, threshold, args.model_out)
    print(f"Saved final model (trained on all {len(pairs_all)} pairs) to {args.model_out}")


if __name__ == "__main__":
    main()
