"""
F_0.5 macro-average scorer, matching the challenge's evaluation exactly:
computed per Source-1 entity, then averaged. A correctly-predicted singleton
(no true matches, no predicted matches) scores 1.0.
"""
from typing import Dict, Set

import pandas as pd


def parse_id_list(cell) -> Set[str]:
    if not isinstance(cell, str) or not cell.strip():
        return set()
    return {x.strip() for x in cell.split(",") if x.strip()}


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    return {
        row["source1_entity_id"]: parse_id_list(row["matched_entity_ids"])
        for _, row in df.iterrows()
    }


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def score_entity(true_ids: Set[str], pred_ids: Set[str]) -> float:
    if not true_ids and not pred_ids:
        return 1.0  # correctly predicted singleton
    if not pred_ids:  # missed everything, but there were true matches
        return 0.0
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids) if true_ids else 0.0
    return f_beta(precision, recall, beta=0.5)


def macro_f_half(
    predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]
) -> float:
    """Macro-average F_0.5 over every S1 entity present in `ground_truth`."""
    if not ground_truth:
        return 0.0
    scores = [
        score_entity(true_ids, predictions.get(s1, set()))
        for s1, true_ids in ground_truth.items()
    ]
    return sum(scores) / len(scores)
