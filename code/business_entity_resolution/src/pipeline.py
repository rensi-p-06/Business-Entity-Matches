"""Shared steps between training and inference: candidate-table construction,
TF-IDF index building over a source pool, and pair labelling against ground truth."""
from typing import Dict, Set

import pandas as pd

from features import TfidfIndex, TokenSetIndex
from normalize import address_tokens, name_core_tokens, normalize_address, normalize_name


def candidates_to_pair_table(candidates: Dict[str, list]) -> pd.DataFrame:
    """Flatten {s1_id: [cand_ids]} into a long (s1_id, cand_id) pair table."""
    rows = [(s1, cid) for s1, cids in candidates.items() for cid in cids]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])


def build_tfidf_indices(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame):
    """Fit name/address char n-gram TF (cosine) and token-Jaccard indices over
    every entity in the pool (S1 + S2 + S3), so S1 rows and S2/S3 rows live
    in the same vector space for both similarity families."""
    all_names, all_addrs = {}, {}
    all_name_tokens, all_addr_tokens = {}, {}
    for df in (s1, s2, s3):
        for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
            all_names[eid] = normalize_name(name)
            all_addrs[eid] = normalize_address(addr)
            all_name_tokens[eid] = name_core_tokens(name)
            all_addr_tokens[eid] = address_tokens(addr)
    name_index = TfidfIndex.build(all_names)
    addr_index = TfidfIndex.build(all_addrs)
    name_token_index = TokenSetIndex.build(all_name_tokens)
    addr_token_index = TokenSetIndex.build(all_addr_tokens)
    return name_index, addr_index, name_token_index, addr_token_index


def label_pairs(pairs: pd.DataFrame, ground_truth: Dict[str, Set[str]]) -> pd.Series:
    """1 if (s1_id, cand_id) is a true match per ground truth, else 0."""
    return pairs.apply(
        lambda r: 1 if r["cand_id"] in ground_truth.get(r["s1_id"], set()) else 0, axis=1
    )


def group_matches(pairs: pd.DataFrame, keep_mask, all_s1_ids) -> Dict[str, Set[str]]:
    """Group kept (s1_id, cand_id) rows into {s1_id: {cand_ids}}, including
    an empty set for every id in `all_s1_ids` that has no kept rows."""
    result: Dict[str, Set[str]] = {eid: set() for eid in all_s1_ids}
    kept = pairs[keep_mask]
    for s1, cid in zip(kept["s1_id"], kept["cand_id"]):
        result[s1].add(cid)
    return result
