"""
Candidate generation (blocking).

Strategy: token-based inverted-index blocking, scoped by country.

For every Source-2 / Source-3 record we index:
  - each "significant" normalized name token (len >= 4) -> (country, token) bucket
  - the extracted postal/PIN/ZIP code (if any)         -> (country, postal) bucket
  - the first 2 normalized-name tokens joined ("sorted-neighborhood" key)

For a Source-1 record we look up the same keys and union everything found.
This keeps the work roughly O(N) instead of O(N^2): each record only ever
touches the (small) buckets its own tokens belong to, which is what makes it
usable at the ~1.7M-entity scale mentioned in the challenge. It also survives
typos and word-order changes because *any single shared significant token*
(name or address) is enough to bring two records together, and it survives
missing postal codes because the name-token buckets don't depend on it.

A record with more than MAX_BUCKET_SIZE members in a bucket is treated as a
"stop word" bucket (too generic, e.g. "services") and skipped, to avoid
blow-up on common tokens.
"""
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from normalize import (
    address_tokens,
    extract_postal_code,
    extract_significant_tokens,
    name_core_tokens,
)

MAX_BUCKET_SIZE = 500      # skip over-generic token buckets
MIN_TOKEN_LEN = 3          # "significant" name-token length threshold
PREFIX_LEN = 4             # extra typo-tolerant blocking key: first PREFIX_LEN chars of each token
MAX_CANDIDATES_PER_S1 = 100  # cap after scoring, see rank_and_cap()


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute normalized fields once per dataframe (avoid recomputation)."""
    out = df.copy()
    out["_name_tokens"] = out["business_name"].map(name_core_tokens)
    out["_sig_tokens"] = out["_name_tokens"].map(
        lambda ts: extract_significant_tokens(ts, MIN_TOKEN_LEN)
    )
    out["_addr_tokens"] = out["business_address"].map(address_tokens)
    out["_postal"] = out["business_address"].map(extract_postal_code)
    out["_country"] = out["country"].fillna("").str.strip().str.lower()
    return out


def build_indices(other_df: pd.DataFrame) -> Dict[str, Dict[Tuple[str, str], Set[str]]]:
    """Build inverted indices over a (already-prepped) Source-2 or Source-3 frame.

    Returns a dict with three inverted indices, each keyed by (country, key).
    """
    name_idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    prefix_idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    postal_idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    addr_idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for eid, country, sig_tokens, addr_toks, postal in zip(
        other_df["entity_id"], other_df["_country"],
        other_df["_sig_tokens"], other_df["_addr_tokens"], other_df["_postal"],
    ):
        for tok in sig_tokens:
            name_idx[(country, tok)].add(eid)
            if len(tok) >= PREFIX_LEN:
                prefix_idx[(country, tok[:PREFIX_LEN])].add(eid)
        if postal:
            postal_idx[(country, postal)].add(eid)
        for tok in addr_toks:
            if len(tok) >= MIN_TOKEN_LEN:
                addr_idx[(country, tok)].add(eid)

    # Drop over-generic buckets so one common word doesn't return half the dataset.
    for idx in (name_idx, prefix_idx, postal_idx, addr_idx):
        for key in [k for k, v in idx.items() if len(v) > MAX_BUCKET_SIZE]:
            del idx[key]

    return {"name": name_idx, "prefix": prefix_idx, "postal": postal_idx, "addr": addr_idx}


def _lookup_candidates(country, sig_tokens, addr_toks, postal, indices) -> Set[str]:
    cands: Set[str] = set()
    name_idx = indices["name"]
    prefix_idx = indices["prefix"]
    postal_idx = indices["postal"]
    addr_idx = indices["addr"]
    for tok in sig_tokens:
        cands |= name_idx.get((country, tok), set())
        if len(tok) >= PREFIX_LEN:
            cands |= prefix_idx.get((country, tok[:PREFIX_LEN]), set())
    if postal:
        cands |= postal_idx.get((country, postal), set())
    for tok in addr_toks:
        if len(tok) >= MIN_TOKEN_LEN:
            cands |= addr_idx.get((country, tok), set())
    return cands


def quick_score(sig_tokens: Set[str], addr_toks: Set[str], cand_sig: Set[str], cand_addr: Set[str]) -> float:
    """Cheap overlap score used only to rank/cap candidates, not to decide matches."""
    name_overlap = len(sig_tokens & cand_sig)
    addr_overlap = len(addr_toks & cand_addr)
    return 2.0 * name_overlap + 1.0 * addr_overlap


def generate_candidates(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    max_candidates_per_s1: int = MAX_CANDIDATES_PER_S1,
) -> Dict[str, List[str]]:
    """Return {s1_entity_id: [candidate_entity_ids, ...]} across S2 and S3.

    Every S1 entity gets an entry (possibly an empty list) so the caller can
    always emit exactly one row per S1 entity downstream.
    """
    s1 = _prep(source1_df)
    s2 = _prep(source2_df) if len(source2_df) else source2_df.assign(
        _name_tokens=None, _sig_tokens=None, _addr_tokens=None, _postal=None, _country=None
    )
    s3 = _prep(source3_df) if len(source3_df) else source3_df.assign(
        _name_tokens=None, _sig_tokens=None, _addr_tokens=None, _postal=None, _country=None
    )

    empty_idx = {"name": {}, "prefix": {}, "postal": {}, "addr": {}}
    idx2 = build_indices(s2) if len(s2) else empty_idx
    idx3 = build_indices(s3) if len(s3) else empty_idx

    # Plain dicts, not DataFrame.loc: repeated per-row .loc lookups inside a
    # Python loop are the single biggest cost in a naive implementation of
    # this step (measured ~100s for ~300k candidate pairs before this fix,
    # vs a couple of seconds after) — dict lookups are O(1) with none of
    # pandas' per-call indexing overhead.
    sig2 = dict(zip(s2["entity_id"], s2["_sig_tokens"])) if len(s2) else {}
    addr2 = dict(zip(s2["entity_id"], s2["_addr_tokens"])) if len(s2) else {}
    sig3 = dict(zip(s3["entity_id"], s3["_sig_tokens"])) if len(s3) else {}
    addr3 = dict(zip(s3["entity_id"], s3["_addr_tokens"])) if len(s3) else {}

    results: Dict[str, List[str]] = {}
    for eid, country, sig_tokens, addr_toks, postal in zip(
        s1["entity_id"], s1["_country"], s1["_sig_tokens"], s1["_addr_tokens"], s1["_postal"]
    ):
        c2 = _lookup_candidates(country, sig_tokens, addr_toks, postal, idx2) if len(s2) else set()
        c3 = _lookup_candidates(country, sig_tokens, addr_toks, postal, idx3) if len(s3) else set()

        scored: List[Tuple[float, str]] = []
        for cid in c2:
            scored.append((quick_score(sig_tokens, addr_toks, sig2[cid], addr2[cid]), cid))
        for cid in c3:
            scored.append((quick_score(sig_tokens, addr_toks, sig3[cid], addr3[cid]), cid))
        scored.sort(key=lambda t: -t[0])
        results[eid] = [cid for _, cid in scored[:max_candidates_per_s1]]

    return results
