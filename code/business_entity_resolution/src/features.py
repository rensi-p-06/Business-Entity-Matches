"""
Pairwise feature engineering for candidate (S1, candidate) record pairs.

All features are computed purely from the two records' own fields
(business_name, business_address, country) — no external data sources.

Performance design: at the ~1.7M-entity test scale mentioned in the
challenge's validator, a plain per-pair Python loop over every candidate
pair would dominate runtime. So every feature that *can* be expressed as a
row-wise array/sparse-matrix operation is computed that way, over all pairs
at once:
  - name/address char n-gram cosine similarity -> sparse matrix multiply+sum
  - name/address token Jaccard similarity      -> sparse matrix multiply+sum
    (via a binary multi-hot encoding of each record's token set)
  - exact-match / length / country / postal comparisons -> plain NumPy array
    comparisons

Only two features have no convenient vectorized form and stay in a per-pair
loop: `name_seq_ratio` (difflib's alignment-based ratio) and
`name_common_prefix` (longest common prefix length).
"""
from difflib import SequenceMatcher
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from normalize import (
    address_tokens,
    extract_postal_code,
    name_core_tokens,
    normalize_address,
    normalize_name,
)

FEATURE_COLUMNS = [
    "name_exact", "name_seq_ratio", "name_token_jaccard", "name_len_diff",
    "name_common_prefix", "name_tfidf_cosine",
    "addr_token_jaccard", "addr_tfidf_cosine", "postal_match", "postal_both_present",
    "country_match", "name_first_token_match",
]


def prep_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute every normalized field a record needs for featurization."""
    out = df.copy()
    out["_norm_name"] = out["business_name"].map(normalize_name)
    out["_name_tokens"] = out["business_name"].map(name_core_tokens)
    out["_first_token"] = out["_name_tokens"].map(lambda ts: min(ts) if ts else "")
    out["_norm_addr"] = out["business_address"].map(normalize_address)
    out["_addr_tokens"] = out["business_address"].map(address_tokens)
    out["_postal"] = out["business_address"].map(extract_postal_code)
    out["_country"] = out["country"].fillna("").astype(str).str.strip().str.lower()
    return out.set_index("entity_id", drop=False)


def _pairwise_dot(mat, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Row-wise dot product for paired row indices — one vectorized op over
    every pair at once (fancy-index both sides, elementwise multiply, sum)."""
    a = mat[idx_a]
    b = mat[idx_b]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


class TfidfIndex:
    """One fitted vectorizer + matrix + entity_id->row-index map for a text field.

    Built once per dataset (train or test), over all entities from all three
    sources, so S1 and S2/S3 rows live in the *same* matrix and can be
    compared directly.

    IMPORTANT: use_idf=False. This is deliberately plain character n-gram
    TF cosine similarity, not classic TF-IDF. If IDF weighting were fit
    per-dataset, the resulting cosine scale would depend on that dataset's
    vocabulary statistics — a model trained on train-set cosine values would
    then be miscalibrated against test-set cosine values from a *different*
    IDF fit (this was caught during local testing: it silently pushed every
    test-set probability below the trained threshold). Disabling IDF makes
    each pair's cosine similarity a function of the two strings alone, so
    features computed at train time and at test time are on the same scale.
    """

    def __init__(self, matrix: sparse.csr_matrix, pos: Dict[str, int]):
        self.matrix = matrix
        self.pos = pos

    @classmethod
    def build(cls, id_to_text: Dict[str, str]) -> "TfidfIndex":
        ids = list(id_to_text.keys())
        texts = [id_to_text[i] for i in ids]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, use_idf=False)
        matrix = vec.fit_transform(texts if texts else [""])
        pos = {eid: i for i, eid in enumerate(ids)}
        return cls(matrix, pos)

    def cosine(self, ids_a: pd.Series, ids_b: pd.Series) -> np.ndarray:
        idx_a = ids_a.map(self.pos).to_numpy()
        idx_b = ids_b.map(self.pos).to_numpy()
        return _pairwise_dot(self.matrix, idx_a, idx_b)  # rows are L2-normalized -> dot == cosine


class TokenSetIndex:
    """Multi-hot binary encoding of each entity's token set, for vectorized
    Jaccard similarity: with binary rows, dot(A, B) = |A ∩ B|, and
    |A ∪ B| = |A| + |B| - |A ∩ B| (row sums, precomputed once)."""

    def __init__(self, matrix: sparse.csr_matrix, row_sizes: np.ndarray, pos: Dict[str, int]):
        self.matrix = matrix
        self.row_sizes = row_sizes
        self.pos = pos

    @classmethod
    def build(cls, id_to_tokens: Dict[str, Iterable[str]]) -> "TokenSetIndex":
        ids = list(id_to_tokens.keys())
        token_sets = [id_to_tokens[i] for i in ids]
        mlb = MultiLabelBinarizer(sparse_output=True)
        matrix = mlb.fit_transform(token_sets if token_sets else [[]]).astype(np.float64).tocsr()
        row_sizes = np.asarray(matrix.sum(axis=1)).ravel()
        pos = {eid: i for i, eid in enumerate(ids)}
        return cls(matrix, row_sizes, pos)

    def jaccard(self, ids_a: pd.Series, ids_b: pd.Series) -> np.ndarray:
        idx_a = ids_a.map(self.pos).to_numpy()
        idx_b = ids_b.map(self.pos).to_numpy()
        intersection = _pairwise_dot(self.matrix, idx_a, idx_b)
        union = self.row_sizes[idx_a] + self.row_sizes[idx_b] - intersection
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def compute_features(
    pairs: pd.DataFrame,
    s1_prepped: pd.DataFrame,
    other_prepped: pd.DataFrame,
    name_index: TfidfIndex,
    addr_index: TfidfIndex,
    name_token_index: TokenSetIndex,
    addr_token_index: TokenSetIndex,
) -> pd.DataFrame:
    """Compute FEATURE_COLUMNS for every row of `pairs` (columns: s1_id, cand_id).

    Every feature here is either a single vectorized array/sparse operation
    over the whole `pairs` table, or (for the two loop-only features) a
    lightweight per-pair pass over already-aligned NumPy arrays.
    """
    s1_ids, cand_ids = pairs["s1_id"], pairs["cand_id"]
    s1_rows = s1_prepped.loc[s1_ids]
    other_rows = other_prepped.loc[cand_ids]

    feats = pd.DataFrame(index=pairs.index)

    # --- vectorized sparse-matrix features ---
    feats["name_tfidf_cosine"] = name_index.cosine(s1_ids, cand_ids)
    feats["addr_tfidf_cosine"] = addr_index.cosine(s1_ids, cand_ids)
    feats["name_token_jaccard"] = name_token_index.jaccard(s1_ids, cand_ids)
    feats["addr_token_jaccard"] = addr_token_index.jaccard(s1_ids, cand_ids)

    # --- vectorized NumPy/pandas array features ---
    n1 = s1_rows["_norm_name"].to_numpy()
    n2 = other_rows["_norm_name"].to_numpy()
    c1 = s1_rows["_country"].to_numpy()
    c2 = other_rows["_country"].to_numpy()
    p1 = s1_rows["_postal"].to_numpy()
    p2 = other_rows["_postal"].to_numpy()
    f1 = s1_rows["_first_token"].to_numpy()
    f2 = other_rows["_first_token"].to_numpy()

    feats["name_exact"] = ((n1 == n2) & (n1 != "")).astype(float)
    feats["name_len_diff"] = np.abs(
        s1_rows["_norm_name"].str.len().to_numpy() - other_rows["_norm_name"].str.len().to_numpy()
    ).astype(float)
    feats["country_match"] = ((c1 == c2) & (c1 != "")).astype(float)
    postal_both = (p1 != "") & (p2 != "")
    feats["postal_both_present"] = postal_both.astype(float)
    feats["postal_match"] = (postal_both & (p1 == p2)).astype(float)
    feats["name_first_token_match"] = ((f1 == f2) & (f1 != "")).astype(float)

    # --- the two features with no convenient vectorized form ---
    seq_ratio = np.empty(len(pairs), dtype=float)
    prefix_len = np.empty(len(pairs), dtype=float)
    for i, (a, b) in enumerate(zip(n1, n2)):
        seq_ratio[i] = SequenceMatcher(None, a, b).ratio() if (a or b) else 0.0
        prefix_len[i] = len(_common_prefix(a, b))
    feats["name_seq_ratio"] = seq_ratio
    feats["name_common_prefix"] = prefix_len

    return feats[FEATURE_COLUMNS]


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]
