"""Shared TSV loading / writing helpers, kept in one place so the exact
column names and separators always match the challenge's required format."""
from typing import Dict, Set

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for col in SOURCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SOURCE_COLUMNS]


def load_ground_truth(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_id_list_tsv(mapping: Dict[str, Set[str]], id_col: str, list_col: str, path: str) -> None:
    """Write a {s1_id: {matched/candidate ids}} mapping as a two-column TSV.

    One row per key, comma-joined IDs (sorted for determinism), empty string
    for entities with no matches — matching the required output format.
    """
    rows = [
        {id_col: s1, list_col: ",".join(sorted(ids))}
        for s1, ids in mapping.items()
    ]
    out = pd.DataFrame(rows, columns=[id_col, list_col])
    out.to_csv(path, sep="\t", index=False)
