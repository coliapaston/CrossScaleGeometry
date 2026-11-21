"""Permutation-control kernels: global and length-bucket token permutation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from crossscale.tokens import token_id_to_text


REQUIRED_CLUSTER_COLS = [
    "token_id",
    "cluster_id",
]


def filter_valid_cluster_df(cluster_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CLUSTER_COLS if c not in cluster_df.columns]
    if missing:
        raise KeyError(f"cluster_df missing required columns: {missing}")

    out = cluster_df.copy()
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="raise").astype(int)
    out["token_id"] = pd.to_numeric(out["token_id"], errors="raise").astype(int)
    out = out.loc[out["cluster_id"] != -1].copy()
    if out.empty:
        raise ValueError("No valid clustered tokens remain after filtering cluster_id == -1.")
    return out


def permute_token_ids(valid_cluster_df: pd.DataFrame, *, random_state: Optional[int] = None) -> np.ndarray:
    token_ids = valid_cluster_df["token_id"].to_numpy(dtype=int).copy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(token_ids)
    return token_ids


def build_permuted_cluster_df(valid_cluster_df: pd.DataFrame, permuted_token_ids: Sequence[int]) -> pd.DataFrame:
    permuted_token_ids = np.asarray(permuted_token_ids, dtype=int)
    if len(valid_cluster_df) != len(permuted_token_ids):
        raise ValueError(
            "Length mismatch between valid_cluster_df and permuted_token_ids: "
            f"{len(valid_cluster_df)} vs {len(permuted_token_ids)}"
        )

    out = valid_cluster_df.copy()
    out["token_id"] = permuted_token_ids
    return out


def token_id_to_token_text(
    token_id: int,
    *,
    tokenizer,
    model_name: str,
) -> str:
    return str(token_id_to_text(token_id, tokenizer=tokenizer, model_name=model_name))


def coarse_character_length_bucket(token_text: Any) -> str:
    length = len(str(token_text))
    if length <= 2:
        return "1-2"
    if length <= 5:
        return "3-5"
    if length <= 10:
        return "6-10"
    return "11+"


def attach_length_bucket_column(
    cluster_df: pd.DataFrame,
    *,
    tokenizer,
    model_name: str,
    token_text_cache: Dict[int, str] | None = None,
    token_id_col: str = "token_id",
    token_text_col: str = "token_str",
    bucket_col: str = "length_bucket",
) -> pd.DataFrame:
    if token_text_col not in cluster_df.columns and token_id_col not in cluster_df.columns:
        raise KeyError(
            f"cluster_df missing both token text column '{token_text_col}' and token id column '{token_id_col}'."
        )

    out = cluster_df.copy()

    if token_text_col not in out.columns:
        if tokenizer is None:
            raise ValueError(f"{token_text_col} not found and tokenizer is None.")
        if token_text_cache is None:
            token_text_cache = {}

        def _map_token_id(value: Any) -> str:
            token_id = int(value)
            if token_id not in token_text_cache:
                token_text_cache[token_id] = token_id_to_token_text(
                    token_id,
                    tokenizer=tokenizer,
                    model_name=model_name,
                )
            return token_text_cache[token_id]

        out[token_text_col] = out[token_id_col].map(_map_token_id)

    out[bucket_col] = out[token_text_col].map(coarse_character_length_bucket)
    return out


def permute_token_ids_within_length_buckets(
    valid_cluster_df: pd.DataFrame,
    *,
    tokenizer,
    model_name: str,
    random_state: int,
    token_text_cache: Dict[int, str] | None = None,
    token_id_col: str = "token_id",
    token_text_col: str = "token_str",
    bucket_col: str = "length_bucket",
) -> pd.DataFrame:
    import numpy as np

    if token_id_col not in valid_cluster_df.columns:
        raise KeyError(f"valid_cluster_df missing required column: {token_id_col}")

    out = attach_length_bucket_column(
        valid_cluster_df,
        tokenizer=tokenizer,
        model_name=model_name,
        token_text_cache=token_text_cache,
        token_id_col=token_id_col,
        token_text_col=token_text_col,
        bucket_col=bucket_col,
    )

    rng = np.random.default_rng(int(random_state))
    permuted_token_ids = out[token_id_col].to_numpy(dtype=int).copy()

    for bucket_value in sorted(out[bucket_col].dropna().unique()):
        bucket_mask = out[bucket_col].to_numpy() == bucket_value
        bucket_positions = np.flatnonzero(bucket_mask)
        bucket_token_ids = permuted_token_ids[bucket_positions].copy()
        rng.shuffle(bucket_token_ids)
        permuted_token_ids[bucket_positions] = bucket_token_ids

    result = valid_cluster_df.copy()
    result[token_id_col] = permuted_token_ids
    return result
