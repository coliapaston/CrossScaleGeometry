"""Multi-Script Entropy (MS-E): Unicode script labels and cluster entropy."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

import pandas as pd
import unicodedataplus as udp

from crossscale.tokens import token_ids_to_text


OTHER_LABEL = "Other"
MIXED_LABEL = "Mixed"
RAW_TO_OTHER = {"Common", "Inherited", "Unknown"}
NOISE_CLUSTER_ID = -1


# =========================================================
# Token decoding / input normalization
# =========================================================

def decode_token_ids_to_strings(
    token_ids: Sequence[int],
    tokenizer: Any,
    *,
    model_name: str,
) -> List[str]:
    return token_ids_to_text(token_ids, tokenizer=tokenizer, model_name=model_name)


def standardize_cluster_input_df(
    cluster_df: pd.DataFrame,
    *,
    model_name: str,
    tokenizer: Any = None,
    cluster_id_col: str = "cluster_id",
    token_str_col: str = "token_str",
    token_id_col: str = "token_id",
) -> pd.DataFrame:
    if cluster_id_col not in cluster_df.columns:
        raise KeyError(f"cluster_df missing required column: {cluster_id_col}")

    out = cluster_df.copy()

    if token_str_col in out.columns:
        out[token_str_col] = out[token_str_col].astype(str)
    elif token_id_col in out.columns:
        if tokenizer is None:
            raise ValueError(
                "cluster_df has token_id but no token_str; tokenizer must be provided."
            )
        token_ids = pd.to_numeric(out[token_id_col], errors="raise").astype(int).tolist()
        out[token_str_col] = decode_token_ids_to_strings(
            token_ids,
            tokenizer=tokenizer,
            model_name=model_name,
        )
    else:
        raise KeyError(
            "cluster_df must contain either token_str or token_id. "
            f"Found columns: {list(out.columns)}"
        )

    out[cluster_id_col] = pd.to_numeric(out[cluster_id_col], errors="raise").astype(int)

    return out[[cluster_id_col, token_str_col]].copy()


# =========================================================
# Character-level script utilities
# =========================================================

def raw_char_script(ch: str) -> str:
    if not isinstance(ch, str) or len(ch) != 1:
        raise ValueError(f"raw_char_script expects a single character, got: {repr(ch)}")
    return udp.script(ch)


def normalized_char_script(ch: str) -> str:
    raw = raw_char_script(ch)
    if raw in RAW_TO_OTHER:
        return OTHER_LABEL
    return raw


# =========================================================
# Token-level script classification
# =========================================================

def token_script_counts(token: str) -> Dict[str, int]:
    if token is None:
        return {OTHER_LABEL: 1}

    if not isinstance(token, str):
        token = str(token)

    if token == "":
        return {OTHER_LABEL: 1}

    counter: Counter[str] = Counter()
    for ch in token:
        counter[normalized_char_script(ch)] += 1

    if not counter:
        counter[OTHER_LABEL] += 1

    return dict(counter)


def dominant_script(token: str) -> str:
    counts = token_script_counts(token)
    if not counts:
        return OTHER_LABEL

    max_count = max(counts.values())
    winners = [script for script, count in counts.items() if count == max_count]

    if len(winners) == 1:
        return winners[0]

    if OTHER_LABEL in winners:
        winners_wo_other = [script for script in winners if script != OTHER_LABEL]

        if len(winners_wo_other) == 0:
            return OTHER_LABEL
        if len(winners_wo_other) == 1:
            return winners_wo_other[0]
        return MIXED_LABEL

    return MIXED_LABEL


def annotate_token_scripts(
    df: pd.DataFrame,
    *,
    token_col: str = "token_str",
    script_col: str = "script",
) -> pd.DataFrame:
    if token_col not in df.columns:
        raise KeyError(f"DataFrame missing token column: {token_col}")

    out = df.copy()
    out[script_col] = out[token_col].map(dominant_script)
    return out


# =========================================================
# Model-level cache
# =========================================================

def get_full_vocab_token_strings(
    tokenizer: Any,
    *,
    model_name: str,
) -> List[str]:
    if tokenizer is None:
        raise ValueError("tokenizer must not be None for full-vocab script cache construction")

    if hasattr(tokenizer, "vocab_size"):
        vocab_size = int(tokenizer.vocab_size)
        token_ids = list(range(vocab_size))
        return decode_token_ids_to_strings(
            token_ids,
            tokenizer=tokenizer,
            model_name=model_name,
        )

    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        return [inv_vocab[i] for i in range(len(inv_vocab))]

    raise ValueError("Unable to extract full vocab from tokenizer")


def build_global_script_cache_from_tokenizer(
    tokenizer: Any,
    *,
    model_name: str,
) -> Dict[str, Any]:
    """
    Build model-level script cache once per tokenizer/model.

    Returns:
        {
            "vocab_df": DataFrame[token_id, token_str, script],
            "global_script_set": list[str],
            "global_script_set_size": int,
        }
    """
    vocab_tokens = get_full_vocab_token_strings(
        tokenizer,
        model_name=model_name,
    )
    vocab_df = pd.DataFrame(
        {
            "token_id": list(range(len(vocab_tokens))),
            "token_str": vocab_tokens,
        }
    )
    vocab_df = annotate_token_scripts(vocab_df, token_col="token_str", script_col="script")

    global_script_set = list(set(vocab_df["script"].tolist()))

    return {
        "vocab_df": vocab_df,
        "global_script_set": global_script_set,
        "global_script_set_size": len(global_script_set),
    }


# =========================================================
# Cluster-level counts / distributions
# =========================================================

def filter_noise_clusters(
    df: pd.DataFrame,
    *,
    cluster_id_col: str = "cluster_id",
    noise_cluster_id: int = NOISE_CLUSTER_ID,
) -> pd.DataFrame:
    if cluster_id_col not in df.columns:
        raise KeyError(f"DataFrame missing cluster_id column: {cluster_id_col}")
    return df.loc[df[cluster_id_col] != noise_cluster_id].copy()


def build_cluster_script_count_df(
    token_df: pd.DataFrame,
    *,
    global_script_set: Sequence[str],
    cluster_id_col: str = "cluster_id",
    script_col: str = "script",
) -> pd.DataFrame:
    if cluster_id_col not in token_df.columns:
        raise KeyError(f"token_df missing cluster_id column: {cluster_id_col}")
    if script_col not in token_df.columns:
        raise KeyError(f"token_df missing script column: {script_col}")

    scripts = list(global_script_set)
    rows: List[Dict[str, Any]] = []

    for cluster_id, group in token_df.groupby(cluster_id_col, sort=True):
        counts = group[script_col].value_counts(dropna=False).to_dict()
        row = {cluster_id_col: int(cluster_id)}
        for s in scripts:
            row[s] = int(counts.get(s, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def cluster_script_distribution_from_count_row(
    count_row: pd.Series,
    *,
    global_script_set: Sequence[str],
) -> Dict[str, float]:
    scripts = list(global_script_set)
    counts = [int(count_row[s]) for s in scripts]
    total = sum(counts)

    if total <= 0:
        raise ValueError("Encountered a cluster row with total script count <= 0")

    return {s: int(count_row[s]) / total for s in scripts}


# =========================================================
# Entropy
# =========================================================

def entropy_from_distribution(
    dist: Dict[str, float],
) -> float:
    H = 0.0
    for p in dist.values():
        if p > 0.0:
            H -= p * math.log(p)
    return float(H)


def normalized_entropy(
    H: float,
    *,
    global_script_set_size: int,
) -> float:
    if global_script_set_size <= 1:
        return 0.0
    return float(H / math.log(global_script_set_size))


def compute_cluster_entropy_df(
    cluster_script_count_df: pd.DataFrame,
    *,
    global_script_set: Sequence[str],
    cluster_id_col: str = "cluster_id",
    warn_on_degenerate_global_set: bool = True,
) -> pd.DataFrame:
    scripts = list(global_script_set)
    global_script_set_size = len(scripts)

    if warn_on_degenerate_global_set and global_script_set_size <= 1:
        print(
            "[WARNING] global_script_set_size <= 1. "
            "All normalized entropies will be set to 0."
        )

    rows: List[Dict[str, Any]] = []

    for _, row in cluster_script_count_df.iterrows():
        dist = cluster_script_distribution_from_count_row(
            row,
            global_script_set=scripts,
        )
        H = entropy_from_distribution(dist)
        H_norm = normalized_entropy(H, global_script_set_size=global_script_set_size)

        out_row = {cluster_id_col: int(row[cluster_id_col])}
        for s in scripts:
            out_row[s] = int(row[s])
        out_row["H"] = float(H)
        out_row["H_norm"] = float(H_norm)
        rows.append(out_row)

    return pd.DataFrame(rows)


# =========================================================
# Partition-level aggregation
# =========================================================

def partition_entropy_stats(
    cluster_entropy_df: pd.DataFrame,
    *,
    raw_entropy_col: str = "H",
    norm_entropy_col: str = "H_norm",
) -> Dict[str, float]:
    """
    Equal-weight aggregation over clusters.

    Returns:
        {
            "num_clusters": ...,
            "mean_H": ...,
            "std_H": ...,
            "mean_H_norm": ...,
            "std_H_norm": ...
        }
    """
    for col in [raw_entropy_col, norm_entropy_col]:
        if col not in cluster_entropy_df.columns:
            raise KeyError(f"cluster_entropy_df missing entropy column: {col}")

    raw_values = pd.to_numeric(
        cluster_entropy_df[raw_entropy_col], errors="raise"
    ).to_numpy(dtype=float)

    norm_values = pd.to_numeric(
        cluster_entropy_df[norm_entropy_col], errors="raise"
    ).to_numpy(dtype=float)

    if len(raw_values) == 0:
        raise ValueError("No valid non-noise clusters available for partition aggregation.")

    return {
        "num_clusters": int(len(raw_values)),
        "mean_H": float(raw_values.mean()),
        "std_H": float(raw_values.std(ddof=0)),
        "mean_H_norm": float(norm_values.mean()),
        "std_H_norm": float(norm_values.std(ddof=0)),
    }
