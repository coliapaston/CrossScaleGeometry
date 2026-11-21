"""Orthographic Similarity (O-S): Kondrak trigram N-DIST and cluster scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from numba import njit


# =========================================================
# 2) Token preprocessing
# =========================================================

def preprocess_token_for_kondrak(
    token: Any,
    *,
    affix_prefix: str = "##",
) -> str:
    """
    Fixed preprocessing for morphology scoring:
      - convert to string
      - case-insensitive via str.casefold()
      - prepend fixed literal prefix "##"
      - otherwise keep string as-is
    """
    s = str(token).casefold()
    return affix_prefix + s


def add_processed_token_column(
    cluster_df: pd.DataFrame,
    *,
    token_col: str = "token_str",
    processed_col: str = "token_proc",
    affix_prefix: str = "##",
) -> pd.DataFrame:
    """
    Add preprocessed token column:
        token_proc = "##" + token_str.casefold()
    """
    if token_col not in cluster_df.columns:
        raise KeyError(f"cluster_df missing required token column: {token_col}")

    out = cluster_df.copy()
    out[processed_col] = out[token_col].map(lambda x: preprocess_token_for_kondrak(x, affix_prefix=affix_prefix))
    return out


# =========================================================
# 3) Positional trigram cost + DP N-DIST
# =========================================================

@njit(cache=False)
def _positional_bigram_cost_numba(x_codes, y_codes, i, j):
    """Numba kernel for positional bigram substitution cost."""
    if i < 2 or j < 2:
        return 0.0 if x_codes[i - 1] == y_codes[j - 1] else 1.0

    mismatches = 0
    if x_codes[i - 2] != y_codes[j - 2]:
        mismatches += 1
    if x_codes[i - 1] != y_codes[j - 1]:
        mismatches += 1
    return mismatches / 2.0


def positional_bigram_cost(x_aff, y_aff, i, j):
    """
    Bigram cost ending at position i, j (1-based for original string)

    x_aff, y_aff: casefolded strings (NO affix)
    i, j: positions in original string (1-based)
    """
    return _positional_bigram_cost_numba(
        _string_to_codepoints(str(x_aff)),
        _string_to_codepoints(str(y_aff)),
        i,
        j,
    )


def _string_to_codepoints(text: str) -> np.ndarray:
    """Convert a Python string into a compact int32 codepoint array for Numba kernels."""
    return np.fromiter((ord(ch) for ch in text), dtype=np.int32)


@njit(cache=False)
def _kondrak_trigram_ndist_numba(x_codes, y_codes):
    k = len(x_codes)
    l = len(y_codes)

    d = np.zeros((k + 1, l + 1), dtype=np.float64)

    for i in range(k + 1):
        d[i, 0] = float(i)
    for j in range(l + 1):
        d[0, j] = float(j)

    for i in range(1, k + 1):
        for j in range(1, l + 1):
            sub_cost = _positional_bigram_cost_numba(x_codes, y_codes, i, j)

            deletion_cost = d[i - 1, j] + 1.0
            insertion_cost = d[i, j - 1] + 1.0
            substitution_cost = d[i - 1, j - 1] + sub_cost

            best_cost = deletion_cost
            if insertion_cost < best_cost:
                best_cost = insertion_cost
            if substitution_cost < best_cost:
                best_cost = substitution_cost

            d[i, j] = best_cost

    return d[k, l] / max(k, l)


def kondrak_trigram_ndist(
    x_raw: str,
    y_raw: str,
    *,
    affix_prefix: str = "##",
) -> float:
    """
    Compute normalized N-DIST using dynamic programming with:
      - n = 3
      - no affix; affix_prefix is accepted for compatibility and ignored
      - case-insensitive via casefold()
      - deletion cost = 1
      - insertion cost = 1
      - substitution cost = positional trigram cost
      - normalization by max(original_len_x, original_len_y)

    Notes:
    - DP runs over the original character positions.
    - Trigrams end at the current position of the casefolded strings.
    """
    x_raw = str(x_raw)
    y_raw = str(y_raw)

    k = len(x_raw)
    l = len(y_raw)

    # Defensive check: if empty strings somehow occur, avoid division by zero.
    if max(k, l) == 0:
        return 0.0

    x_aff = x_raw.casefold()
    y_aff = y_raw.casefold()

    x_codes = _string_to_codepoints(x_aff)
    y_codes = _string_to_codepoints(y_aff)

    return float(_kondrak_trigram_ndist_numba(x_codes, y_codes))


def kondrak_trigram_similarity(
    x_raw: str,
    y_raw: str,
    *,
    affix_prefix: str = "##",
) -> float:
    """
    Convert distance to similarity:
        S(X, Y) = 1 - N-DIST(X, Y)
    """
    nd = kondrak_trigram_ndist(x_raw, y_raw, affix_prefix=affix_prefix)
    return 1.0 - nd


# =========================================================
# 4) Cluster-level scoring
# =========================================================

def compute_cluster_morphology_score(
    tokens: Sequence[Any],
    *,
    affix_prefix: str = "##",
    exact_token_limit: int = 512,
    sample_size: int = 128,
    n_repeats: int = 10,
    random_state: int = 1813382118,
) -> float:
    """
    Compute cluster-level morphology score:

        M(c) = mean pairwise similarity within the cluster

    using unordered pairs only (i < j).

    Assumes cluster size >= 2.

    Clusters with at most exact_token_limit tokens use all unordered pairs.
    Larger clusters use n_repeats independent token subsamples, each drawn
    without replacement, and return the mean of the subsample scores.
    """
    n = len(tokens)
    if n < 2:
        raise ValueError(f"Cluster must contain at least 2 tokens, got n={n}")

    if n > exact_token_limit:
        import hashlib

        if exact_token_limit < 2:
            raise ValueError("exact_token_limit must be at least 2")
        if sample_size < 2:
            raise ValueError("sample_size must be at least 2")
        if sample_size > exact_token_limit:
            raise ValueError("sample_size must not exceed exact_token_limit")
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
        if random_state < 0 or random_state > np.iinfo(np.uint64).max:
            raise ValueError("random_state must fit in an unsigned 64-bit integer")

        sorted_token_list = sorted(str(token) for token in tokens)
        digest = hashlib.blake2b(digest_size=8)
        digest.update(
            int(random_state).to_bytes(
                8,
                byteorder="little",
                signed=False,
            )
        )
        for token in sorted_token_list:
            encoded = token.encode("utf-8", errors="surrogatepass")
            digest.update(
                len(encoded).to_bytes(
                    8,
                    byteorder="little",
                    signed=False,
                )
            )
            digest.update(encoded)

        cluster_seed = int.from_bytes(
            digest.digest(),
            byteorder="little",
            signed=False,
        )
        rng = np.random.default_rng(cluster_seed)
        repeat_scores: List[float] = []

        for _ in range(n_repeats):
            sampled_indices = rng.choice(
                n,
                size=sample_size,
                replace=False,
            )
            sampled_tokens = [
                sorted_token_list[int(index)]
                for index in sampled_indices
            ]
            repeat_scores.append(
                compute_cluster_morphology_score(
                    sampled_tokens,
                    affix_prefix=affix_prefix,
                    exact_token_limit=exact_token_limit,
                    sample_size=sample_size,
                    n_repeats=n_repeats,
                    random_state=random_state,
                )
            )

        return float(np.mean(repeat_scores))

    sims: List[float] = []
    token_list = [str(t) for t in tokens]

    for i in range(n):
        xi = token_list[i]
        for j in range(i + 1, n):
            yj = token_list[j]
            sims.append(kondrak_trigram_similarity(xi, yj, affix_prefix=affix_prefix))

    return float(np.mean(sims))


def compute_cluster_level_morphology_df(
    cluster_df: pd.DataFrame,
    *,
    cluster_id_col: str = "cluster_id",
    token_col: str = "token_str",
    affix_prefix: str = "##",
    sort_by_cluster_id: bool = True,
) -> pd.DataFrame:
    """
    Compute cluster-level outputs only:

      - cluster_id
      - cluster_size
      - M(c)

    Noise cluster must already be removed before calling this function.
    """
    required = [cluster_id_col, token_col]
    missing = [c for c in required if c not in cluster_df.columns]
    if missing:
        raise KeyError(f"cluster_df missing required columns: {missing}")

    grouped = cluster_df.groupby(cluster_id_col, sort=sort_by_cluster_id)

    rows: List[Dict[str, Any]] = []
    for cluster_id, group in grouped:
        tokens = group[token_col].tolist()
        cluster_size = len(tokens)

        # You stated this won't happen for valid clusters, so keep it strict.
        if cluster_size < 2:
            raise ValueError(f"Valid cluster has size < 2: cluster_id={cluster_id}, size={cluster_size}")

        m_c = compute_cluster_morphology_score(
            tokens,
            affix_prefix=affix_prefix,
        )

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_size),
                "M(c)": float(m_c),
            }
        )

    return pd.DataFrame(rows, columns=["cluster_id", "cluster_size", "M(c)"])


def compute_global_cluster_stats(
    cluster_score_df: pd.DataFrame,
    *,
    score_col: str = "M(c)",
    ddof: int = 0,
) -> Dict[str, float]:
    """
    Equal-weight global aggregation over clusters.

    Returns:
        {
            "global_mean": ...,
            "global_std": ...,
            "n_clusters": ...
        }
    """
    if score_col not in cluster_score_df.columns:
        raise KeyError(f"cluster_score_df missing score column: {score_col}")

    scores = pd.to_numeric(cluster_score_df[score_col], errors="raise").to_numpy(dtype=float)
    if len(scores) == 0:
        raise ValueError("No valid clusters available for global aggregation.")

    return {
        "global_mean": float(np.mean(scores)),
        "global_std": float(np.std(scores, ddof=ddof)),
        "n_clusters": int(len(scores)),
    }
