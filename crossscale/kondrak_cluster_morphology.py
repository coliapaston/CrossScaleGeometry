from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numba import njit


# =========================================================
# 1) Summary selection / cluster loading
# =========================================================

def load_summary_df(
    out_root: str | Path,
    model_name: str,
    space_name: str,
    summary_filename: str = "summary.csv",
) -> pd.DataFrame:
    """
    Load summary.csv for a given model/space.

    Path:
        {out_root}/{model_name}/{space_name}/{summary_filename}
    """
    summary_path = Path(out_root) / model_name / space_name / summary_filename
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    return pd.read_csv(summary_path)


def _normalize_bool_like(x: Any) -> Any:
    """
    Normalize common CSV bool-like values to Python bool when possible.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1"}:
            return True
        if s in {"false", "0"}:
            return False
    return x

def attach_token_str_from_tokenizer(
    cluster_df: pd.DataFrame,
    *,
    tokenizer,
    model_name: str,
    token_id_col: str = "token_id",
    token_str_col: str = "token_str",
) -> pd.DataFrame:
    """
    Use tokenizer to map token_id -> token_str.

    tokenizer must support:
        tokenizer.convert_ids_to_tokens(int)
    """
    if token_str_col in cluster_df.columns:
        return cluster_df

    if token_id_col not in cluster_df.columns:
        raise KeyError(
            f"cluster_df missing both '{token_str_col}' and '{token_id_col}'"
        )

    df = cluster_df.copy()

    use_gpt_decode = str(model_name).startswith("gpt")

    def _map_id(x):
        tid = int(x)
        if use_gpt_decode:
            return tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        return tokenizer.convert_ids_to_tokens(tid)

    df[token_str_col] = df[token_id_col].map(_map_id)

    return df

def select_summary_row(
    summary_df: pd.DataFrame,
    *,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    epsilon_atol: float = 1e-12,
) -> pd.Series:
    """
    Filter summary_df to a unique row using the fixed parameter set.

    Raises:
        ValueError: if zero or multiple rows match.
    """
    required_cols = [
        "pca_dim",
        "l2_norm",
        "min_cluster_size",
        "min_samples",
        "metric",
        "cluster_selection_method",
        "cluster_selection_epsilon",
        "cluster_csv_path",
        "meta_json_path",
    ]
    missing = [c for c in required_cols if c not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    l2_series = summary_df["l2_norm"].map(_normalize_bool_like)
    eps_series = pd.to_numeric(summary_df["cluster_selection_epsilon"], errors="raise")

    mask = (
        (pd.to_numeric(summary_df["pca_dim"], errors="raise") == int(pca_dim))
        & (l2_series == bool(l2_norm))
        & (pd.to_numeric(summary_df["min_cluster_size"], errors="raise") == int(min_cluster_size))
        & (pd.to_numeric(summary_df["min_samples"], errors="raise") == int(min_samples))
        & (summary_df["metric"].astype(str) == str(metric))
        & (summary_df["cluster_selection_method"].astype(str) == str(cluster_selection_method))
        & (np.isclose(eps_series.to_numpy(dtype=float), float(cluster_selection_epsilon), atol=epsilon_atol))
    )

    matched = summary_df.loc[mask]
    if len(matched) == 0:
        raise ValueError("No row matched the requested parameter set.")
    if len(matched) > 1:
        raise ValueError(f"Expected exactly one matched row, got {len(matched)} rows.")

    return matched.iloc[0]


def load_cluster_df_from_summary_row(
    summary_row: pd.Series,
    *,
    print_columns: bool = True,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Load cluster_df from `cluster_csv_path` stored in a selected summary row.

    Returns:
        cluster_df, cluster_csv_path, meta_json_path
    """
    cluster_csv_path = str(summary_row["cluster_csv_path"])
    meta_json_path = str(summary_row["meta_json_path"])

    cluster_df = pd.read_csv(cluster_csv_path)

    if print_columns:
        print("cluster_df.columns =", list(cluster_df.columns))

    return cluster_df, cluster_csv_path, meta_json_path


def filter_valid_clusters(
    cluster_df: pd.DataFrame,
    cluster_id_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Exclude HDBSCAN noise cluster: cluster_id == -1
    """
    if cluster_id_col not in cluster_df.columns:
        raise KeyError(f"cluster_df missing required column: {cluster_id_col}")

    out = cluster_df.copy()
    out[cluster_id_col] = pd.to_numeric(out[cluster_id_col], errors="raise").astype(int)
    out = out.loc[out[cluster_id_col] != -1].copy()
    return out


def load_partition_cluster_df(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    summary_filename: str = "summary.csv",
    print_columns: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    End-to-end partition lookup:
        summary.csv -> unique summary row -> cluster_df -> remove noise

    Returns:
        valid_cluster_df, selected_summary_row, full_summary_df
    """
    summary_df = load_summary_df(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        summary_filename=summary_filename,
    )

    summary_row = select_summary_row(
        summary_df,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    cluster_df, _, _ = load_cluster_df_from_summary_row(
        summary_row,
        print_columns=print_columns,
    )

    valid_cluster_df = filter_valid_clusters(cluster_df, cluster_id_col="cluster_id")
    return valid_cluster_df, summary_row, summary_df


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
      - fixed affix "##"
      - case-insensitive via casefold()
      - deletion cost = 1
      - insertion cost = 1
      - substitution cost = positional trigram cost
      - normalization by max(original_len_x, original_len_y)

    Notes:
    - DP runs over the original character positions.
    - Trigrams are taken from the affixed strings and end at the current position.
    - Raw strings are normalized internally using the fixed preprocessing.
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


def make_global_summary_row(
    *,
    model_name: str,
    space_name: str,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    global_mean: float,
    global_std: float,
    n_clusters: int,
) -> pd.DataFrame:
    """
    Create a one-row DataFrame for later concatenation across partitions.
    """
    row = {
        "model_name": model_name,
        "space_name": space_name,
        "pca_dim": int(pca_dim),
        "l2_norm": bool(l2_norm),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "metric": str(metric),
        "cluster_selection_method": str(cluster_selection_method),
        "cluster_selection_epsilon": float(cluster_selection_epsilon),
        "global_mean": float(global_mean),
        "global_std": float(global_std),
        "n_clusters": int(n_clusters),
    }
    return pd.DataFrame([row])


# =========================================================
# 5) Orchestration helpers
# =========================================================

def run_kondrak_morphology_for_partition(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    token_col: str = "token_str",
    cluster_id_col: str = "cluster_id",
    affix_prefix: str = "##",
    ddof: int = 0,
    summary_filename: str = "summary.csv",
    print_columns: bool = True,
    save_cluster_csv: bool = False,
    tokenizer=None,
    cluster_csv_outpath: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end run for one partition.

    Returns:
        cluster_score_df, global_summary_row_df

    cluster_score_df columns:
        - cluster_id
        - cluster_size
        - M(c)

    global_summary_row_df columns:
        - model_name
        - space_name
        - pca_dim
        - l2_norm
        - min_cluster_size
        - min_samples
        - metric
        - cluster_selection_method
        - cluster_selection_epsilon
        - global_mean
        - global_std
        - n_clusters
    """
    valid_cluster_df, _, _ = load_partition_cluster_df(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        summary_filename=summary_filename,
        print_columns=print_columns,
    )

    # =========================================================
    # token_str fallback via tokenizer
    # =========================================================
    if token_col not in valid_cluster_df.columns:
        if tokenizer is None:
            raise ValueError(
                f"{token_col} not found and tokenizer is None"
            )
        valid_cluster_df = attach_token_str_from_tokenizer(
            valid_cluster_df,
            tokenizer=tokenizer,
            model_name=model_name,
            token_id_col="token_id",
            token_str_col=token_col,
        )

    cluster_score_df = compute_cluster_level_morphology_df(
        valid_cluster_df,
        cluster_id_col=cluster_id_col,
        token_col=token_col,
        affix_prefix=affix_prefix,
    )

    global_stats = compute_global_cluster_stats(
        cluster_score_df,
        score_col="M(c)",
        ddof=ddof,
    )

    global_summary_row_df = make_global_summary_row(
        model_name=model_name,
        space_name=space_name,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        global_mean=global_stats["global_mean"],
        global_std=global_stats["global_std"],
        n_clusters=global_stats["n_clusters"],
    )

    if save_cluster_csv:
        if cluster_csv_outpath is None:
            raise ValueError("cluster_csv_outpath must be provided when save_cluster_csv=True")
        outpath = Path(cluster_csv_outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        cluster_score_df.to_csv(outpath, index=False)

    return cluster_score_df, global_summary_row_df


# =========================================================
# 6) Convenience helper for many partitions
# =========================================================

def concat_global_summary_rows(rows: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate one-row summary DataFrames across many partitions.
    """
    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "model_name",
                "space_name",
                "pca_dim",
                "l2_norm",
                "min_cluster_size",
                "min_samples",
                "metric",
                "cluster_selection_method",
                "cluster_selection_epsilon",
                "global_mean",
                "global_std",
                "n_clusters",
            ]
        )
    return pd.concat(rows, axis=0, ignore_index=True)


# =========================================================
# 6) Random-PCA path-separated pipeline
#    - keep full pipeline untouched
#    - only path logic differs
#    - morphology compute functions are reused
# =========================================================

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def load_random_pca_summary_df(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    pca_seed: int,
    summary_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load random-PCA summary DataFrame from:

        {out_root}/{model_name}/{space_name}/random/seed_{pca_seed}/
            summary_random_pca_seed_{pca_seed}.csv

    If `summary_filename` is provided, use it directly inside that directory.
    """
    random_dir = (
        Path(out_root)
        / model_name
        / space_name
        / "random"
        / f"seed_{pca_seed}"
    )

    if summary_filename is None:
        summary_path = random_dir / f"summary_random_pca_seed_{pca_seed}.csv"
    else:
        summary_path = random_dir / summary_filename

    if not summary_path.exists():
        raise FileNotFoundError(f"Random PCA summary not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    return summary_df


def select_random_pca_summary_row(
    summary_df: pd.DataFrame,
    *,
    pca_seed: int,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
) -> pd.Series:
    """
    Select exactly one row from random-PCA summary_df by parameter match.
    Mirrors the full-PCA selector behavior as closely as possible.
    """
    required_cols = [
        "pca_mode",
        "pca_seed",
        "pca_dim",
        "l2_norm",
        "min_cluster_size",
        "min_samples",
        "metric",
        "cluster_selection_method",
        "cluster_selection_epsilon",
        "cluster_csv_path",
        "meta_json_path",
    ]
    missing = [c for c in required_cols if c not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    # 单 partition selector 只接受标量 pca_dim
    if isinstance(pca_dim, (list, tuple, set)):
        raise TypeError(
            "select_random_pca_summary_row expects a scalar pca_dim, "
            f"but got {type(pca_dim).__name__}: {pca_dim}"
        )

    pca_dim = int(pca_dim)
    pca_seed = int(pca_seed)
    min_cluster_size = int(min_cluster_size)
    min_samples = int(min_samples)
    cluster_selection_epsilon = float(cluster_selection_epsilon)
    l2_norm = bool(l2_norm)

    l2_norm_col = summary_df["l2_norm"].map(_normalize_bool_like)

    mask = (
        (summary_df["pca_mode"].astype(str) == "randomized")
        & (pd.to_numeric(summary_df["pca_seed"], errors="raise").astype(int) == pca_seed)
        & (pd.to_numeric(summary_df["pca_dim"], errors="raise").astype(int) == pca_dim)
        & (l2_norm_col == l2_norm)
        & (pd.to_numeric(summary_df["min_cluster_size"], errors="raise").astype(int) == min_cluster_size)
        & (pd.to_numeric(summary_df["min_samples"], errors="raise").astype(int) == min_samples)
        & (summary_df["metric"].astype(str) == str(metric))
        & (summary_df["cluster_selection_method"].astype(str) == str(cluster_selection_method))
        & (
            pd.to_numeric(summary_df["cluster_selection_epsilon"], errors="raise").astype(float)
            == cluster_selection_epsilon
        )
    )

    matched = summary_df.loc[mask]
    if len(matched) == 0:
        raise ValueError("No row matched the requested random-PCA parameter set.")
    if len(matched) > 1:
        raise ValueError(f"Expected exactly one matched row, got {len(matched)} rows.")

    return matched.iloc[0]


def load_random_pca_cluster_df_from_summary_row(
    summary_row: pd.Series,
    *,
    print_columns: bool = True,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Load cluster_df from `cluster_csv_path` stored in a selected random-PCA summary row.

    Returns:
        cluster_df, cluster_csv_path, meta_json_path
    """
    cluster_csv_path = str(summary_row["cluster_csv_path"])
    meta_json_path = str(summary_row["meta_json_path"])

    if not Path(cluster_csv_path).exists():
        raise FileNotFoundError(f"cluster_csv_path not found: {cluster_csv_path}")
    if not Path(meta_json_path).exists():
        raise FileNotFoundError(f"meta_json_path not found: {meta_json_path}")

    cluster_df = pd.read_csv(cluster_csv_path)

    if print_columns:
        print("cluster_df.columns =", list(cluster_df.columns))

    return cluster_df, cluster_csv_path, meta_json_path


def filter_valid_random_pca_clusters(
    cluster_df: pd.DataFrame,
    cluster_id_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Exclude HDBSCAN noise cluster: cluster_id == -1
    """
    if cluster_id_col not in cluster_df.columns:
        raise KeyError(f"cluster_df missing required column: {cluster_id_col}")

    out = cluster_df.copy()
    out[cluster_id_col] = pd.to_numeric(out[cluster_id_col], errors="raise").astype(int)
    out = out.loc[out[cluster_id_col] != -1].copy()
    return out


def load_random_pca_partition_cluster_df(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    pca_seed: int,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    summary_filename: Optional[str] = None,
    print_columns: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    End-to-end random-PCA partition lookup:
        random summary csv -> unique summary row -> cluster_df -> remove noise

    Returns:
        valid_cluster_df, selected_summary_row, full_summary_df
    """
    summary_df = load_random_pca_summary_df(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
        summary_filename=summary_filename,
    )

    summary_row = select_random_pca_summary_row(
        summary_df,
        pca_seed=pca_seed,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    cluster_df, _, _ = load_random_pca_cluster_df_from_summary_row(
        summary_row,
        print_columns=print_columns,
    )

    valid_cluster_df = filter_valid_random_pca_clusters(cluster_df, cluster_id_col="cluster_id")
    return valid_cluster_df, summary_row, summary_df


def make_random_pca_global_summary_row(
    *,
    model_name: str,
    space_name: str,
    pca_seed: int,
    pca_source_dim: int,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    global_mean: float,
    global_std: float,
    n_clusters: int,
) -> pd.DataFrame:
    """
    Create a one-row DataFrame for later concatenation across random-PCA partitions.
    """
    row = {
        "model_name": model_name,
        "space_name": space_name,
        "pca_mode": "random",
        "pca_seed": int(pca_seed),
        "pca_source_dim": int(pca_source_dim),
        "pca_dim": int(pca_dim),
        "l2_norm": bool(l2_norm),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "metric": str(metric),
        "cluster_selection_method": str(cluster_selection_method),
        "cluster_selection_epsilon": float(cluster_selection_epsilon),
        "global_mean": float(global_mean),
        "global_std": float(global_std),
        "n_clusters": int(n_clusters),
    }
    return pd.DataFrame([row])


def _default_random_pca_cluster_csv_outpath(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    pca_seed: int,
    pca_dim: int,
) -> Path:
    """
    Keep the full-PCA filename rule, only insert:
        /random/seed_{pca_seed}/
    """
    return (
        Path(out_root)
        / model_name
        / space_name
        / "random"
        / f"seed_{pca_seed}"
        / f"kondrak_clusters_pca{pca_dim}.csv"
    )


def run_kondrak_morphology_for_random_pca_partition(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    pca_seed: int,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    token_col: str = "token_str",
    cluster_id_col: str = "cluster_id",
    affix_prefix: str = "##",
    ddof: int = 0,
    summary_filename: Optional[str] = None,
    print_columns: bool = True,
    save_cluster_csv: bool = False,
    tokenizer=None,
    cluster_csv_outpath: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end run for one random-PCA partition.

    Returns:
        cluster_score_df, global_summary_row_df
    """
    valid_cluster_df, summary_row, _ = load_random_pca_partition_cluster_df(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        summary_filename=summary_filename,
        print_columns=print_columns,
    )

    # mirror original behavior: if token_str is missing, recover it from token_id via tokenizer
    if token_col not in valid_cluster_df.columns:
        if tokenizer is None:
            raise KeyError(
                f"valid_cluster_df missing token column: {token_col}. "
                "Pass tokenizer=... to recover token_str from token_id."
            )
        if "token_id" not in valid_cluster_df.columns:
            raise KeyError(
                f"valid_cluster_df missing both token column '{token_col}' and token_id."
            )
        valid_cluster_df = attach_token_str_from_tokenizer(
            valid_cluster_df,
            tokenizer=tokenizer,
            model_name=model_name,
            token_id_col="token_id",
            token_str_col=token_col,
        )

    cluster_score_df = compute_cluster_level_morphology_df(
        valid_cluster_df,
        cluster_id_col=cluster_id_col,
        token_col=token_col,
        affix_prefix=affix_prefix,
        sort_by_cluster_id=True,
    )

    stats = compute_global_cluster_stats(
        cluster_score_df,
        score_col="M(c)",
        ddof=ddof,
    )

    pca_source_dim = int(summary_row["pca_source_dim"]) if "pca_source_dim" in summary_row.index else int(pca_dim)

    global_summary_row_df = make_random_pca_global_summary_row(
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
        pca_source_dim=pca_source_dim,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        global_mean=stats["global_mean"],
        global_std=stats["global_std"],
        n_clusters=stats["n_clusters"],
    )

    if save_cluster_csv:
        if cluster_csv_outpath is None:
            cluster_csv_outpath = _default_random_pca_cluster_csv_outpath(
                out_root=out_root,
                model_name=model_name,
                space_name=space_name,
                pca_seed=pca_seed,
                pca_dim=pca_dim,
            )
        cluster_csv_outpath = Path(cluster_csv_outpath)
        cluster_csv_outpath.parent.mkdir(parents=True, exist_ok=True)
        cluster_score_df.to_csv(cluster_csv_outpath, index=False)

    return cluster_score_df, global_summary_row_df


def run_kondrak_morphology_for_random_pca_partitions(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    pca_seed_list: Sequence[int],
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    token_col: str = "token_str",
    cluster_id_col: str = "cluster_id",
    affix_prefix: str = "##",
    ddof: int = 0,
    summary_filename: Optional[str] = None,
    print_columns: bool = True,
    save_cluster_csv: bool = False,
    tokenizer=None,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Batch wrapper over multiple random PCA seeds.

    Returns:
        seed_to_cluster_score_df, all_summary_df
    """
    seed_to_cluster_score_df: Dict[int, pd.DataFrame] = {}
    global_rows: List[pd.DataFrame] = []

    for pca_seed in pca_seed_list:
        cluster_score_df, global_summary_row_df = run_kondrak_morphology_for_random_pca_partition(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            pca_seed=pca_seed,
            pca_dim=pca_dim,
            l2_norm=l2_norm,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            token_col=token_col,
            cluster_id_col=cluster_id_col,
            affix_prefix=affix_prefix,
            ddof=ddof,
            summary_filename=summary_filename,
            print_columns=print_columns,
            save_cluster_csv=save_cluster_csv,
            tokenizer=tokenizer,
            cluster_csv_outpath=None,
        )
        seed_to_cluster_score_df[int(pca_seed)] = cluster_score_df
        global_rows.append(global_summary_row_df)

    all_summary_df = concat_global_summary_rows(global_rows)
    return seed_to_cluster_score_df, all_summary_df

# =========================================================
# 7) Permutation multi-seed pipeline
# =========================================================

def permutation_seed_summary_filename(perm_seed: int) -> str:
    return f"seeds/seed_{int(perm_seed)}/summary.csv"



def permutation_seed_morph_dir(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    perm_seed: int,
) -> Path:
    return Path(out_root) / model_name / space_name / "seeds" / f"seed_{int(perm_seed)}" / "morph"



def permutation_seed_morph_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    perm_seed: int,
    filename: str = "kondrak_global_summary.csv",
) -> Path:
    return Path(out_root) / model_name / space_name / "seeds" / f"seed_{int(perm_seed)}" / filename



def permutation_all_seed_morph_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    filename: str = "kondrak_global_summary_all_seeds.csv",
) -> Path:
    return Path(out_root) / model_name / space_name / filename



def permutation_mean_morph_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    filename: str = "kondrak_global_summary_mean.csv",
) -> Path:
    return Path(out_root) / model_name / space_name / filename



def aggregate_permutation_morphology_means(all_seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    if all_seed_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "space_name",
                "pca_dim",
                "l2_norm",
                "min_cluster_size",
                "min_samples",
                "metric",
                "cluster_selection_method",
                "cluster_selection_epsilon",
                "global_mean",
                "global_std",
                "n_clusters",
            ]
        )

    group_cols = [
        "model_name",
        "space_name",
        "pca_dim",
        "l2_norm",
        "min_cluster_size",
        "min_samples",
        "metric",
        "cluster_selection_method",
        "cluster_selection_epsilon",
    ]
    mean_df = (
        all_seed_summary_df
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(
            global_mean=("global_mean", "mean"),
            global_std=("global_std", "mean"),
            n_clusters=("n_clusters", "mean"),
        )
    )
    mean_df["n_clusters"] = mean_df["n_clusters"].round().astype(int)
    return mean_df.sort_values(["pca_dim"]).reset_index(drop=True)



def save_permutation_morphology_summary_df(summary_df: pd.DataFrame, outpath: str | Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outpath, index=False)
    return outpath



def run_kondrak_morphology_for_permutation_seeds(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    perm_seed_list: Sequence[int],
    pca_dim_list: Sequence[int],
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    token_col: str = "token_str",
    cluster_id_col: str = "cluster_id",
    affix_prefix: str = "##",
    ddof: int = 0,
    print_columns: bool = False,
    save_cluster_csv: bool = True,
    tokenizer=None,
) -> Dict[str, pd.DataFrame]:
    per_seed_summary: Dict[int, pd.DataFrame] = {}
    all_rows: List[pd.DataFrame] = []

    for perm_seed in perm_seed_list:
        perm_seed = int(perm_seed)
        global_rows: List[pd.DataFrame] = []

        for pca_dim in pca_dim_list:
            cluster_outpath = permutation_seed_morph_dir(
                out_root=out_root,
                model_name=model_name,
                space_name=space_name,
                perm_seed=perm_seed,
            ) / f"kondrak_clusters_pca_{int(pca_dim)}.csv"

            _, global_row = run_kondrak_morphology_for_partition(
                out_root=out_root,
                model_name=model_name,
                space_name=space_name,
                pca_dim=int(pca_dim),
                l2_norm=l2_norm,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                cluster_selection_epsilon=cluster_selection_epsilon,
                token_col=token_col,
                cluster_id_col=cluster_id_col,
                affix_prefix=affix_prefix,
                ddof=ddof,
                summary_filename=permutation_seed_summary_filename(perm_seed),
                print_columns=print_columns,
                save_cluster_csv=save_cluster_csv,
                tokenizer=tokenizer,
                cluster_csv_outpath=cluster_outpath,
            )
            global_row = global_row.copy()
            global_row["perm_seed"] = perm_seed
            global_rows.append(global_row)

        seed_summary_df = concat_global_summary_rows(global_rows)
        if not seed_summary_df.empty:
            seed_summary_df = seed_summary_df.sort_values(["pca_dim"]).reset_index(drop=True)
        seed_summary_path = permutation_seed_morph_summary_path(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            perm_seed=perm_seed,
        )
        save_permutation_morphology_summary_df(seed_summary_df, seed_summary_path)
        per_seed_summary[perm_seed] = seed_summary_df
        all_rows.append(seed_summary_df)

    if all_rows:
        all_seed_summary_df = pd.concat(all_rows, axis=0, ignore_index=True)
        all_seed_summary_df = all_seed_summary_df.sort_values(["perm_seed", "pca_dim"]).reset_index(drop=True)
    else:
        all_seed_summary_df = pd.DataFrame()

    mean_summary_df = aggregate_permutation_morphology_means(all_seed_summary_df)

    all_seed_path = permutation_all_seed_morph_summary_path(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    )
    mean_path = permutation_mean_morph_summary_path(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    )
    save_permutation_morphology_summary_df(all_seed_summary_df, all_seed_path)
    save_permutation_morphology_summary_df(mean_summary_df, mean_path)
    save_permutation_morphology_summary_df(
        mean_summary_df,
        Path(out_root) / model_name / space_name / "kondrak_global_summary.csv",
    )

    return {
        "all_seed_summary_df": all_seed_summary_df,
        "mean_summary_df": mean_summary_df,
    }
