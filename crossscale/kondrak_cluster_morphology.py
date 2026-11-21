from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


from crossscale.partition.hdbscan import (  # noqa: F401
    filter_valid_clusters,
)

from crossscale.tokens import (  # noqa: F401
    attach_token_str_from_tokenizer,
)


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


from crossscale.metrics.orthographic import (  # noqa: F401
    preprocess_token_for_kondrak,
    add_processed_token_column,
    _positional_bigram_cost_numba,
    positional_bigram_cost,
    _string_to_codepoints,
    _kondrak_trigram_ndist_numba,
    kondrak_trigram_ndist,
    kondrak_trigram_similarity,
    compute_cluster_morphology_score,
    compute_cluster_level_morphology_df,
    compute_global_cluster_stats,
)


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
