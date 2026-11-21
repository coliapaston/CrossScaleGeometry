from __future__ import annotations

import json
import math
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd



# =========================================================
# Constants
# =========================================================

from crossscale.metrics.script_entropy import (  # noqa: F401
    MIXED_LABEL,
    NOISE_CLUSTER_ID,
    OTHER_LABEL,
    RAW_TO_OTHER,
)

SUMMARY_FILENAME = "summary.csv"


# =========================================================
# Path / summary resolution
# =========================================================

def get_summary_csv_path(
    out_root: str,
    model_name: str,
    space_name: str,
    summary_filename: str = SUMMARY_FILENAME,
) -> Path:
    return Path(out_root) / model_name / space_name / summary_filename


def load_summary_df(
    out_root: str,
    model_name: str,
    space_name: str,
    summary_filename: str = SUMMARY_FILENAME,
) -> pd.DataFrame:
    summary_path = get_summary_csv_path(
        out_root,
        model_name,
        space_name,
        summary_filename=summary_filename,
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")
    return pd.read_csv(summary_path)


def select_unique_summary_row(
    summary_df: pd.DataFrame,
    *,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
) -> pd.Series:
    required_cols = [
        "pca_dim",
        "l2_norm",
        "min_cluster_size",
        "min_samples",
        "metric",
        "cluster_selection_method",
        "cluster_selection_epsilon",
    ]
    missing = [c for c in required_cols if c not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    mask = (
        (summary_df["pca_dim"] == pca_dim)
        & (summary_df["l2_norm"] == l2_norm)
        & (summary_df["min_cluster_size"] == min_cluster_size)
        & (summary_df["min_samples"] == min_samples)
        & (summary_df["metric"] == metric)
        & (summary_df["cluster_selection_method"] == cluster_selection_method)
        & (summary_df["cluster_selection_epsilon"] == cluster_selection_epsilon)
    )

    matched = summary_df.loc[mask]
    if len(matched) != 1:
        raise ValueError(
            "Expected exactly one matching row in summary_df, "
            f"but found {len(matched)} rows."
        )
    return matched.iloc[0]


def load_cluster_df_from_summary_row(
    summary_row: pd.Series,
    *,
    out_root: str,
    model_name: str,
    space_name: str,
    print_columns: bool = True,
) -> Tuple[pd.DataFrame, Path, Path]:
    if "cluster_csv_path" not in summary_row.index:
        raise KeyError("Selected summary row missing column: cluster_csv_path")
    if "meta_json_path" not in summary_row.index:
        raise KeyError("Selected summary row missing column: meta_json_path")

    cluster_csv_path = Path(summary_row["cluster_csv_path"])
    meta_json_path   = Path(summary_row["meta_json_path"])

    if not cluster_csv_path.exists():
        raise FileNotFoundError(f"cluster_csv_path not found: {cluster_csv_path}")

    cluster_df = pd.read_csv(cluster_csv_path)

    if print_columns:
        print("cluster_df.columns =", list(cluster_df.columns))

    return cluster_df, cluster_csv_path, meta_json_path


from crossscale.metrics.script_entropy import (  # noqa: F401
    decode_token_ids_to_strings,
    standardize_cluster_input_df,
    raw_char_script,
    normalized_char_script,
    token_script_counts,
    dominant_script,
    annotate_token_scripts,
    get_full_vocab_token_strings,
    build_global_script_cache_from_tokenizer,
    filter_noise_clusters,
    build_cluster_script_count_df,
    cluster_script_distribution_from_count_row,
    entropy_from_distribution,
    normalized_entropy,
    compute_cluster_entropy_df,
    partition_entropy_stats,
)


# =========================================================
# Single-partition runner
# =========================================================

def run_script_entropy_for_partition(
    *,
    summary_df: pd.DataFrame,
    out_root: str,
    model_name: str,
    space_name: str,
    pca_dim: int,
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    tokenizer: Any,
    global_script_set: Sequence[str],
    cluster_id_col: str = "cluster_id",
    token_str_col: str = "token_str",
    token_id_col: str = "token_id",
    script_col: str = "script",
    print_columns: bool = False,
) -> Dict[str, Any]:
    """
    Run script entropy for one partition only.
    Assumes global_script_set has already been prepared once at model level.
    """
    summary_row = select_unique_summary_row(
        summary_df,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    cluster_df_raw, cluster_csv_abs_path, meta_json_abs_path = load_cluster_df_from_summary_row(
        summary_row,
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        print_columns=print_columns,
    )

    cluster_df = standardize_cluster_input_df(
        cluster_df_raw,
        model_name=model_name,
        tokenizer=tokenizer,
        cluster_id_col=cluster_id_col,
        token_str_col=token_str_col,
        token_id_col=token_id_col,
    )

    token_df = annotate_token_scripts(
        cluster_df,
        token_col=token_str_col,
        script_col=script_col,
    )

    token_df_non_noise = filter_noise_clusters(
        token_df,
        cluster_id_col=cluster_id_col,
        noise_cluster_id=NOISE_CLUSTER_ID,
    )

    cluster_script_count_df = build_cluster_script_count_df(
        token_df_non_noise,
        global_script_set=global_script_set,
        cluster_id_col=cluster_id_col,
        script_col=script_col,
    )

    cluster_entropy_df = compute_cluster_entropy_df(
        cluster_script_count_df,
        global_script_set=global_script_set,
        cluster_id_col=cluster_id_col,
        warn_on_degenerate_global_set=True,
    )

    stats = partition_entropy_stats(
        cluster_entropy_df,
        raw_entropy_col= "H",
        norm_entropy_col="H_norm",
    )

    return {
        "summary_row": summary_row,
        "cluster_csv_abs_path": cluster_csv_abs_path,
        "meta_json_abs_path": meta_json_abs_path,
        "token_df": token_df,
        "token_df_non_noise": token_df_non_noise,
        "cluster_script_count_df": cluster_script_count_df,
        "cluster_entropy_df": cluster_entropy_df,
        "partition_stats": stats,
    }


# =========================================================
# Batch runner over pca_dim_list
# =========================================================

def run_script_entropy_over_pca_dims(
    *,
    out_root: str,
    model_name: str,
    space_name: str,
    pca_dim_list: Sequence[int],
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    tokenizer: Any,
    cluster_id_col: str = "cluster_id",
    token_str_col: str = "token_str",
    token_id_col: str = "token_id",
    script_col: str = "script",
    print_columns_first_only: bool = True,
) -> Dict[str, Any]:
    """
    Model-level batch runner:
    - load summary_df once
    - build global script cache once
    - loop over pca_dim_list
    """
    summary_df = load_summary_df(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    )

    script_cache = build_global_script_cache_from_tokenizer(
        tokenizer,
        model_name=model_name,
    )
    global_script_set = script_cache["global_script_set"]

    rows: List[Dict[str, Any]] = []
    per_partition_results: Dict[int, Dict[str, Any]] = {}

    for idx, pca_dim in enumerate(pca_dim_list):
        print(f"\n=== Running script entropy: pca_dim={pca_dim} ===")

        part_result = run_script_entropy_for_partition(
            summary_df=summary_df,
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
            tokenizer=tokenizer,
            global_script_set=global_script_set,
            cluster_id_col=cluster_id_col,
            token_str_col=token_str_col,
            token_id_col=token_id_col,
            script_col=script_col,
            print_columns=(print_columns_first_only and idx == 0),
        )

        stats = part_result["partition_stats"]

        row = {
            "model_name": model_name,
            "space_name": space_name,
            "pca_dim": pca_dim,
            "l2_norm": l2_norm,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_selection_method": cluster_selection_method,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "num_clusters": stats["num_clusters"],
            "mean_H": stats["mean_H"],
            "std_H": stats["std_H"],
            "mean_H_norm": stats["mean_H_norm"],
            "std_H_norm": stats["std_H_norm"],
        }
        rows.append(row)
        per_partition_results[pca_dim] = part_result

    summary_out_df = pd.DataFrame(rows)

    return {
        "script_cache": script_cache,
        "summary_df": summary_out_df,
        "per_partition_results": per_partition_results,
    }


# =========================================================
# Saving helpers
# =========================================================

def save_single_partition_outputs(
    *,
    result: Dict[str, Any],
    out_dir: str | Path,
    cluster_filename: str = "cluster_script_entropy.csv",
    summary_filename: str = "partition_script_entropy_summary.json",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_path = out_dir / cluster_filename
    summary_path = out_dir / summary_filename

    result["cluster_entropy_df"].to_csv(cluster_path, index=False)

    payload = {
        "num_clusters": result["partition_stats"]["num_clusters"],
        "mean_H_norm": result["partition_stats"]["mean_H_norm"],
        "std_H_norm": result["partition_stats"]["std_H_norm"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved cluster-level output -> {cluster_path}")
    print(f"Saved summary output       -> {summary_path}")


def save_batch_summary_df(
    *,
    summary_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"Saved batch summary -> {out_path}")


# =========================================================
# Minimal sanity tests
# =========================================================

def sanity_check_words() -> pd.DataFrame:
    words = ["shoulder", "shoulders", "homb", "Shoulder", "肩", "Schulter", "плеч"]
    df = pd.DataFrame({"token_str": words})
    df = annotate_token_scripts(df, token_col="token_str", script_col="script")
    return df


def toy_partition_test() -> Dict[str, Any]:
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 1, 1, 1],
            "token_str": ["shoulder", "shoulders", "Schulter", "肩", "плеч", "肩"],
        }
    )

    token_df = annotate_token_scripts(df, token_col="token_str", script_col="script")
    global_script_set = list(set(token_df["script"].tolist()))

    token_df_non_noise = filter_noise_clusters(token_df, cluster_id_col="cluster_id")
    count_df = build_cluster_script_count_df(
        token_df_non_noise,
        global_script_set=global_script_set,
        cluster_id_col="cluster_id",
        script_col="script",
    )
    entropy_df = compute_cluster_entropy_df(
        count_df,
        global_script_set=global_script_set,
        cluster_id_col="cluster_id",
    )
    stats = partition_entropy_stats(entropy_df, raw_entropy_col= "H", norm_entropy_col="H_norm")

    return {
        "token_df": token_df,
        "global_script_set": global_script_set,
        "cluster_entropy_df": entropy_df,
        "partition_stats": stats,
    }

# =========================================================
# Random-PCA path / summary resolution
# =========================================================

RANDOM_SUMMARY_FILENAME_TEMPLATE = "summary_random_pca_seed_{pca_seed}.csv"


def get_random_summary_csv_path(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
) -> Path:
    return (
        Path(out_root)
        / model_name
        / space_name
        / "random"
        / f"seed_{pca_seed}"
        / RANDOM_SUMMARY_FILENAME_TEMPLATE.format(pca_seed=pca_seed)
    )


def load_random_summary_df(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
) -> pd.DataFrame:
    summary_path = get_random_summary_csv_path(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"random summary csv not found: {summary_path}")
    return pd.read_csv(summary_path)


# =========================================================
# Random-PCA path / summary resolution
# =========================================================

def get_random_summary_csv_path(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
) -> Path:
    return (
        Path(out_root)
        / model_name
        / space_name
        / "random"
        / f"seed_{pca_seed}"
        / f"summary_random_pca_seed_{pca_seed}.csv"
    )


def load_random_summary_df(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
) -> pd.DataFrame:
    summary_path = get_random_summary_csv_path(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"random summary csv not found: {summary_path}"
        )
    return pd.read_csv(summary_path)


# =========================================================
# Random-PCA single-partition runner
# =========================================================

def run_script_entropy_for_random_partition(
    *,
    summary_df: pd.DataFrame,
    out_root: str,
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
    tokenizer: Any,
    global_script_set: Sequence[str],
    cluster_id_col: str = "cluster_id",
    token_str_col: str = "token_str",
    token_id_col: str = "token_id",
    script_col: str = "script",
    print_columns: bool = True,
) -> Dict[str, Any]:
    """
    Random-PCA version of the single-partition runner.

    Same logic as run_script_entropy_for_partition(...),
    but bound to one random PCA seed task.
    """
    summary_row = select_unique_summary_row(
        summary_df,
        pca_dim=pca_dim,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    cluster_df_raw, cluster_csv_abs_path, meta_json_abs_path = load_cluster_df_from_summary_row(
        summary_row,
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        print_columns=print_columns,
    )

    cluster_df = standardize_cluster_input_df(
        cluster_df_raw,
        model_name=model_name,
        tokenizer=tokenizer,
        cluster_id_col=cluster_id_col,
        token_str_col=token_str_col,
        token_id_col=token_id_col,
    )

    token_df = annotate_token_scripts(
        cluster_df,
        token_col=token_str_col,
        script_col=script_col,
    )

    token_df_non_noise = filter_noise_clusters(
        token_df,
        cluster_id_col=cluster_id_col,
        noise_cluster_id=NOISE_CLUSTER_ID,
    )

    cluster_script_count_df = build_cluster_script_count_df(
        token_df_non_noise,
        global_script_set=global_script_set,
        cluster_id_col=cluster_id_col,
        script_col=script_col,
    )

    cluster_entropy_df = compute_cluster_entropy_df(
        cluster_script_count_df,
        global_script_set=global_script_set,
        cluster_id_col=cluster_id_col,
        warn_on_degenerate_global_set=True,
    )

    stats = partition_entropy_stats(
        cluster_entropy_df,
        raw_entropy_col="H",
        norm_entropy_col="H_norm",
    )

    return {
        "pca_seed": pca_seed,
        "summary_row": summary_row,
        "cluster_csv_abs_path": cluster_csv_abs_path,
        "meta_json_abs_path": meta_json_abs_path,
        "token_df": token_df,
        "token_df_non_noise": token_df_non_noise,
        "cluster_script_count_df": cluster_script_count_df,
        "cluster_entropy_df": cluster_entropy_df,
        "partition_stats": stats,
    }


# =========================================================
# Random-PCA batch runner over pca_dim_list
# =========================================================

def run_script_entropy_over_random_pca_dims(
    *,
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
    pca_dim_list: Sequence[int],
    l2_norm: bool,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    tokenizer: Any,
    cluster_id_col: str = "cluster_id",
    token_str_col: str = "token_str",
    token_id_col: str = "token_id",
    script_col: str = "script",
    print_columns_first_only: bool = True,
) -> Dict[str, Any]:
    """
    Random-PCA model-level batch runner for a single seed:
    - load random summary_df once
    - build global script cache once
    - loop over pca_dim_list
    """
    summary_df = load_random_summary_df(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
    )

    script_cache = build_global_script_cache_from_tokenizer(
        tokenizer,
        model_name=model_name,
    )
    global_script_set = script_cache["global_script_set"]

    rows: List[Dict[str, Any]] = []
    per_partition_results: Dict[int, Dict[str, Any]] = {}

    for idx, pca_dim in enumerate(pca_dim_list):
        print(
            f"\n=== Running random script entropy: "
            f"seed={pca_seed}, pca_dim={pca_dim} ==="
        )

        part_result = run_script_entropy_for_random_partition(
            summary_df=summary_df,
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
            tokenizer=tokenizer,
            global_script_set=global_script_set,
            cluster_id_col=cluster_id_col,
            token_str_col=token_str_col,
            token_id_col=token_id_col,
            script_col=script_col,
            print_columns=(print_columns_first_only and idx == 0),
        )

        stats = part_result["partition_stats"]

        row = {
            "model_name": model_name,
            "space_name": space_name,
            "pca_seed": pca_seed,
            "pca_dim": pca_dim,
            "l2_norm": l2_norm,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_selection_method": cluster_selection_method,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "num_clusters": stats["num_clusters"],
            "mean_H": stats["mean_H"],
            "std_H": stats["std_H"],
            "mean_H_norm": stats["mean_H_norm"],
            "std_H_norm": stats["std_H_norm"],
        }
        rows.append(row)
        per_partition_results[pca_dim] = part_result

    summary_out_df = pd.DataFrame(rows)

    return {
        "script_cache": script_cache,
        "summary_df": summary_out_df,
        "per_partition_results": per_partition_results,
    }


# =========================================================
# Random-PCA output path helpers
# =========================================================

def get_random_seed_output_dir(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
) -> Path:
    return (
        Path(out_root)
        / model_name
        / space_name
        / "random"
        / f"seed_{pca_seed}"
    )


def get_random_partition_output_dir(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
    pca_dim: int,
) -> Path:
    return get_random_seed_output_dir(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
    ) / f"script_entropy_pca{pca_dim}"


def get_random_batch_summary_out_path(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_seed: int,
) -> Path:
    return get_random_seed_output_dir(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_seed=pca_seed,
    ) / f"summary_random_pca_seed_{pca_seed}_script_entropy.csv"

# =========================================================
# Permutation multi-seed pipeline
# =========================================================

def permutation_seed_summary_filename(perm_seed: int) -> str:
    return f"seeds/seed_{int(perm_seed)}/summary.csv"



def permutation_seed_script_dir(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    perm_seed: int,
    pca_dim: int,
) -> Path:
    return (
        Path(out_root)
        / model_name
        / space_name
        / "seeds"
        / f"seed_{int(perm_seed)}"
        / "script_entropy"
        / f"pca_{int(pca_dim)}"
    )



def permutation_seed_script_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    perm_seed: int,
    filename: str = "script_entropy_summary.csv",
) -> Path:
    return Path(out_root) / model_name / space_name / "seeds" / f"seed_{int(perm_seed)}" / filename



def permutation_all_seed_script_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    filename: str = "script_entropy_summary_all_seeds.csv",
) -> Path:
    return Path(out_root) / model_name / space_name / filename



def permutation_mean_script_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    filename: str = "script_entropy_summary_mean.csv",
) -> Path:
    return Path(out_root) / model_name / space_name / filename



def aggregate_permutation_script_means(all_seed_summary_df: pd.DataFrame) -> pd.DataFrame:
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
                "num_clusters",
                "mean_H",
                "std_H",
                "mean_H_norm",
                "std_H_norm",
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
            num_clusters=("num_clusters", "mean"),
            mean_H=("mean_H", "mean"),
            std_H=("std_H", "mean"),
            mean_H_norm=("mean_H_norm", "mean"),
            std_H_norm=("std_H_norm", "mean"),
        )
    )
    mean_df["num_clusters"] = mean_df["num_clusters"].round().astype(int)
    return mean_df.sort_values(["pca_dim"]).reset_index(drop=True)



def run_script_entropy_over_permutation_seeds(
    *,
    out_root: str,
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
    tokenizer: Any,
    cluster_id_col: str = "cluster_id",
    token_str_col: str = "token_str",
    token_id_col: str = "token_id",
    script_col: str = "script",
    print_columns_first_only: bool = True,
) -> Dict[str, Any]:
    script_cache = build_global_script_cache_from_tokenizer(
        tokenizer,
        model_name=model_name,
    )
    global_script_set = script_cache["global_script_set"]

    per_seed_results: Dict[int, Dict[str, Any]] = {}
    summary_frames: List[pd.DataFrame] = []

    for perm_seed in perm_seed_list:
        perm_seed = int(perm_seed)
        summary_df = load_summary_df(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            summary_filename=permutation_seed_summary_filename(perm_seed),
        )

        rows: List[Dict[str, Any]] = []
        per_partition_results: Dict[int, Dict[str, Any]] = {}

        for idx, pca_dim in enumerate(pca_dim_list):
            part_result = run_script_entropy_for_partition(
                summary_df=summary_df,
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
                tokenizer=tokenizer,
                global_script_set=global_script_set,
                cluster_id_col=cluster_id_col,
                token_str_col=token_str_col,
                token_id_col=token_id_col,
                script_col=script_col,
                print_columns=(print_columns_first_only and idx == 0),
            )

            stats = part_result["partition_stats"]
            row = {
                "model_name": model_name,
                "space_name": space_name,
                "perm_seed": perm_seed,
                "pca_dim": int(pca_dim),
                "l2_norm": l2_norm,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": metric,
                "cluster_selection_method": cluster_selection_method,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "num_clusters": stats["num_clusters"],
                "mean_H": stats["mean_H"],
                "std_H": stats["std_H"],
                "mean_H_norm": stats["mean_H_norm"],
                "std_H_norm": stats["std_H_norm"],
            }
            rows.append(row)
            per_partition_results[int(pca_dim)] = part_result

            out_dir = permutation_seed_script_dir(
                out_root=out_root,
                model_name=model_name,
                space_name=space_name,
                perm_seed=perm_seed,
                pca_dim=int(pca_dim),
            )
            save_single_partition_outputs(
                result=part_result,
                out_dir=out_dir,
            )

        seed_summary_df = pd.DataFrame(rows).sort_values(["pca_dim"]).reset_index(drop=True)
        seed_summary_path = permutation_seed_script_summary_path(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            perm_seed=perm_seed,
        )
        save_batch_summary_df(summary_df=seed_summary_df, out_path=seed_summary_path)

        per_seed_results[perm_seed] = {
            "summary_df": seed_summary_df,
            "per_partition_results": per_partition_results,
        }
        summary_frames.append(seed_summary_df)

    if summary_frames:
        all_seed_summary_df = pd.concat(summary_frames, axis=0, ignore_index=True)
        all_seed_summary_df = all_seed_summary_df.sort_values(["perm_seed", "pca_dim"]).reset_index(drop=True)
    else:
        all_seed_summary_df = pd.DataFrame()

    mean_summary_df = aggregate_permutation_script_means(all_seed_summary_df)

    save_batch_summary_df(
        summary_df=all_seed_summary_df,
        out_path=permutation_all_seed_script_summary_path(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
        ),
    )
    save_batch_summary_df(
        summary_df=mean_summary_df,
        out_path=permutation_mean_script_summary_path(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
        ),
    )
    save_batch_summary_df(
        summary_df=mean_summary_df,
        out_path=Path(out_root) / model_name / space_name / "script_entropy_summary.csv",
    )

    return {
        "script_cache": script_cache,
        "per_seed_results": per_seed_results,
        "all_seed_summary_df": all_seed_summary_df,
        "mean_summary_df": mean_summary_df,
    }
