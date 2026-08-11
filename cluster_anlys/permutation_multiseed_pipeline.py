from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

try:
    from .partition_permutation_control import run_partition_permutation_multi_seed
    from .kondrak_cluster_morphology import (
        run_kondrak_morphology_for_permutation_seeds,
    )
    from .script_entropy import run_script_entropy_over_permutation_seeds
except ImportError:
    from partition_permutation_control import run_partition_permutation_multi_seed
    from kondrak_cluster_morphology import run_kondrak_morphology_for_permutation_seeds
    from script_entropy import run_script_entropy_over_permutation_seeds


DEFAULT_OUT_ROOT = "comp"
DEFAULT_SPACE_NAME = "output_proj"
DEFAULT_L2_NORM = True
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 5
DEFAULT_METRIC = "euclidean"
DEFAULT_CLUSTER_SELECTION_METHOD = "eom"
DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-oss": {
        "source_model_name": "gpt-oss",
        "perm_model_name": "permutation/gpt-oss_rand",
        "tokenizer_path": "gpt-oss/tokenizer",
        "pca_dim_list": [6, 182, 466, 739, 1591, 2264, 2532, 2868, 2880],
    },
    "mistral": {
        "source_model_name": "mistralai/Mistral-7B-v0.1",
        "perm_model_name": "permutation/mistralai/Mistral-7B-v0.1_rand",
        "tokenizer_path": "mistralai/Mistral-7B-v0.1/tokenizer",
        "pca_dim_list": [5, 142, 997, 2084, 3031, 3459, 3938, 4088, 4096],
    },
    "mixtral": {
        "source_model_name": "mistralai/Mixtral-8x7B-v0.1",
        "perm_model_name": "permutation/mistralai/Mixtral-8x7B-v0.1_rand",
        "tokenizer_path": "mistralai/Mixtral-8x7B-v0.1/tokenizer",
        "pca_dim_list": [8, 158, 1111, 2156, 3052, 3457, 3957, 4093, 4096],
    },
}


def run_permutation_multiseed_pipeline(
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    source_model_name: str,
    perm_model_name: str,
    space_name: str = DEFAULT_SPACE_NAME,
    seed_list: Sequence[int],
    pca_dim_list: Sequence[int],
    tokenizer,
    l2_norm: bool = DEFAULT_L2_NORM,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    metric: str = DEFAULT_METRIC,
    cluster_selection_method: str = DEFAULT_CLUSTER_SELECTION_METHOD,
    cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
) -> Dict[str, pd.DataFrame]:
    _, permutation_summary_all_df, _ = run_partition_permutation_multi_seed(
        out_root=out_root,
        model_name=source_model_name,
        fake_model_name=perm_model_name,
        space_name=space_name,
        seed_list=seed_list,
    )

    morph_result = run_kondrak_morphology_for_permutation_seeds(
        out_root=out_root,
        model_name=perm_model_name,
        space_name=space_name,
        perm_seed_list=seed_list,
        pca_dim_list=pca_dim_list,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        tokenizer=tokenizer,
        print_columns=False,
        save_cluster_csv=True,
    )

    script_result = run_script_entropy_over_permutation_seeds(
        out_root=str(out_root),
        model_name=perm_model_name,
        space_name=space_name,
        perm_seed_list=seed_list,
        pca_dim_list=pca_dim_list,
        l2_norm=l2_norm,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        tokenizer=tokenizer,
        print_columns_first_only=False,
    )

    return {
        "permutation_summary_all_df": permutation_summary_all_df,
        "morphology_all_seed_summary_df": morph_result["all_seed_summary_df"],
        "morphology_mean_summary_df": morph_result["mean_summary_df"],
        "script_all_seed_summary_df": script_result["all_seed_summary_df"],
        "script_mean_summary_df": script_result["mean_summary_df"],
    }


def make_length_bucketed_perm_model_name(perm_model_name: str) -> str:
    if not str(perm_model_name).endswith("_rand"):
        raise ValueError(f"Expected perm_model_name to end with '_rand': {perm_model_name}")
    return str(perm_model_name)[:-len("_rand")] + "_length_bucket_rand"


def load_tokenizer_for_model_config(model_config: Dict[str, Any]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_config["tokenizer_path"])


def token_id_to_token_text(
    token_id: int,
    *,
    tokenizer,
    model_name: str,
) -> str:
    tid = int(token_id)
    if str(model_name).startswith("gpt"):
        return str(tokenizer.decode([tid], clean_up_tokenization_spaces=False))
    return str(tokenizer.convert_ids_to_tokens(tid))


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


def run_single_length_bucketed_partition_permutation(
    *,
    cluster_csv_path: str | Path,
    out_root: str | Path,
    source_model_name: str,
    fake_model_name: str,
    space_name: str,
    l2_norm: bool,
    run_id: int,
    random_state: int,
    perm_seed: int,
    tokenizer,
    token_text_cache: Dict[int, str] | None = None,
) -> Path:
    from partition_permutation_control import (
        build_permutation_cluster_csv_path,
        filter_valid_cluster_df,
        load_cluster_df,
        save_permuted_cluster_df,
    )

    cluster_df = load_cluster_df(cluster_csv_path)
    valid_cluster_df = filter_valid_cluster_df(cluster_df)
    permuted_cluster_df = permute_token_ids_within_length_buckets(
        valid_cluster_df,
        tokenizer=tokenizer,
        model_name=source_model_name,
        random_state=int(random_state),
        token_text_cache=token_text_cache,
    )
    outpath = build_permutation_cluster_csv_path(
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        l2_norm=l2_norm,
        run_id=run_id,
        perm_seed=perm_seed,
    )
    return save_permuted_cluster_df(permuted_cluster_df, outpath)


def run_length_bucketed_partition_permutation_batch_for_seed(
    *,
    out_root: str | Path,
    source_model_name: str,
    fake_model_name: str,
    space_name: str,
    perm_seed: int,
    tokenizer,
    token_text_cache: Dict[int, str] | None = None,
    source_summary_filename: str = "summary.csv",
    output_summary_filename: str = "summary.csv",
) -> tuple[pd.DataFrame, Path]:
    from partition_permutation_control import (
        load_summary_df,
        rewrite_summary_df_for_permutation,
        save_permutation_summary_df,
    )

    summary_df = load_summary_df(
        out_root=out_root,
        model_name=source_model_name,
        space_name=space_name,
        summary_filename=source_summary_filename,
    )

    required_cols = [
        "run_id",
        "model_name",
        "space_name",
        "pca_dim",
        "l2_norm",
        "cluster_csv_path",
    ]
    missing = [col for col in required_cols if col not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    new_cluster_paths_by_run_id: Dict[int, str] = {}

    for _, row in summary_df.iterrows():
        run_id = int(row["run_id"])
        l2_norm = bool(row["l2_norm"])
        saved_path = run_single_length_bucketed_partition_permutation(
            cluster_csv_path=row["cluster_csv_path"],
            out_root=out_root,
            source_model_name=source_model_name,
            fake_model_name=fake_model_name,
            space_name=space_name,
            l2_norm=l2_norm,
            run_id=run_id,
            random_state=int(perm_seed),
            perm_seed=int(perm_seed),
            tokenizer=tokenizer,
            token_text_cache=token_text_cache,
        )
        new_cluster_paths_by_run_id[run_id] = str(saved_path)

    perm_summary_df = rewrite_summary_df_for_permutation(
        summary_df,
        fake_model_name=fake_model_name,
        new_cluster_paths_by_run_id=new_cluster_paths_by_run_id,
        perm_seed=int(perm_seed),
    )
    summary_outpath = save_permutation_summary_df(
        perm_summary_df,
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        summary_filename=output_summary_filename,
        perm_seed=int(perm_seed),
    )
    return perm_summary_df, summary_outpath


def run_length_bucketed_partition_permutation_multi_seed(
    *,
    out_root: str | Path,
    source_model_name: str,
    fake_model_name: str,
    space_name: str,
    seed_list: Sequence[int],
    tokenizer,
    source_summary_filename: str = "summary.csv",
) -> tuple[Dict[int, pd.DataFrame], pd.DataFrame, Path]:
    from partition_permutation_control import (
        aggregate_permutation_seed_summaries,
        save_permutation_all_seed_summary_df,
    )

    summary_df_by_seed: Dict[int, pd.DataFrame] = {}
    token_text_cache: Dict[int, str] = {}

    for perm_seed in seed_list:
        perm_seed = int(perm_seed)
        perm_summary_df, _ = run_length_bucketed_partition_permutation_batch_for_seed(
            out_root=out_root,
            source_model_name=source_model_name,
            fake_model_name=fake_model_name,
            space_name=space_name,
            perm_seed=perm_seed,
            tokenizer=tokenizer,
            token_text_cache=token_text_cache,
            source_summary_filename=source_summary_filename,
            output_summary_filename="summary.csv",
        )
        summary_df_by_seed[perm_seed] = perm_summary_df

    all_seed_summary_df = aggregate_permutation_seed_summaries(summary_df_by_seed)
    all_seed_summary_path = save_permutation_all_seed_summary_df(
        all_seed_summary_df,
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        filename="summary_all_seeds.csv",
    )
    return summary_df_by_seed, all_seed_summary_df, all_seed_summary_path


def run_length_bucketed_partition_permutation_multi_seed_for_model_key(
    *,
    model_key: str,
    seed_list: Sequence[int],
    out_root: str | Path = DEFAULT_OUT_ROOT,
    space_name: str = DEFAULT_SPACE_NAME,
) -> tuple[Dict[int, pd.DataFrame], pd.DataFrame, Path]:
    if model_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model_key: {model_key}")

    model_config = MODEL_CONFIGS[model_key]
    source_model_name = model_config["source_model_name"]
    fake_model_name = make_length_bucketed_perm_model_name(model_config["perm_model_name"])
    tokenizer = load_tokenizer_for_model_config(model_config)

    return run_length_bucketed_partition_permutation_multi_seed(
        out_root=out_root,
        source_model_name=source_model_name,
        fake_model_name=fake_model_name,
        space_name=space_name,
        seed_list=seed_list,
        tokenizer=tokenizer,
    )
