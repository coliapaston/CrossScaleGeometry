from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


REQUIRED_SUMMARY_COLS = [
    "run_id",
    "model_name",
    "space_name",
    "pca_dim",
    "l2_norm",
    "cluster_csv_path",
]

REQUIRED_CLUSTER_COLS = [
    "token_id",
    "cluster_id",
]

PERMUTATION_SUMMARY_FILENAME = "summary.csv"
PERMUTATION_ALL_SEEDS_FILENAME = "summary_all_seeds.csv"


def load_summary_df(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    summary_filename: str = PERMUTATION_SUMMARY_FILENAME,
) -> pd.DataFrame:
    summary_path = Path(out_root) / model_name / space_name / summary_filename
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")
    return pd.read_csv(summary_path)



def load_cluster_df(cluster_csv_path: str | Path) -> pd.DataFrame:
    cluster_csv_path = Path(cluster_csv_path)
    if not cluster_csv_path.exists():
        raise FileNotFoundError(f"cluster csv not found: {cluster_csv_path}")
    return pd.read_csv(cluster_csv_path)



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



def build_permutation_root_dir(
    *,
    out_root: str | Path,
    fake_model_name: str,
    space_name: str,
) -> Path:
    return Path(out_root) / fake_model_name / space_name



def build_permutation_seed_dir(
    *,
    out_root: str | Path,
    fake_model_name: str,
    space_name: str,
    perm_seed: Optional[int] = None,
) -> Path:
    root_dir = build_permutation_root_dir(
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
    )
    if perm_seed is None:
        return root_dir
    return root_dir / "seeds" / f"seed_{int(perm_seed)}"



def build_permutation_cluster_csv_path(
    *,
    out_root: str | Path,
    fake_model_name: str,
    space_name: str,
    l2_norm: bool,
    run_id: int,
    perm_seed: Optional[int] = None,
) -> Path:
    cluster_dir = build_permutation_seed_dir(
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        perm_seed=perm_seed,
    ) / f"clusters_l2={bool(l2_norm)}"
    return cluster_dir / f"run_{int(run_id):05d}_clusters.csv"



def build_permutation_summary_path(
    *,
    out_root: str | Path,
    fake_model_name: str,
    space_name: str,
    perm_seed: Optional[int] = None,
    summary_filename: str = PERMUTATION_SUMMARY_FILENAME,
) -> Path:
    return build_permutation_seed_dir(
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        perm_seed=perm_seed,
    ) / summary_filename



def save_permuted_cluster_df(permuted_cluster_df: pd.DataFrame, outpath: str | Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    permuted_cluster_df.to_csv(outpath, index=False)
    return outpath



def rewrite_summary_df_for_permutation(
    summary_df: pd.DataFrame,
    *,
    fake_model_name: str,
    new_cluster_paths_by_run_id: Dict[int, str],
    perm_seed: Optional[int] = None,
) -> pd.DataFrame:
    missing = [c for c in REQUIRED_SUMMARY_COLS if c not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    out = summary_df.copy()
    out["run_id"] = pd.to_numeric(out["run_id"], errors="raise").astype(int)
    out["model_name"] = str(fake_model_name)
    out["cluster_csv_path"] = out["run_id"].map(new_cluster_paths_by_run_id)
    if perm_seed is not None:
        out["perm_seed"] = int(perm_seed)

    missing_paths = out.loc[out["cluster_csv_path"].isna(), "run_id"].tolist()
    if missing_paths:
        raise ValueError(f"Missing rewritten cluster paths for run_id values: {missing_paths}")

    return out



def save_permutation_summary_df(
    summary_df: pd.DataFrame,
    *,
    out_root: str | Path,
    fake_model_name: str,
    space_name: str,
    summary_filename: str = PERMUTATION_SUMMARY_FILENAME,
    perm_seed: Optional[int] = None,
) -> Path:
    outpath = build_permutation_summary_path(
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        perm_seed=perm_seed,
        summary_filename=summary_filename,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outpath, index=False)
    return outpath



def run_single_partition_permutation(
    *,
    cluster_csv_path: str | Path,
    out_root: str | Path,
    fake_model_name: str,
    space_name: str,
    l2_norm: bool,
    run_id: int,
    random_state: Optional[int] = None,
    perm_seed: Optional[int] = None,
) -> Path:
    cluster_df = load_cluster_df(cluster_csv_path)
    valid_cluster_df = filter_valid_cluster_df(cluster_df)
    permuted_token_ids = permute_token_ids(valid_cluster_df, random_state=random_state)
    permuted_cluster_df = build_permuted_cluster_df(valid_cluster_df, permuted_token_ids)
    outpath = build_permutation_cluster_csv_path(
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        l2_norm=l2_norm,
        run_id=run_id,
        perm_seed=perm_seed,
    )
    return save_permuted_cluster_df(permuted_cluster_df, outpath)



def run_partition_permutation_batch_for_seed(
    *,
    out_root: str | Path,
    model_name: str,
    fake_model_name: str,
    space_name: str,
    perm_seed: int,
    source_summary_filename: str = PERMUTATION_SUMMARY_FILENAME,
    output_summary_filename: str = PERMUTATION_SUMMARY_FILENAME,
) -> tuple[pd.DataFrame, Path]:
    summary_df = load_summary_df(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        summary_filename=source_summary_filename,
    )

    missing = [c for c in REQUIRED_SUMMARY_COLS if c not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    new_cluster_paths_by_run_id: Dict[int, str] = {}

    for _, row in summary_df.iterrows():
        run_id = int(row["run_id"])
        l2_norm = bool(row["l2_norm"])
        saved_path = run_single_partition_permutation(
            cluster_csv_path=row["cluster_csv_path"],
            out_root=out_root,
            fake_model_name=fake_model_name,
            space_name=space_name,
            l2_norm=l2_norm,
            run_id=run_id,
            random_state=int(perm_seed),
            perm_seed=int(perm_seed),
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



def aggregate_permutation_seed_summaries(
    summary_df_by_seed: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    frames = []
    for perm_seed, summary_df in summary_df_by_seed.items():
        frame = summary_df.copy()
        if "perm_seed" not in frame.columns:
            frame["perm_seed"] = int(perm_seed)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=True)
    sort_cols = [c for c in ["perm_seed", "run_id", "pca_dim"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out



def save_permutation_all_seed_summary_df(
    summary_df: pd.DataFrame,
    *,
    out_root: str | Path,
    fake_model_name: str,
    space_name: str,
    filename: str = PERMUTATION_ALL_SEEDS_FILENAME,
) -> Path:
    outpath = build_permutation_root_dir(
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
    ) / filename
    outpath.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outpath, index=False)
    return outpath



def run_partition_permutation_multi_seed(
    *,
    out_root: str | Path,
    model_name: str,
    fake_model_name: str,
    space_name: str,
    seed_list: Sequence[int],
    source_summary_filename: str = PERMUTATION_SUMMARY_FILENAME,
) -> tuple[Dict[int, pd.DataFrame], pd.DataFrame, Path]:
    summary_df_by_seed: Dict[int, pd.DataFrame] = {}

    for perm_seed in seed_list:
        perm_seed = int(perm_seed)
        perm_summary_df, _ = run_partition_permutation_batch_for_seed(
            out_root=out_root,
            model_name=model_name,
            fake_model_name=fake_model_name,
            space_name=space_name,
            perm_seed=perm_seed,
            source_summary_filename=source_summary_filename,
            output_summary_filename=PERMUTATION_SUMMARY_FILENAME,
        )
        summary_df_by_seed[perm_seed] = perm_summary_df

    all_seed_summary_df = aggregate_permutation_seed_summaries(summary_df_by_seed)
    all_seed_summary_path = save_permutation_all_seed_summary_df(
        all_seed_summary_df,
        out_root=out_root,
        fake_model_name=fake_model_name,
        space_name=space_name,
        filename=PERMUTATION_ALL_SEEDS_FILENAME,
    )
    return summary_df_by_seed, all_seed_summary_df, all_seed_summary_path



def run_partition_permutation_batch(
    *,
    out_root: str | Path,
    model_name: str,
    fake_model_name: str,
    space_name: str,
    summary_filename: str = PERMUTATION_SUMMARY_FILENAME,
    base_random_state: int = 42,
) -> tuple[pd.DataFrame, Path]:
    return run_partition_permutation_batch_for_seed(
        out_root=out_root,
        model_name=model_name,
        fake_model_name=fake_model_name,
        space_name=space_name,
        perm_seed=int(base_random_state),
        source_summary_filename=summary_filename,
        output_summary_filename=summary_filename,
    )
