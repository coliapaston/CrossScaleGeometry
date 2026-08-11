"""Utilities for building paper tables from clustering outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_output_proj_dir(model_name: str, project_root: str | Path | None = None) -> Path:
    """Return the output_proj directory for a model."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root / "comp" / model_name / "output_proj"



def load_output_proj_summary(model_name: str, project_root: str | Path | None = None) -> pd.DataFrame:
    """Load comp/{model_name}/output_proj/summary.csv."""
    summary_path = get_output_proj_dir(model_name=model_name, project_root=project_root) / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)
    required_columns = {"model_name", "pca_dim", "cluster_csv_path"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"summary.csv is missing required columns: {missing_text}")

    return df



def resolve_cluster_csv_path(cluster_csv_path: str, project_root: str | Path | None = None) -> Path:
    """Resolve a cluster csv path recorded in summary.csv."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    path = Path(cluster_csv_path)
    if path.is_absolute():
        return path
    return root / path



def summarize_cluster_csv(
    cluster_csv_path: str | Path,
    include_noise: bool = False,
) -> Dict[str, Any]:
    """Compute cluster count and cluster size statistics from one cluster csv."""
    csv_path = Path(cluster_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"cluster csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"token_id", "cluster_id", "probability"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"cluster csv is missing required columns: {missing_text}")

    cluster_ids = df["cluster_id"]
    noise_mask = cluster_ids == -1
    valid_df = df if include_noise else df.loc[~noise_mask]

    if valid_df.empty:
        cluster_sizes = pd.Series(dtype="int64")
    else:
        cluster_sizes = valid_df.groupby("cluster_id").size()

    n_clusters = int(cluster_sizes.shape[0])
    cluster_size_mean = float(cluster_sizes.mean()) if n_clusters > 0 else 0.0
    cluster_size_std = float(cluster_sizes.std(ddof=0)) if n_clusters > 0 else 0.0

    return {
        "n_clusters": n_clusters,
        "cluster_size_mean": cluster_size_mean,
        "cluster_size_std": cluster_size_std,
        "n_noise": int(noise_mask.sum()),
        "n_rows": int(df.shape[0]),
        "include_noise": bool(include_noise),
    }



def build_output_proj_cluster_stats(
    model_name: str,
    project_root: str | Path | None = None,
    include_noise: bool = False,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Build a reusable nested dict of cluster statistics by model and PCA dim."""
    summary_df = load_output_proj_summary(model_name=model_name, project_root=project_root)

    stats_by_dim: Dict[int, Dict[str, Any]] = {}
    for row in summary_df.itertuples(index=False):
        pca_dim = int(row.pca_dim)
        cluster_csv = resolve_cluster_csv_path(
            cluster_csv_path=row.cluster_csv_path,
            project_root=project_root,
        )

        cluster_stats = summarize_cluster_csv(
            cluster_csv_path=cluster_csv,
            include_noise=include_noise,
        )
        cluster_stats.update(
            {
                "model_name": str(row.model_name),
                "space_name": getattr(row, "space_name", "output_proj"),
                "pca_dim": pca_dim,
                "cluster_csv_path": str(cluster_csv),
            }
        )

        if hasattr(row, "run_id"):
            cluster_stats["run_id"] = int(row.run_id)
        if hasattr(row, "meta_json_path") and pd.notna(row.meta_json_path):
            meta_path = resolve_cluster_csv_path(
                cluster_csv_path=str(row.meta_json_path),
                project_root=project_root,
            )
            cluster_stats["meta_json_path"] = str(meta_path)

        stats_by_dim[pca_dim] = cluster_stats

    return {model_name: dict(sorted(stats_by_dim.items(), key=lambda item: item[0]))}



def get_model_dim_cluster_stats(
    model_name: str,
    pca_dim: int,
    project_root: str | Path | None = None,
    include_noise: bool = False,
) -> Dict[str, Any]:
    """Return cluster statistics for one model and one PCA dim."""
    stats = build_output_proj_cluster_stats(
        model_name=model_name,
        project_root=project_root,
        include_noise=include_noise,
    )
    dim_key = int(pca_dim)
    if dim_key not in stats[model_name]:
        raise KeyError(f"pca_dim not found for model {model_name}: {dim_key}")
    return stats[model_name][dim_key]
