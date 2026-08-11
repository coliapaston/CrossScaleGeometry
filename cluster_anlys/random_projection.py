from __future__ import annotations

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from pca_hdbscan import (
    apply_l2_normalization,
    build_token_cluster_df,
    cleanup_gpu_memory,
    prepare_embedding_matrix,
    run_gpu_hdbscan,
)


DEFAULT_SEEDS = [
    1813382118,
    827307999,
    1627694678,
    1911784257,
    903170602,
    86939546,
    556019485,
    2073320061,
    1097954097,
    1043521778,
]


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mistral": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "tokenizer_path": "mistralai/Mistral-7B-v0.1/tokenizer",
        "candidate_dims": [5, 142, 997, 2084, 3031, 3459, 3938, 4088, 4096],
    },
    "mixtral": {
        "model_name": "mistralai/Mixtral-8x7B-v0.1",
        "tokenizer_path": "mistralai/Mixtral-8x7B-v0.1/tokenizer",
        "candidate_dims": [8, 158, 1111, 2156, 3052, 3457, 3957, 4093, 4096],
    },
    "gpt-oss": {
        "model_name": "gpt-oss",
        "tokenizer_path": "gpt-oss/tokenizer",
        "candidate_dims": [6, 182, 466, 739, 1591, 2264, 2532, 2868, 2880],
    },
}


def random_projection_root(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
) -> Path:
    return Path(out_root) / model_name / space_name / "random_projection"


def random_projection_seed_dir(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    rp_seed: int,
) -> Path:
    return random_projection_root(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    ) / f"seed_{int(rp_seed)}"


def random_projection_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    rp_seed: int,
) -> Path:
    return random_projection_seed_dir(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        rp_seed=rp_seed,
    ) / "summary.csv"


def random_projection_all_seed_summary_path(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
) -> Path:
    return random_projection_root(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    ) / "summary_all_seeds.csv"


def full_pca_mean_path(
    *,
    out_root: str | Path,
    model_name: str,
) -> Path:
    return Path(out_root) / model_name / "full_pca" / "mean.npy"


def generate_random_orthogonal_basis(
    feature_dim: int,
    *,
    random_state: int,
) -> np.ndarray:
    feature_dim = int(feature_dim)
    if feature_dim < 1:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")

    rng = np.random.default_rng(int(random_state))
    gaussian_matrix = rng.standard_normal((feature_dim, feature_dim), dtype=np.float64)
    basis, upper = np.linalg.qr(gaussian_matrix, mode="reduced")

    diagonal = np.diag(upper)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    basis *= signs[np.newaxis, :]
    return np.asarray(basis, dtype=np.float64, order="C")


def write_full_random_projection(
    embedding_matrix: Any,
    *,
    mean: np.ndarray,
    basis: np.ndarray,
    outpath: str | Path,
    row_batch_size: int = 4096,
) -> np.memmap:
    matrix = prepare_embedding_matrix(embedding_matrix)
    n_rows, feature_dim = matrix.shape

    mean = np.asarray(mean, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if mean.shape != (feature_dim,):
        raise ValueError(
            f"Full-PCA mean shape mismatch: expected {(feature_dim,)}, got {mean.shape}"
        )
    if basis.shape != (feature_dim, feature_dim):
        raise ValueError(
            "Random basis shape mismatch: "
            f"expected {(feature_dim, feature_dim)}, got {basis.shape}"
        )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    projected = np.lib.format.open_memmap(
        outpath,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, feature_dim),
    )

    row_batch_size = max(1, int(row_batch_size))
    for row_start in range(0, n_rows, row_batch_size):
        row_stop = min(row_start + row_batch_size, n_rows)
        centered_chunk = np.asarray(
            matrix[row_start:row_stop],
            dtype=np.float64,
            order="C",
        )
        centered_chunk -= mean
        projected[row_start:row_stop] = centered_chunk @ basis
        projected.flush()
        print(f"Projected rows {row_start}:{row_stop} / {n_rows}")

    return projected


def _cluster_statistics(cluster_df: pd.DataFrame) -> Dict[str, Any]:
    labels = cluster_df["cluster_id"].to_numpy(dtype=np.int64)
    probabilities = cluster_df["probability"].to_numpy(dtype=np.float32)
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    n_total = int(labels.size)
    n_noise = int(np.sum(~non_noise_mask))

    if non_noise_labels.size:
        _, cluster_sizes = np.unique(non_noise_labels, return_counts=True)
        n_clusters = int(cluster_sizes.size)
        mean_cluster_size = float(np.mean(cluster_sizes))
        median_cluster_size = float(np.median(cluster_sizes))
    else:
        n_clusters = 0
        mean_cluster_size = 0.0
        median_cluster_size = 0.0

    return {
        "n_clusters_excl_noise": n_clusters,
        "n_noise": n_noise,
        "n_total": n_total,
        "noise_ratio": float(n_noise / n_total) if n_total else 0.0,
        "avg_prob_all": float(np.mean(probabilities)) if n_total else 0.0,
        "avg_prob_assigned": (
            float(np.mean(probabilities[non_noise_mask])) if np.any(non_noise_mask) else 0.0
        ),
        "mean_cluster_size": mean_cluster_size,
        "median_cluster_size": median_cluster_size,
    }


def _save_random_projection_run(
    *,
    cluster_df: pd.DataFrame,
    seed_dir: Path,
    run_id: int,
    model_name: str,
    space_name: str,
    rp_seed: int,
    matrix_shape: Sequence[int],
    pca_dim: int,
    l2_norm: bool,
    hdbscan_params: Dict[str, Any],
    mean_path: Path,
) -> Dict[str, Any]:
    clusters_dir = seed_dir / f"clusters_l2={bool(l2_norm)}"
    meta_dir = seed_dir / f"meta_l2={bool(l2_norm)}"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"run_{int(run_id):05d}"
    cluster_csv_path = clusters_dir / f"{run_name}_clusters.csv"
    meta_json_path = meta_dir / f"{run_name}_meta.json"
    cluster_df.to_csv(cluster_csv_path, index=False, encoding="utf-8")

    stats = _cluster_statistics(cluster_df)
    metadata = {
        "run_id": int(run_id),
        "model_name": model_name,
        "space_name": space_name,
        "projection_mode": "random_projection",
        "rp_seed": int(rp_seed),
        "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
        "pca_dim": int(pca_dim),
        "l2_norm": bool(l2_norm),
        "centering_mean_path": str(mean_path),
        "basis": {
            "distribution": "standard_normal_qr",
            "sign_normalized": True,
            "nested_prefix_slices": True,
        },
        "hdbscan": {
            "min_cluster_size": int(hdbscan_params["min_cluster_size"]),
            "min_samples": int(hdbscan_params["min_samples"]),
            "metric": str(hdbscan_params["metric"]),
            "cluster_selection_method": str(hdbscan_params["cluster_selection_method"]),
            "cluster_selection_epsilon": float(
                hdbscan_params["cluster_selection_epsilon"]
            ),
        },
        "stats": stats,
        "outputs": {
            "cluster_csv": str(cluster_csv_path),
            "meta_json": str(meta_json_path),
        },
        "timestamp_unix": time.time(),
    }
    with meta_json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return {
        "run_id": int(run_id),
        "model_name": model_name,
        "space_name": space_name,
        "projection_mode": "random_projection",
        "rp_seed": int(rp_seed),
        "pca_dim": int(pca_dim),
        "l2_norm": bool(l2_norm),
        "min_cluster_size": int(hdbscan_params["min_cluster_size"]),
        "min_samples": int(hdbscan_params["min_samples"]),
        "metric": str(hdbscan_params["metric"]),
        "cluster_selection_method": str(hdbscan_params["cluster_selection_method"]),
        "cluster_selection_epsilon": float(
            hdbscan_params["cluster_selection_epsilon"]
        ),
        **stats,
        "cluster_csv_path": str(cluster_csv_path),
        "meta_json_path": str(meta_json_path),
    }


def _load_complete_seed_summary(
    summary_path: Path,
    *,
    candidate_dims: Sequence[int],
) -> pd.DataFrame | None:
    if not summary_path.exists():
        return None

    summary_df = pd.read_csv(summary_path)
    if "pca_dim" not in summary_df.columns:
        return None
    available_dims = set(pd.to_numeric(summary_df["pca_dim"], errors="raise").astype(int))
    if available_dims != set(int(value) for value in candidate_dims):
        return None

    required_path_cols = ["cluster_csv_path", "meta_json_path"]
    if any(column not in summary_df.columns for column in required_path_cols):
        return None
    for column in required_path_cols:
        if not all(Path(value).exists() for value in summary_df[column].astype(str)):
            return None
    return summary_df.sort_values("pca_dim").reset_index(drop=True)


def run_random_projection_hdbscan_grid(
    *,
    embedding_matrix: Any,
    candidate_dims: Sequence[int],
    model_name: str,
    space_name: str,
    rp_seed: int,
    out_root: str | Path = "comp",
    l2_norm: bool = True,
    min_cluster_size: int = 5,
    min_samples: int = 5,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    row_batch_size: int = 4096,
    reuse_complete: bool = True,
) -> pd.DataFrame:
    matrix = prepare_embedding_matrix(embedding_matrix)
    _, feature_dim = matrix.shape
    candidate_dims = sorted(set(int(value) for value in candidate_dims))
    if not candidate_dims:
        raise ValueError("candidate_dims must not be empty")
    invalid_dims = [value for value in candidate_dims if value < 1 or value > feature_dim]
    if invalid_dims:
        raise ValueError(
            f"Invalid candidate dimensions {invalid_dims} for feature_dim={feature_dim}"
        )

    seed_dir = random_projection_seed_dir(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        rp_seed=rp_seed,
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = random_projection_summary_path(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        rp_seed=rp_seed,
    )
    if reuse_complete:
        existing_summary = _load_complete_seed_summary(
            summary_path,
            candidate_dims=candidate_dims,
        )
        if existing_summary is not None:
            print(f"Reusing complete random-projection summary: {summary_path}")
            return existing_summary

    mean_path = full_pca_mean_path(out_root=out_root, model_name=model_name)
    if not mean_path.exists():
        raise FileNotFoundError(f"Full-PCA mean not found: {mean_path}")
    mean = np.load(mean_path)
    basis = generate_random_orthogonal_basis(
        feature_dim,
        random_state=int(rp_seed),
    )
    hdbscan_params = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "metric": str(metric),
        "cluster_selection_method": str(cluster_selection_method),
        "cluster_selection_epsilon": float(cluster_selection_epsilon),
    }

    temporary_dir = Path(tempfile.mkdtemp(prefix=".rp_projection_", dir=seed_dir))
    projection_path = temporary_dir / "X_random_projection.npy"
    summary_records = []
    try:
        full_projection = write_full_random_projection(
            matrix,
            mean=mean,
            basis=basis,
            outpath=projection_path,
            row_batch_size=row_batch_size,
        )
        del basis
        gc.collect()

        for run_id, pca_dim in enumerate(candidate_dims, start=1):
            print(
                f"Random projection HDBSCAN | model={model_name} "
                f"seed={int(rp_seed)} dim={int(pca_dim)}"
            )
            projection_slice = np.asarray(
                full_projection[:, : int(pca_dim)],
                dtype=np.float32,
            )
            processed_slice = apply_l2_normalization(
                projection_slice,
                l2_norm=l2_norm,
            )
            labels, probabilities = run_gpu_hdbscan(
                processed_slice,
                **hdbscan_params,
            )
            cluster_df = build_token_cluster_df(labels, probabilities)
            summary_record = _save_random_projection_run(
                cluster_df=cluster_df,
                seed_dir=seed_dir,
                run_id=run_id,
                model_name=model_name,
                space_name=space_name,
                rp_seed=rp_seed,
                matrix_shape=processed_slice.shape,
                pca_dim=pca_dim,
                l2_norm=l2_norm,
                hdbscan_params=hdbscan_params,
                mean_path=mean_path,
            )
            summary_records.append(summary_record)
            pd.DataFrame(summary_records).to_csv(summary_path, index=False)

            del projection_slice, processed_slice, labels, probabilities, cluster_df
            cleanup_gpu_memory()

        summary_df = pd.DataFrame(summary_records).sort_values("pca_dim").reset_index(drop=True)
        summary_df.to_csv(summary_path, index=False)
        return summary_df
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        cleanup_gpu_memory()
        gc.collect()


def aggregate_random_projection_seed_summaries(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    seed_list: Sequence[int],
) -> pd.DataFrame:
    frames = []
    for rp_seed in seed_list:
        summary_path = random_projection_summary_path(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            rp_seed=int(rp_seed),
        )
        if not summary_path.exists():
            raise FileNotFoundError(f"Random-projection summary not found: {summary_path}")
        frame = pd.read_csv(summary_path)
        frame["rp_seed"] = int(rp_seed)
        frames.append(frame)

    all_seed_df = pd.concat(frames, axis=0, ignore_index=True)
    all_seed_df = all_seed_df.sort_values(["rp_seed", "pca_dim"]).reset_index(drop=True)
    outpath = random_projection_all_seed_summary_path(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    all_seed_df.to_csv(outpath, index=False)
    return all_seed_df
