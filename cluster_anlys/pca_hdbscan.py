import os
import gc
import json
import time
import random
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
import pandas as pd
import cupy as cp

from sklearn.decomposition import PCA
from cuml.cluster import HDBSCAN as cuHDBSCAN


# =========================================================
# 1. Seed
# =========================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cp.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# 2. Input matrix prep
# =========================================================
def prepare_embedding_matrix(embedding_matrix: Any) -> np.ndarray:
    """
    Convert input embedding matrix to CPU numpy float32.

    Expected input:
        - torch.Tensor [V, D]
        - or numpy array-like

    Returns:
        X_np: np.ndarray [V, D], dtype=float32
    """
    if isinstance(embedding_matrix, torch.Tensor):
        X_np = embedding_matrix.detach().cpu().to(torch.float32).numpy()
    else:
        X_np = np.asarray(embedding_matrix, dtype=np.float32)

    if X_np.ndim != 2:
        raise ValueError(f"embedding_matrix must be 2D, got shape={X_np.shape}")

    return np.asarray(X_np, dtype=np.float32)


# =========================================================
# 3. Path helpers
# =========================================================
def get_full_pca_save_dir(
    out_root: str,
    model_name: str,
) -> str:
    """
    New canonical full PCA save dir.
    Kept identical to your previous layout for compatibility.
    """
    return os.path.join(out_root, model_name, "full_pca")


def get_random_pca_save_dir(
    out_root: str,
    model_name: str,
    pca_seed: int,
    pca_dim: int,
) -> str:
    """
    Random PCA save dir.

    Layout:
        comp/{model_name}/random_pca/seed_00042/pca_dim_256/
    """
    return os.path.join(
        out_root,
        model_name,
        "random_pca",
        f"seed_{int(pca_seed)}",
        f"pca_dim_{int(pca_dim)}",
    )


def get_cluster_base_dir(
    out_root: str,
    model_name: str,
    space_name: str,
    pca_mode: str,
    pca_seed: Optional[int] = None,
) -> str:
    """
    Clustering result base dir.

    full:
        comp/{model_name}/{space_name}/full/

    randomized:
        comp/{model_name}/{space_name}/random/seed_00042/
    """
    pca_mode = str(pca_mode)

    if pca_mode == "full":
        # return os.path.join(out_root, model_name, space_name, "full")
        return os.path.join(out_root, model_name, space_name)

    if pca_mode == "randomized":
        if pca_seed is None:
            raise ValueError("pca_seed must be provided when pca_mode='randomized'")
        return os.path.join(
            out_root,
            model_name,
            space_name,
            "random",
            f"seed_{int(pca_seed)}",
        )

    raise ValueError(f"Unsupported pca_mode={pca_mode!r}; expected 'full' or 'randomized'")


def get_pca_artifact_paths(save_dir: str) -> Dict[str, str]:
    return {
        "x_pca": os.path.join(save_dir, "X_pca.npy"),
        "components": os.path.join(save_dir, "components.npy"),
        "explained_ratio": os.path.join(save_dir, "explained_ratio.npy"),
        "mean": os.path.join(save_dir, "mean.npy"),
        "meta": os.path.join(save_dir, "meta.json"),
    }


# =========================================================
# 4. PCA runners
# =========================================================
def run_full_pca(
    X_np: np.ndarray,
    model_name: str,
    out_root: str = "comp",
    svd_solver: str = "full",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run deterministic FULL PCA once and save complete results.

    Save layout:
        comp/{model_name}/full_pca/
    """
    if X_np.ndim != 2:
        raise ValueError(f"X_np must be 2D, got shape={X_np.shape}")

    n_samples, n_features = X_np.shape
    full_dim = min(n_samples, n_features)

    if full_dim < 1:
        raise ValueError(f"Invalid matrix shape {X_np.shape}; full_dim must be >= 1")

    if svd_solver != "full":
        raise ValueError("run_full_pca enforces deterministic PCA (svd_solver='full')")

    X_np = np.asarray(X_np, dtype=np.float64, order="C")

    start = time.time()

    pca = PCA(
        n_components=full_dim,
        svd_solver="full",
    )
    X_pca = pca.fit_transform(X_np).astype(np.float32, copy=False)

    elapsed = time.time() - start
    explained_ratio = pca.explained_variance_ratio_.astype(np.float32, copy=False)
    cum_explained_ratio = float(np.sum(explained_ratio))

    pca_meta = {
        "model_name": model_name,
        "input_shape": [int(n_samples), int(n_features)],
        "pca_mode": "full",
        "full_dim": int(full_dim),
        "svd_solver": "full",
        "fit_time_sec": float(elapsed),
        "explained_variance_ratio_sum": cum_explained_ratio,
        "explained_variance_ratio_first10": explained_ratio[:10].tolist(),
        "saved_as": "full_pca",
    }

    save_dir = get_full_pca_save_dir(out_root=out_root, model_name=model_name)
    os.makedirs(save_dir, exist_ok=True)

    paths = get_pca_artifact_paths(save_dir)
    np.save(paths["x_pca"], X_pca)
    np.save(paths["components"], pca.components_)
    np.save(paths["explained_ratio"], explained_ratio)
    np.save(paths["mean"], pca.mean_)

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(pca_meta, f, ensure_ascii=False, indent=2)

    print(
        f"[✓] Full PCA finished | model={model_name} full_dim={full_dim} "
        f"time={elapsed:.2f}s cum_var={cum_explained_ratio:.6f}"
    )
    print(f"[✓] Saved to: {save_dir}")

    # clear large local objects no longer needed
    del X_pca
    del pca
    del explained_ratio
    del pca_meta
    gc.collect()
    print("Memory Cleaned.")


def run_randomized_pca(
    X_np: np.ndarray,
    pca_dim: int,
    pca_seed: int,
    model_name: str,
    out_root: str = "comp",
    n_oversamples: int = 10,
    power_iteration_normalizer: str = "auto",
    iterated_power: Any = "auto",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run randomized PCA once and save results.

    Save layout:
        comp/{model_name}/random_pca/seed_00042/pca_dim_256/
    """
    if X_np.ndim != 2:
        raise ValueError(f"X_np must be 2D, got shape={X_np.shape}")

    n_samples, n_features = X_np.shape
    max_dim = min(n_samples, n_features)

    if not (1 <= int(pca_dim) <= max_dim):
        raise ValueError(
            f"Invalid pca_dim={pca_dim}; must satisfy 1 <= pca_dim <= {max_dim}"
        )

    X_np = np.asarray(X_np, dtype=np.float32, order="C")

    start = time.time()

    pca = PCA(
        n_components=int(pca_dim),
        svd_solver="randomized",
        random_state=int(pca_seed),
        n_oversamples=int(n_oversamples),
        power_iteration_normalizer=power_iteration_normalizer,
        iterated_power=iterated_power,
    )
    X_pca = pca.fit_transform(X_np).astype(np.float32, copy=False)

    elapsed = time.time() - start
    explained_ratio = pca.explained_variance_ratio_.astype(np.float32, copy=False)
    cum_explained_ratio = float(np.sum(explained_ratio))

    pca_meta = {
        "model_name": model_name,
        "input_shape": [int(n_samples), int(n_features)],
        "pca_mode": "randomized",
        "pca_dim": int(pca_dim),
        "pca_seed": int(pca_seed),
        "svd_solver": "randomized",
        "fit_time_sec": float(elapsed),
        "explained_variance_ratio_sum": cum_explained_ratio,
        "explained_variance_ratio_first10": explained_ratio[:10].tolist(),
        "n_oversamples": int(n_oversamples),
        "power_iteration_normalizer": str(power_iteration_normalizer),
        "iterated_power": iterated_power,
        "saved_as": "randomized_pca",
    }

    save_dir = get_random_pca_save_dir(
        out_root=out_root,
        model_name=model_name,
        pca_seed=int(pca_seed),
        pca_dim=int(pca_dim),
    )
    os.makedirs(save_dir, exist_ok=True)

    paths = get_pca_artifact_paths(save_dir)
    np.save(paths["x_pca"], X_pca)
    np.save(paths["components"], pca.components_)
    np.save(paths["explained_ratio"], explained_ratio)
    np.save(paths["mean"], pca.mean_)

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(pca_meta, f, ensure_ascii=False, indent=2)

    print(
        f"[✓] Randomized PCA finished | model={model_name} "
        f"dim={pca_dim} seed={pca_seed} time={elapsed:.2f}s "
        f"cum_var={cum_explained_ratio:.6f}"
    )
    print(f"[✓] Saved to: {save_dir}")

    # clear large local objects no longer needed
    del X_pca
    del pca
    del explained_ratio
    del pca_meta
    gc.collect()
    print("Memory Cleaned.")



# =========================================================
# 5. PCA loaders
# =========================================================
def _load_pca_bundle_from_dir(save_dir: str) -> Dict[str, Any]:
    paths = get_pca_artifact_paths(save_dir)

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing PCA file: {path}")

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "X_pca": np.load(paths["x_pca"], mmap_mode="r"),
        "components": np.load(paths["components"], mmap_mode="r"),
        "explained_ratio": np.load(paths["explained_ratio"], mmap_mode="r"),
        "mean": np.load(paths["mean"], mmap_mode="r"),
        "meta": meta,
        "save_dir": save_dir,
    }


def load_full_pca_result(
    out_root: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Load full PCA result.

    Compatible with old layout:
        comp/{model_name}/full_pca/
    """
    save_dir = get_full_pca_save_dir(out_root=out_root, model_name=model_name)
    return _load_pca_bundle_from_dir(save_dir)


def load_random_pca_result(
    out_root: str,
    model_name: str,
    pca_seed: int,
    pca_dim: int,
) -> Dict[str, Any]:
    save_dir = get_random_pca_save_dir(
        out_root=out_root,
        model_name=model_name,
        pca_seed=int(pca_seed),
        pca_dim=int(pca_dim),
    )
    return _load_pca_bundle_from_dir(save_dir)


def load_saved_pca_result(
    out_root: str,
    model_name: str,
    pca_mode: str,
    pca_seed: Optional[int] = None,
    pca_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unified PCA loader.

    full:
        load_saved_pca_result(..., pca_mode='full')

    randomized:
        load_saved_pca_result(..., pca_mode='randomized', pca_seed=42, pca_dim=256)
    """
    pca_mode = str(pca_mode)

    if pca_mode == "full":
        return load_full_pca_result(out_root=out_root, model_name=model_name)

    if pca_mode == "randomized":
        if pca_seed is None:
            raise ValueError("pca_seed must be provided when pca_mode='randomized'")
        if pca_dim is None:
            raise ValueError("pca_dim must be provided when pca_mode='randomized'")
        return load_random_pca_result(
            out_root=out_root,
            model_name=model_name,
            pca_seed=int(pca_seed),
            pca_dim=int(pca_dim),
        )

    raise ValueError(f"Unsupported pca_mode={pca_mode!r}; expected 'full' or 'randomized'")


# =========================================================
# 6. L2 normalization
# =========================================================
def apply_l2_normalization(
    X_np: np.ndarray,
    l2_norm: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Row-wise L2 normalization AFTER PCA.
    """
    if not l2_norm:
        return np.asarray(X_np, dtype=np.float32)

    norms = np.linalg.norm(X_np, axis=1, keepdims=True).astype(np.float32)
    X_out = X_np / (norms + np.float32(eps))
    return X_out.astype(np.float32, copy=False)


# =========================================================
# 7. GPU HDBSCAN
# =========================================================
def run_gpu_hdbscan(
    X_np: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run GPU cuML HDBSCAN.
    """
    print(
        "[i] HDBSCAN params | "
        f"min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples}, "
        f"metric={metric}, "
        f"cluster_selection_method={cluster_selection_method}, "
        f"cluster_selection_epsilon={cluster_selection_epsilon}"
    )

    X = cp.asarray(np.asarray(X_np, dtype=np.float32))

    start = time.time()
    clusterer = cuHDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        prediction_data=False,
        output_type="cupy",
    )
    clusterer.fit(X)
    elapsed = time.time() - start
    print(f"[✓] cuML HDBSCAN finished ({elapsed:.2f}s)")

    labels_np = cp.asnumpy(clusterer.labels_).astype(np.int64, copy=False)
    probs_np = cp.asnumpy(clusterer.probabilities_).astype(np.float32, copy=False)

    try:
        del X, clusterer
    except Exception:
        pass

    return labels_np, probs_np


# =========================================================
# 8. Token-level output table
# =========================================================
def build_token_cluster_df(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
) -> pd.DataFrame:
    """
    Build minimal token-level cluster DataFrame.

    Columns:
        token_id
        cluster_id
        probability
    """
    if labels_np.shape[0] != probs_np.shape[0]:
        raise ValueError(
            f"labels/probs size mismatch: {labels_np.shape[0]} vs {probs_np.shape[0]}"
        )

    return pd.DataFrame(
        {
            "token_id": np.arange(labels_np.shape[0], dtype=np.int64),
            "cluster_id": labels_np.astype(np.int64, copy=False),
            "probability": probs_np.astype(np.float32, copy=False),
        }
    )


# =========================================================
# 9. HDBSCAN param grid
# =========================================================
def make_hdbscan_param_grid(
    min_cluster_size_list: Sequence[int],
    min_samples_list: Sequence[int],
    metric_list: Sequence[str] = ("euclidean",),
    cluster_selection_method_list: Sequence[str] = ("eom",),
    cluster_selection_epsilon_list: Sequence[float] = (0.0,),
) -> List[Dict[str, Any]]:
    """
    Build a list of HDBSCAN parameter dictionaries.
    """
    grid: List[Dict[str, Any]] = []

    for vals in product(
        min_cluster_size_list,
        min_samples_list,
        metric_list,
        cluster_selection_method_list,
        cluster_selection_epsilon_list,
    ):
        (
            min_cluster_size,
            min_samples,
            metric,
            cluster_selection_method,
            cluster_selection_epsilon,
        ) = vals

        grid.append(
            {
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "metric": str(metric),
                "cluster_selection_method": str(cluster_selection_method),
                "cluster_selection_epsilon": float(cluster_selection_epsilon),
            }
        )

    return grid


# =========================================================
# 10. Cleanup
# =========================================================
def cleanup_gpu_memory() -> None:
    gc.collect()

    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# =========================================================
# 11. Save clustering outputs
# =========================================================
def save_run_outputs(
    df: pd.DataFrame,
    out_root: str,
    model_name: str,
    space_name: str,
    pca_mode: str,
    run_id: int,
    matrix_shape: Tuple[int, int],
    pca_dim: int,
    l2_norm: bool,
    hdbscan_params: Dict[str, Any],
    pca_seed: Optional[int] = None,
    pca_source_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Save:
        - token-level cluster CSV
        - run-level metadata JSON
        - return one summary record
    """
    base_dir = get_cluster_base_dir(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_mode=pca_mode,
        pca_seed=pca_seed,
    )

    clusters_dir = os.path.join(base_dir, f"clusters_l2={l2_norm}")
    meta_dir = os.path.join(base_dir, f"meta_l2={l2_norm}")

    os.makedirs(clusters_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    run_name = f"run_{run_id:05d}"
    cluster_csv_path = os.path.join(clusters_dir, f"{run_name}_clusters.csv")
    meta_json_path = os.path.join(meta_dir, f"{run_name}_meta.json")

    df.to_csv(cluster_csv_path, index=False, encoding="utf-8")

    labels_np = df["cluster_id"].to_numpy()
    probs_np = df["probability"].to_numpy()

    unique_clusters = np.unique(labels_np)
    n_clusters_excl_noise = int(np.sum(unique_clusters >= 0))
    n_noise = int(np.sum(labels_np == -1))
    n_total = int(labels_np.shape[0])
    noise_ratio = float(n_noise / n_total) if n_total > 0 else 0.0
    avg_prob_all = float(np.mean(probs_np)) if n_total > 0 else 0.0

    assigned_mask = labels_np != -1
    avg_prob_assigned = (
        float(np.mean(probs_np[assigned_mask])) if np.any(assigned_mask) else 0.0
    )

    meta = {
        "run_id": int(run_id),
        "model_name": model_name,
        "space_name": space_name,
        "pca_mode": str(pca_mode),
        "pca_seed": int(pca_seed) if pca_seed is not None else None,
        "pca_source_dim": int(pca_source_dim) if pca_source_dim is not None else None,
        "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
        "pca_dim": int(pca_dim),
        "l2_norm": bool(l2_norm),
        "hdbscan": {
            "min_cluster_size": int(hdbscan_params["min_cluster_size"]),
            "min_samples": int(hdbscan_params["min_samples"]),
            "metric": str(hdbscan_params["metric"]),
            "cluster_selection_method": str(
                hdbscan_params["cluster_selection_method"]
            ),
            "cluster_selection_epsilon": float(
                hdbscan_params["cluster_selection_epsilon"]
            ),
        },
        "outputs": {
            "cluster_csv": cluster_csv_path,
            "meta_json": meta_json_path,
        },
        "stats": {
            "n_clusters_excl_noise": n_clusters_excl_noise,
            "n_noise": n_noise,
            "n_total": n_total,
            "noise_ratio": noise_ratio,
            "avg_prob_all": avg_prob_all,
            "avg_prob_assigned": avg_prob_assigned,
        },
        "timestamp_unix": time.time(),
    }

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[✓] CSV  → {cluster_csv_path}")
    print(f"[✓] META → {meta_json_path}")

    summary_record = {
        "run_id": int(run_id),
        "model_name": model_name,
        "space_name": space_name,
        "pca_mode": str(pca_mode),
        "pca_seed": int(pca_seed) if pca_seed is not None else None,
        "pca_source_dim": int(pca_source_dim) if pca_source_dim is not None else None,
        "pca_dim": int(pca_dim),
        "l2_norm": bool(l2_norm),
        "min_cluster_size": int(hdbscan_params["min_cluster_size"]),
        "min_samples": int(hdbscan_params["min_samples"]),
        "metric": str(hdbscan_params["metric"]),
        "cluster_selection_method": str(hdbscan_params["cluster_selection_method"]),
        "cluster_selection_epsilon": float(
            hdbscan_params["cluster_selection_epsilon"]
        ),
        "n_clusters_excl_noise": n_clusters_excl_noise,
        "n_noise": n_noise,
        "n_total": n_total,
        "noise_ratio": noise_ratio,
        "avg_prob_all": avg_prob_all,
        "avg_prob_assigned": avg_prob_assigned,
        "cluster_csv_path": cluster_csv_path,
        "meta_json_path": meta_json_path,
    }

    return summary_record


# =========================================================
# 12. Generic clustering runner from saved PCA
# =========================================================
def run_saved_pca_hdbscan_grid(
    candidate_dims: Sequence[int],
    hdbscan_param_grid: Sequence[Dict[str, Any]],
    model_name: str,
    space_name: str,
    pca_mode: str,
    out_root: str = "comp",
    l2_norm: bool = True,
    pca_seed: Optional[int] = None,
    pca_source_dim: Optional[int] = None,
    summary_filename: Optional[str] = None,
    pca_source_root: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generic clustering runner for both full PCA and randomized PCA.

    full:
        load one full PCA result, then slice [:, :pca_dim]

    randomized:
        for each pca_dim, load the exact saved randomized PCA result at that dim
        (no slicing from a larger randomized PCA result)

    By default PCA inputs and clustering outputs share out_root for backward
    compatibility. Set pca_source_root to read PCA inputs from a separate root.
    """
    pca_mode = str(pca_mode)
    resolved_pca_source_root = out_root if pca_source_root is None else pca_source_root

    candidate_dims = sorted(set(int(d) for d in candidate_dims))
    if len(candidate_dims) == 0:
        raise ValueError("candidate_dims is empty")

    summary_records: List[Dict[str, Any]] = []
    run_id = 0

    if pca_mode == "full":
        pca_bundle = load_saved_pca_result(
            out_root=resolved_pca_source_root,
            model_name=model_name,
            pca_mode="full",
        )
        X_pca_full = pca_bundle["X_pca"]

        if X_pca_full.ndim != 2:
            raise ValueError(f"Expected X_pca to be 2D, got shape={X_pca_full.shape}")

        _, available_dim = X_pca_full.shape
        invalid_dims = [d for d in candidate_dims if d < 1 or d > available_dim]
        if invalid_dims:
            raise ValueError(
                f"Invalid candidate_dims={invalid_dims}; available PCA dim={available_dim}"
            )

        print(f"[i] Loaded PCA from: {pca_bundle['save_dir']}")
        print(f"[i] PCA mode: full")
        print(f"[i] PCA shape: {X_pca_full.shape}")
        print(f"[i] Candidate dims: {candidate_dims}")
        print(f"[i] Number of HDBSCAN configs: {len(hdbscan_param_grid)}")

        for pca_dim in candidate_dims:
            X_slice = np.asarray(X_pca_full[:, :pca_dim], dtype=np.float32)
            X_proc = apply_l2_normalization(X_slice, l2_norm=l2_norm)

            for hdbscan_params in hdbscan_param_grid:
                run_id += 1
                print(
                    f"\n[i] Run {run_id:05d} | "
                    f"pca_mode=full dim={pca_dim} params={hdbscan_params}"
                )

                labels_np, probs_np = run_gpu_hdbscan(
                    X_np=X_proc,
                    min_cluster_size=hdbscan_params["min_cluster_size"],
                    min_samples=hdbscan_params["min_samples"],
                    metric=hdbscan_params["metric"],
                    cluster_selection_method=hdbscan_params["cluster_selection_method"],
                    cluster_selection_epsilon=hdbscan_params["cluster_selection_epsilon"],
                )

                df = build_token_cluster_df(labels_np, probs_np)

                summary_record = save_run_outputs(
                    df=df,
                    out_root=out_root,
                    model_name=model_name,
                    space_name=space_name,
                    pca_mode="full",
                    run_id=run_id,
                    matrix_shape=X_proc.shape,
                    pca_dim=int(pca_dim),
                    l2_norm=bool(l2_norm),
                    hdbscan_params=hdbscan_params,
                    pca_seed=None,
                    pca_source_dim=None,
                )
                summary_records.append(summary_record)

                del labels_np, probs_np, df
                cleanup_gpu_memory()

            del X_slice, X_proc
            cleanup_gpu_memory()

    elif pca_mode == "randomized":
        if pca_seed is None:
            raise ValueError("pca_seed must be provided when pca_mode='randomized'")

        print(f"[i] PCA mode: randomized")
        print(f"[i] PCA seed: {pca_seed}")
        print(f"[i] Candidate dims: {candidate_dims}")
        print(f"[i] Number of HDBSCAN configs: {len(hdbscan_param_grid)}")

        for pca_dim in candidate_dims:
            pca_bundle = load_saved_pca_result(
                out_root=resolved_pca_source_root,
                model_name=model_name,
                pca_mode="randomized",
                pca_seed=int(pca_seed),
                pca_dim=int(pca_dim),
            )
            X_pca_exact = pca_bundle["X_pca"]

            if X_pca_exact.ndim != 2:
                raise ValueError(f"Expected X_pca to be 2D, got shape={X_pca_exact.shape}")

            if X_pca_exact.shape[1] != int(pca_dim):
                raise ValueError(
                    f"Loaded randomized PCA dim mismatch: expected {pca_dim}, "
                    f"got {X_pca_exact.shape[1]}"
                )

            print(f"[i] Loaded PCA from: {pca_bundle['save_dir']}")
            print(f"[i] PCA shape: {X_pca_exact.shape}")

            X_proc = apply_l2_normalization(
                np.asarray(X_pca_exact, dtype=np.float32),
                l2_norm=l2_norm,
            )

            for hdbscan_params in hdbscan_param_grid:
                run_id += 1
                print(
                    f"\n[i] Run {run_id:05d} | "
                    f"pca_mode=randomized dim={pca_dim} seed={pca_seed} "
                    f"params={hdbscan_params}"
                )

                labels_np, probs_np = run_gpu_hdbscan(
                    X_np=X_proc,
                    min_cluster_size=hdbscan_params["min_cluster_size"],
                    min_samples=hdbscan_params["min_samples"],
                    metric=hdbscan_params["metric"],
                    cluster_selection_method=hdbscan_params["cluster_selection_method"],
                    cluster_selection_epsilon=hdbscan_params["cluster_selection_epsilon"],
                )

                df = build_token_cluster_df(labels_np, probs_np)

                summary_record = save_run_outputs(
                    df=df,
                    out_root=out_root,
                    model_name=model_name,
                    space_name=space_name,
                    pca_mode="randomized",
                    run_id=run_id,
                    matrix_shape=X_proc.shape,
                    pca_dim=int(pca_dim),
                    l2_norm=bool(l2_norm),
                    hdbscan_params=hdbscan_params,
                    pca_seed=int(pca_seed),
                    pca_source_dim=int(pca_dim),
                )
                summary_records.append(summary_record)

                del labels_np, probs_np, df
                cleanup_gpu_memory()

            del X_proc
            cleanup_gpu_memory()

    else:
        raise ValueError(f"Unsupported pca_mode={pca_mode!r}; expected 'full' or 'randomized'")

    summary_df = pd.DataFrame(summary_records)

    base_dir = get_cluster_base_dir(
        out_root=out_root,
        model_name=model_name,
        space_name=space_name,
        pca_mode=pca_mode,
        pca_seed=pca_seed,
    )
    os.makedirs(base_dir, exist_ok=True)

    if summary_filename is None:
        if pca_mode == "full":
            summary_filename = "summary_full_pca.csv"
        else:
            summary_filename = f"summary_random_pca_seed_{int(pca_seed)}.csv"

    summary_path = os.path.join(base_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"\n[✓] SUMMARY → {summary_path}")
    return summary_df


# =========================================================
# 13. One-shot wrappers
# =========================================================
def run_full_pca_hdbscan_grid(
    embedding_matrix: Any,
    candidate_dims: Sequence[int],
    hdbscan_param_grid: Sequence[Dict[str, Any]],
    model_name: str,
    space_name: str,
    out_root: str = "comp",
    l2_norm: bool = True,
    summary_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-shot full PCA -> clustering.
    """
    X_np = prepare_embedding_matrix(embedding_matrix)

    run_full_pca(
        X_np=X_np,
        model_name=model_name,
        out_root=out_root,
        svd_solver="full",
    )

    return run_saved_pca_hdbscan_grid(
        candidate_dims=candidate_dims,
        hdbscan_param_grid=hdbscan_param_grid,
        model_name=model_name,
        space_name=space_name,
        pca_mode="full",
        out_root=out_root,
        l2_norm=l2_norm,
        summary_filename=summary_filename,
    )


def run_random_pca_hdbscan_grid(
    embedding_matrix: Any,
    candidate_dims: Sequence[int],
    hdbscan_param_grid: Sequence[Dict[str, Any]],
    model_name: str,
    space_name: str,
    pca_seed: int,
    out_root: str = "comp",
    l2_norm: bool = True,
    randomized_fit_dim: Optional[int] = None,   # kept only for compatibility; ignored
    summary_filename: Optional[str] = None,
    n_oversamples: int = 10,
    power_iteration_normalizer: str = "auto",
    iterated_power: Any = "auto",
) -> pd.DataFrame:
    """
    One-shot randomized PCA perturbation experiment.

    Correct behavior:
        for each pca_dim in candidate_dims:
            run exact randomized PCA at that dim with this seed
            save it
        then load each exact pca_dim result and run HDBSCAN
    """
    X_np = prepare_embedding_matrix(embedding_matrix)

    candidate_dims = sorted(set(int(d) for d in candidate_dims))
    if len(candidate_dims) == 0:
        raise ValueError("candidate_dims is empty")

    if randomized_fit_dim is not None:
        print(
            "[warn] randomized_fit_dim is ignored in the corrected per-dim randomized PCA pipeline."
        )

    for pca_dim in candidate_dims:
        run_randomized_pca(
            X_np=X_np,
            pca_dim=int(pca_dim),
            pca_seed=int(pca_seed),
            model_name=model_name,
            out_root=out_root,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            iterated_power=iterated_power,
        )

    return run_saved_pca_hdbscan_grid(
        candidate_dims=candidate_dims,
        hdbscan_param_grid=hdbscan_param_grid,
        model_name=model_name,
        space_name=space_name,
        pca_mode="randomized",
        out_root=out_root,
        l2_norm=l2_norm,
        pca_seed=int(pca_seed),
        pca_source_dim=None,   # no longer used for randomized mode
        summary_filename=summary_filename,
    )


# =========================================================
# 14. Backward-compatible aliases
# =========================================================
def run_pca(
    X_np: np.ndarray,
    model_name: str,
    svd_solver: str = "full",
    out_root: str = "comp",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Backward-compatible alias.
    """
    return run_full_pca(
        X_np=X_np,
        model_name=model_name,
        out_root=out_root,
        svd_solver=svd_solver,
    )


def run_pca_hdbscan_grid(
    candidate_dims: Sequence[int],
    hdbscan_param_grid: Sequence[Dict[str, Any]],
    model_name: str,
    space_name: str,
    full_pca_dim: Optional[int] = None,  # kept for compatibility; ignored
    out_root: str = "comp",
    l2_norm: bool = True,
    summary_filename: str = "summary.csv",
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for the old full-PCA-only interface.

    Notes:
        - full_pca_dim is kept only to avoid breaking old notebook calls.
        - this function now simply loads the saved full PCA and runs clustering.
    """
    return run_saved_pca_hdbscan_grid(
        candidate_dims=candidate_dims,
        hdbscan_param_grid=hdbscan_param_grid,
        model_name=model_name,
        space_name=space_name,
        pca_mode="full",
        out_root=out_root,
        l2_norm=l2_norm,
        summary_filename=summary_filename,
    )


# =========================================================
# 15. Example
# =========================================================
if __name__ == "__main__":
    print("Unified PCA -> L2 -> GPU HDBSCAN pipeline")
    print("Supported PCA modes: full, randomized")
    print("Canonical clustering dirs:")
    print("  full       : comp/{model}/{space}/full/")
    print("  randomized : comp/{model}/{space}/random/seed_XXXXX/")
