"""Partition kernels: row normalization, GPU HDBSCAN, and token-cluster tables."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pandas as pd


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

    import cupy as cp
    from cuml.cluster import HDBSCAN as cuHDBSCAN

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
