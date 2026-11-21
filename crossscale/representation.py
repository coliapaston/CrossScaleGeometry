"""Representation kernels: randomized PCA with the repository's artifact layout."""

from __future__ import annotations

import gc
import json
import os
import time
from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA


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


def get_pca_artifact_paths(save_dir: str) -> Dict[str, str]:
    return {
        "x_pca": os.path.join(save_dir, "X_pca.npy"),
        "components": os.path.join(save_dir, "components.npy"),
        "explained_ratio": os.path.join(save_dir, "explained_ratio.npy"),
        "mean": os.path.join(save_dir, "mean.npy"),
        "meta": os.path.join(save_dir, "meta.json"),
    }


def run_randomized_pca(
    X_np: np.ndarray,
    pca_dim: int,
    pca_seed: int,
    model_name: str,
    out_root: str = "comp",
    n_oversamples: int = 10,
    power_iteration_normalizer: str = "auto",
    iterated_power: Any = "auto",
) -> None:
    """
    Run randomized PCA once and save results; nothing is returned.

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
