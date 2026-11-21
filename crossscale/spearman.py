from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SUMMARY_KEY_COLS = [
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


@dataclass(frozen=True)
class SpearmanResult:
    rho: float
    pvalue: float
    n_obs: int


@dataclass(frozen=True)
class PermutationResult:
    rho: float
    perm_pvalue: float
    n_obs: int
    n_permutations: int


def _ensure_required_columns(df: pd.DataFrame, required_cols: Sequence[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} missing required columns: {missing}")


def load_full_morphology_summary(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    filename: str = "kondrak_global_summary.csv",
) -> pd.DataFrame:
    path = Path(out_root) / model_name / space_name / filename
    if not path.exists():
        raise FileNotFoundError(f"Full morphology summary not found: {path}")
    return pd.read_csv(path)


def load_random_morphology_summary(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    seed: int,
    filename: str = "kondrak_global_summary.csv",
) -> pd.DataFrame:
    path = Path(out_root) / model_name / space_name / "random" / f"seed_{seed}" / filename
    if not path.exists():
        raise FileNotFoundError(f"Random morphology summary not found: {path}")
    return pd.read_csv(path)


def load_full_script_summary(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    filename: str = "script_entropy_summary.csv",
) -> pd.DataFrame:
    path = Path(out_root) / model_name / space_name / filename
    if not path.exists():
        raise FileNotFoundError(f"Full script summary not found: {path}")
    return pd.read_csv(path)


def load_random_script_summary(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    seed: int,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    seed_dir = Path(out_root) / model_name / space_name / "random" / f"seed_{seed}"
    if filename is None:
        filename = f"summary_random_pca_seed_{seed}_script_entropy.csv"
    path = seed_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Random script summary not found: {path}")
    return pd.read_csv(path)


def standardize_morphology_summary_df(df: pd.DataFrame, *, df_name: str = "morphology_summary") -> pd.DataFrame:
    required_cols = SUMMARY_KEY_COLS + ["global_mean", "global_std"]
    _ensure_required_columns(df, required_cols, df_name)

    out = df[SUMMARY_KEY_COLS + ["global_mean", "global_std"]].copy()
    out = out.rename(columns={"global_mean": "mean_value", "global_std": "std_value"})
    out["pca_dim"] = pd.to_numeric(out["pca_dim"], errors="raise").astype(int)
    out["mean_value"] = pd.to_numeric(out["mean_value"], errors="raise").astype(float)
    out["std_value"] = pd.to_numeric(out["std_value"], errors="raise").astype(float)
    out = out.sort_values("pca_dim").reset_index(drop=True)
    return out


def standardize_script_summary_df(df: pd.DataFrame, *, df_name: str = "script_summary") -> pd.DataFrame:
    required_cols = SUMMARY_KEY_COLS + ["mean_H", "std_H"]
    _ensure_required_columns(df, required_cols, df_name)

    out = df[SUMMARY_KEY_COLS + ["mean_H", "std_H"]].copy()
    out = out.rename(columns={"mean_H": "mean_value", "std_H": "std_value"})
    out["pca_dim"] = pd.to_numeric(out["pca_dim"], errors="raise").astype(int)
    out["mean_value"] = pd.to_numeric(out["mean_value"], errors="raise").astype(float)
    out["std_value"] = pd.to_numeric(out["std_value"], errors="raise").astype(float)
    out = out.sort_values("pca_dim").reset_index(drop=True)
    return out


def align_baseline_and_seed_on_pca_dim(
    baseline_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    *,
    baseline_label: str = "baseline",
    seed_label: str = "seed",
) -> pd.DataFrame:
    required_cols = ["pca_dim", "mean_value", "std_value"]
    _ensure_required_columns(baseline_df, required_cols, baseline_label)
    _ensure_required_columns(seed_df, required_cols, seed_label)

    if baseline_df["pca_dim"].duplicated().any():
        raise ValueError(f"{baseline_label} has duplicate pca_dim values")
    if seed_df["pca_dim"].duplicated().any():
        raise ValueError(f"{seed_label} has duplicate pca_dim values")

    merged = baseline_df[["pca_dim", "mean_value", "std_value"]].merge(
        seed_df[["pca_dim", "mean_value", "std_value"]],
        on="pca_dim",
        how="inner",
        suffixes=("_baseline", "_seed"),
    )

    if merged.empty:
        raise ValueError("No overlapping pca_dim values between baseline and seed summaries.")

    merged = merged.sort_values("pca_dim").reset_index(drop=True)
    return merged


def align_joint_mean_df(morph_df: pd.DataFrame, script_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["pca_dim", "mean_value"]
    _ensure_required_columns(morph_df, required_cols, "morph_df")
    _ensure_required_columns(script_df, required_cols, "script_df")

    merged = morph_df[["pca_dim", "mean_value"]].merge(
        script_df[["pca_dim", "mean_value"]],
        on="pca_dim",
        how="inner",
        suffixes=("_morph", "_script"),
    )
    if merged.empty:
        raise ValueError("No overlapping pca_dim values between morphology and script summaries.")
    return merged.sort_values("pca_dim").reset_index(drop=True)


def _safe_standardize_against_baseline(baseline_values: np.ndarray, seed_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    baseline_values = np.asarray(baseline_values, dtype=float)
    seed_values = np.asarray(seed_values, dtype=float)

    mean = float(np.mean(baseline_values))
    std = float(np.std(baseline_values, ddof=0))
    if np.isclose(std, 0.0):
        return baseline_values - mean, seed_values - mean
    return (baseline_values - mean) / std, (seed_values - mean) / std


def build_mean_vectors(aligned_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return (
        aligned_df["mean_value_baseline"].to_numpy(dtype=float),
        aligned_df["mean_value_seed"].to_numpy(dtype=float),
    )


def build_std_vectors(aligned_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return (
        aligned_df["std_value_baseline"].to_numpy(dtype=float),
        aligned_df["std_value_seed"].to_numpy(dtype=float),
    )


def build_joint_mean_std_vectors(aligned_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    baseline_mean, seed_mean = build_mean_vectors(aligned_df)
    baseline_std, seed_std = build_std_vectors(aligned_df)
    baseline = np.concatenate([baseline_mean, baseline_std])
    seed = np.concatenate([seed_mean, seed_std])
    return baseline, seed


def build_joint_mean_zscore_vectors(
    baseline_morph_df: pd.DataFrame,
    seed_morph_df: pd.DataFrame,
    baseline_script_df: pd.DataFrame,
    seed_script_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    morph_aligned = align_baseline_and_seed_on_pca_dim(
        baseline_morph_df,
        seed_morph_df,
        baseline_label="baseline_morph",
        seed_label="seed_morph",
    )
    script_aligned = align_baseline_and_seed_on_pca_dim(
        baseline_script_df,
        seed_script_df,
        baseline_label="baseline_script",
        seed_label="seed_script",
    )

    baseline_joint = align_joint_mean_df(
        morph_aligned.rename(columns={"mean_value_baseline": "mean_value"}),
        script_aligned.rename(columns={"mean_value_baseline": "mean_value"}),
    ).rename(columns={"mean_value_morph": "morph_mean", "mean_value_script": "script_mean"})

    seed_joint = align_joint_mean_df(
        morph_aligned.rename(columns={"mean_value_seed": "mean_value"}),
        script_aligned.rename(columns={"mean_value_seed": "mean_value"}),
    ).rename(columns={"mean_value_morph": "morph_mean", "mean_value_script": "script_mean"})

    joint = baseline_joint.merge(seed_joint, on="pca_dim", suffixes=("_baseline", "_seed"))
    if joint.empty:
        raise ValueError("No overlapping pca_dim values available for joint mean analysis.")
    joint = joint.sort_values("pca_dim").reset_index(drop=True)

    z_morph_baseline, z_morph_seed = _safe_standardize_against_baseline(
        joint["morph_mean_baseline"].to_numpy(dtype=float),
        joint["morph_mean_seed"].to_numpy(dtype=float),
    )
    z_script_baseline, z_script_seed = _safe_standardize_against_baseline(
        joint["script_mean_baseline"].to_numpy(dtype=float),
        joint["script_mean_seed"].to_numpy(dtype=float),
    )

    baseline_vec = np.concatenate([z_morph_baseline, z_script_baseline])
    seed_vec = np.concatenate([z_morph_seed, z_script_seed])
    return baseline_vec, seed_vec, baseline_joint, seed_joint


def compute_spearman_stats(x: Sequence[float], y: Sequence[float]) -> SpearmanResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Spearman inputs must have the same shape, got {x.shape} vs {y.shape}")
    if x.size < 2:
        raise ValueError(f"Spearman requires at least 2 observations, got n={x.size}")

    rho, pvalue = spearmanr(x, y)
    return SpearmanResult(rho=float(rho), pvalue=float(pvalue), n_obs=int(x.size))


def compute_grouped_permutation_pvalue(
    baseline_feature_matrix: np.ndarray,
    seed_feature_matrix: np.ndarray,
    *,
    n_permutations: int = 10000,
    random_state: Optional[int] = 0,
) -> PermutationResult:
    baseline_feature_matrix = np.asarray(baseline_feature_matrix, dtype=float)
    seed_feature_matrix = np.asarray(seed_feature_matrix, dtype=float)

    if baseline_feature_matrix.shape != seed_feature_matrix.shape:
        raise ValueError(
            "Baseline and seed feature matrices must share the same shape, "
            f"got {baseline_feature_matrix.shape} vs {seed_feature_matrix.shape}"
        )
    if baseline_feature_matrix.ndim != 2:
        raise ValueError("Grouped permutation expects 2D feature matrices.")
    if baseline_feature_matrix.shape[0] < 2:
        raise ValueError("Grouped permutation requires at least 2 pca_dim observations.")

    baseline_vec = baseline_feature_matrix.reshape(-1)
    seed_vec = seed_feature_matrix.reshape(-1)
    observed_rho = float(spearmanr(baseline_vec, seed_vec).statistic)

    rng = np.random.default_rng(random_state)
    exceed_count = 0
    n_rows = baseline_feature_matrix.shape[0]

    for _ in range(int(n_permutations)):
        perm_idx = rng.permutation(n_rows)
        perm_seed_vec = seed_feature_matrix[perm_idx, :].reshape(-1)
        perm_rho = float(spearmanr(baseline_vec, perm_seed_vec).statistic)
        if abs(perm_rho) >= abs(observed_rho):
            exceed_count += 1

    perm_pvalue = (exceed_count + 1) / (int(n_permutations) + 1)
    return PermutationResult(
        rho=observed_rho,
        perm_pvalue=float(perm_pvalue),
        n_obs=int(n_rows),
        n_permutations=int(n_permutations),
    )


def compare_seed_to_baseline_for_metric(
    baseline_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    *,
    metric_family: str,
    seed: int,
    n_permutations: int = 10000,
    random_state: Optional[int] = 0,
) -> Dict[str, float | int | str]:
    aligned = align_baseline_and_seed_on_pca_dim(
        baseline_df,
        seed_df,
        baseline_label=f"baseline_{metric_family}",
        seed_label=f"seed_{metric_family}",
    )

    mean_result = compute_spearman_stats(*build_mean_vectors(aligned))
    std_result = compute_spearman_stats(*build_std_vectors(aligned))
    mean_std_perm = compute_grouped_permutation_pvalue(
        baseline_feature_matrix=np.column_stack([
            aligned["mean_value_baseline"].to_numpy(dtype=float),
            aligned["std_value_baseline"].to_numpy(dtype=float),
        ]),
        seed_feature_matrix=np.column_stack([
            aligned["mean_value_seed"].to_numpy(dtype=float),
            aligned["std_value_seed"].to_numpy(dtype=float),
        ]),
        n_permutations=n_permutations,
        random_state=random_state,
    )

    return {
        "seed": int(seed),
        "metric_family": str(metric_family),
        "n_dims": int(len(aligned)),
        "mean_rho": mean_result.rho,
        "mean_p": mean_result.pvalue,
        "std_rho": std_result.rho,
        "std_p": std_result.pvalue,
        "mean_std_joint_rho": mean_std_perm.rho,
        "mean_std_joint_perm_p": mean_std_perm.perm_pvalue,
        "mean_std_joint_n_permutations": mean_std_perm.n_permutations,
    }


def compare_seed_to_baseline_joint_mean(
    baseline_morph_df: pd.DataFrame,
    seed_morph_df: pd.DataFrame,
    baseline_script_df: pd.DataFrame,
    seed_script_df: pd.DataFrame,
    *,
    seed: int,
    n_permutations: int = 10000,
    random_state: Optional[int] = 0,
) -> Dict[str, float | int | str]:
    baseline_vec, seed_vec, baseline_joint_df, seed_joint_df = build_joint_mean_zscore_vectors(
        baseline_morph_df,
        seed_morph_df,
        baseline_script_df,
        seed_script_df,
    )

    rho = float(spearmanr(baseline_vec, seed_vec).statistic)

    z_morph_baseline, z_morph_seed = _safe_standardize_against_baseline(
        baseline_joint_df["morph_mean"].to_numpy(dtype=float),
        seed_joint_df["morph_mean"].to_numpy(dtype=float),
    )
    z_script_baseline, z_script_seed = _safe_standardize_against_baseline(
        baseline_joint_df["script_mean"].to_numpy(dtype=float),
        seed_joint_df["script_mean"].to_numpy(dtype=float),
    )

    perm_result = compute_grouped_permutation_pvalue(
        baseline_feature_matrix=np.column_stack([z_morph_baseline, z_script_baseline]),
        seed_feature_matrix=np.column_stack([z_morph_seed, z_script_seed]),
        n_permutations=n_permutations,
        random_state=random_state,
    )

    return {
        "seed": int(seed),
        "metric_family": "joint_mean",
        "n_dims": int(len(baseline_joint_df)),
        "joint_mean_rho": rho,
        "joint_mean_perm_p": perm_result.perm_pvalue,
        "joint_mean_n_permutations": perm_result.n_permutations,
    }


def merge_seed_level_results(
    morph_result: Dict[str, float | int | str],
    script_result: Dict[str, float | int | str],
    joint_result: Dict[str, float | int | str],
) -> Dict[str, float | int | str]:
    seed = int(morph_result["seed"])
    if int(script_result["seed"]) != seed or int(joint_result["seed"]) != seed:
        raise ValueError("Seed mismatch while merging seed-level Spearman results.")

    return {
        "seed": seed,
        "n_dims_morph": int(morph_result["n_dims"]),
        "morph_mean_rho": float(morph_result["mean_rho"]),
        "morph_mean_p": float(morph_result["mean_p"]),
        "morph_std_rho": float(morph_result["std_rho"]),
        "morph_std_p": float(morph_result["std_p"]),
        "morph_mean_std_joint_rho": float(morph_result["mean_std_joint_rho"]),
        "morph_mean_std_joint_perm_p": float(morph_result["mean_std_joint_perm_p"]),
        "n_dims_script": int(script_result["n_dims"]),
        "script_mean_rho": float(script_result["mean_rho"]),
        "script_mean_p": float(script_result["mean_p"]),
        "script_std_rho": float(script_result["std_rho"]),
        "script_std_p": float(script_result["std_p"]),
        "script_mean_std_joint_rho": float(script_result["mean_std_joint_rho"]),
        "script_mean_std_joint_perm_p": float(script_result["mean_std_joint_perm_p"]),
        "n_dims_joint_mean": int(joint_result["n_dims"]),
        "joint_mean_rho": float(joint_result["joint_mean_rho"]),
        "joint_mean_perm_p": float(joint_result["joint_mean_perm_p"]),
    }


def run_seed_spearman_analysis(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    seed: int,
    n_permutations: int = 10000,
    random_state: Optional[int] = 0,
    morph_full_filename: str = "kondrak_global_summary.csv",
    morph_random_filename: str = "kondrak_global_summary.csv",
    script_full_filename: str = "script_entropy_summary.csv",
    script_random_filename: Optional[str] = None,
) -> pd.DataFrame:
    baseline_morph = standardize_morphology_summary_df(
        load_full_morphology_summary(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            filename=morph_full_filename,
        ),
        df_name="baseline_morphology_summary",
    )
    seed_morph = standardize_morphology_summary_df(
        load_random_morphology_summary(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            seed=seed,
            filename=morph_random_filename,
        ),
        df_name="seed_morphology_summary",
    )
    baseline_script = standardize_script_summary_df(
        load_full_script_summary(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            filename=script_full_filename,
        ),
        df_name="baseline_script_summary",
    )
    seed_script = standardize_script_summary_df(
        load_random_script_summary(
            out_root=out_root,
            model_name=model_name,
            space_name=space_name,
            seed=seed,
            filename=script_random_filename,
        ),
        df_name="seed_script_summary",
    )

    morph_result = compare_seed_to_baseline_for_metric(
        baseline_morph,
        seed_morph,
        metric_family="morph",
        seed=seed,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    script_result = compare_seed_to_baseline_for_metric(
        baseline_script,
        seed_script,
        metric_family="script",
        seed=seed,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    joint_result = compare_seed_to_baseline_joint_mean(
        baseline_morph,
        seed_morph,
        baseline_script,
        seed_script,
        seed=seed,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    row = merge_seed_level_results(morph_result, script_result, joint_result)
    row["model_name"] = str(model_name)
    row["space_name"] = str(space_name)
    return pd.DataFrame([row])


def run_spearman_over_random_seeds(
    *,
    out_root: str | Path,
    model_name: str,
    space_name: str,
    seed_list: Sequence[int],
    n_permutations: int = 10000,
    random_state: Optional[int] = 0,
    morph_full_filename: str = "kondrak_global_summary.csv",
    morph_random_filename: str = "kondrak_global_summary.csv",
    script_full_filename: str = "script_entropy_summary.csv",
    script_random_filename: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for seed in seed_list:
        rows.append(
            run_seed_spearman_analysis(
                out_root=out_root,
                model_name=model_name,
                space_name=space_name,
                seed=int(seed),
                n_permutations=n_permutations,
                random_state=random_state,
                morph_full_filename=morph_full_filename,
                morph_random_filename=morph_random_filename,
                script_full_filename=script_full_filename,
                script_random_filename=script_random_filename,
            )
        )

    if len(rows) == 0:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def save_spearman_summary_df(df: pd.DataFrame, outpath: str | Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    return outpath



def _pretty_model_name(model_name: str) -> str:
    mapping = {
        "gpt-oss": "gpt-oss",
        "mistralai/Mistral-7B-v0.1": "Mistral-7B",
        "mistralai/Mixtral-8x7B-v0.1": "Mixtral-8x7B",
    }
    return mapping.get(str(model_name), str(model_name))


def _format_mean_std(mean_value: float, std_value: float, *, precision: int = 3) -> str:
    return f"{mean_value:.{precision}f} $\pm$ {std_value:.{precision}f}"


def make_main_spearman_latex_table(
    summary_df: pd.DataFrame,
    *,
    precision: int = 3,
    model_order: Optional[Sequence[str]] = None,
    caption: str = "Main Spearman stability results across random PCA seeds.",
    label: str = "tab:spearman_main",
) -> str:
    required_cols = [
        "model_name",
        "seed",
        "morph_mean_rho",
        "morph_mean_p",
        "script_mean_rho",
        "script_mean_p",
        "joint_mean_rho",
    ]
    _ensure_required_columns(summary_df, required_cols, "summary_df")

    grouped = (
        summary_df.groupby("model_name", sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            morph_mean_rho_mean=("morph_mean_rho", "mean"),
            morph_mean_rho_std=("morph_mean_rho", "std"),
            morph_mean_p_mean=("morph_mean_p", "mean"),
            morph_mean_p_std=("morph_mean_p", "std"),
            script_mean_rho_mean=("script_mean_rho", "mean"),
            script_mean_rho_std=("script_mean_rho", "std"),
            script_mean_p_mean=("script_mean_p", "mean"),
            script_mean_p_std=("script_mean_p", "std"),
            joint_mean_rho_mean=("joint_mean_rho", "mean"),
            joint_mean_rho_std=("joint_mean_rho", "std"),
        )
        .reset_index()
    )

    for col in [
        "morph_mean_rho_std",
        "morph_mean_p_std",
        "script_mean_rho_std",
        "script_mean_p_std",
        "joint_mean_rho_std",
    ]:
        grouped[col] = grouped[col].fillna(0.0)

    if model_order is not None:
        order_map = {name: idx for idx, name in enumerate(model_order)}
        grouped = grouped.sort_values(
            "model_name",
            key=lambda s: s.map(lambda x: order_map.get(x, len(order_map))),
        ).reset_index(drop=True)

    table_df = pd.DataFrame(
        {
            "Model": grouped["model_name"].map(_pretty_model_name),
            "Seeds": grouped["n_seeds"].astype(int),
            "Morph Mean $\rho$": [
                _format_mean_std(m, s, precision=precision)
                for m, s in zip(grouped["morph_mean_rho_mean"], grouped["morph_mean_rho_std"])
            ],
            "Morph Mean $p$": [
                _format_mean_std(m, s, precision=precision)
                for m, s in zip(grouped["morph_mean_p_mean"], grouped["morph_mean_p_std"])
            ],
            "Script Mean $\rho$": [
                _format_mean_std(m, s, precision=precision)
                for m, s in zip(grouped["script_mean_rho_mean"], grouped["script_mean_rho_std"])
            ],
            "Script Mean $p$": [
                _format_mean_std(m, s, precision=precision)
                for m, s in zip(grouped["script_mean_p_mean"], grouped["script_mean_p_std"])
            ],
            "Joint Mean $\rho$": [
                _format_mean_std(m, s, precision=precision)
                for m, s in zip(grouped["joint_mean_rho_mean"], grouped["joint_mean_rho_std"])
            ],
        }
    )

    return table_df.to_latex(index=False, escape=False, caption=caption, label=label)


def make_appendix_spearman_latex_table(
    summary_df: pd.DataFrame,
    *,
    precision: int = 3,
    model_order: Optional[Sequence[str]] = None,
    caption: str = "Full seed-level Spearman results.",
    label: str = "tab:spearman_appendix",
    longtable: bool = True,
) -> str:
    required_cols = [
        "model_name",
        "seed",
        "morph_mean_rho",
        "morph_mean_p",
        "script_mean_rho",
        "script_mean_p",
        "joint_mean_rho",
        "joint_mean_perm_p",
        "morph_std_rho",
        "morph_std_p",
        "script_std_rho",
        "script_std_p",
        "morph_mean_std_joint_rho",
        "morph_mean_std_joint_perm_p",
        "script_mean_std_joint_rho",
        "script_mean_std_joint_perm_p",
    ]
    _ensure_required_columns(summary_df, required_cols, "summary_df")

    table_df = summary_df[required_cols].copy()
    if model_order is not None:
        order_map = {name: idx for idx, name in enumerate(model_order)}
        table_df = table_df.sort_values(
            ["model_name", "seed"],
            key=lambda s: s.map(lambda x: order_map.get(x, len(order_map))) if s.name == "model_name" else s,
        )
    else:
        table_df = table_df.sort_values(["model_name", "seed"]).reset_index(drop=True)

    table_df = table_df.rename(
        columns={
            "model_name": "Model",
            "seed": "Seed",
            "morph_mean_rho": "Morph Mean $\\rho$",
            "morph_mean_p": "Morph Mean $p$",
            "script_mean_rho": "Script Mean $\\rho$",
            "script_mean_p": "Script Mean $p$",
            "joint_mean_rho": "Joint Mean $\\rho$",
            "joint_mean_perm_p": "Joint Mean Perm. $p$",
            "morph_std_rho": "Morph Std $\\rho$",
            "morph_std_p": "Morph Std $p$",
            "script_std_rho": "Script Std $\\rho$",
            "script_std_p": "Script Std $p$",
            "morph_mean_std_joint_rho": "Morph Mean+Std $\\rho$",
            "morph_mean_std_joint_perm_p": "Morph Mean+Std Perm. $p$",
            "script_mean_std_joint_rho": "Script Mean+Std $\\rho$",
            "script_mean_std_joint_perm_p": "Script Mean+Std Perm. $p$",
        }
    )
    table_df["Model"] = table_df["Model"].map(_pretty_model_name)

    numeric_cols = [col for col in table_df.columns if col not in {"Model", "Seed"}]
    for col in numeric_cols:
        table_df[col] = table_df[col].map(lambda x: f"{float(x):.{precision}f}")

    return table_df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        longtable=longtable,
    )



def make_permutation_vs_full_latex_table(
    summary_df: pd.DataFrame,
    *,
    precision: int = 3,
    model_order: Optional[Sequence[str]] = None,
    caption: str = "Spearman comparison between the original full-PCA results and the permutation control.",
    label: str = "tab:perm_vs_full_spearman",
) -> str:
    required_cols = [
        "baseline_model_name",
        "morph_mean_rho",
        "morph_mean_p",
        "script_mean_rho",
        "script_mean_p",
        "joint_mean_rho",
        "joint_mean_perm_p",
    ]
    _ensure_required_columns(summary_df, required_cols, "summary_df")

    table_df = summary_df[required_cols].copy()
    if model_order is not None:
        order_map = {name: idx for idx, name in enumerate(model_order)}
        table_df = table_df.sort_values(
            "baseline_model_name",
            key=lambda s: s.map(lambda x: order_map.get(x, len(order_map))),
        ).reset_index(drop=True)
    else:
        table_df = table_df.sort_values("baseline_model_name").reset_index(drop=True)

    table_df = table_df.rename(
        columns={
            "baseline_model_name": "Model",
            "morph_mean_rho": "Morph Mean $\\rho$",
            "morph_mean_p": "Morph Mean $p$",
            "script_mean_rho": "Script Mean $\\rho$",
            "script_mean_p": "Script Mean $p$",
            "joint_mean_rho": "Joint Mean $\\rho$",
            "joint_mean_perm_p": "Joint Mean Perm. $p$",
        }
    )
    table_df["Model"] = table_df["Model"].map(_pretty_model_name)

    numeric_cols = [col for col in table_df.columns if col != "Model"]
    for col in numeric_cols:
        table_df[col] = table_df[col].map(lambda x: f"{float(x):.{precision}f}")

    return table_df.to_latex(index=False, escape=False, caption=caption, label=label)
