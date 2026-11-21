import os

import matplotlib.pyplot as plt
import pandas as pd


def load_morphology_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
        "model_name",
        "pca_dim",
        "global_mean",
        "global_std",
        "n_clusters",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[morphology] Missing columns: {missing}")

    df = df.copy()
    df["pca_dim"] = df["pca_dim"].astype(int)
    df["global_mean"] = df["global_mean"].astype(float)
    df["global_std"] = df["global_std"].astype(float)
    df["n_clusters"] = df["n_clusters"].astype(int)

    df = df.sort_values(["model_name", "pca_dim"]).reset_index(drop=True)
    return df


def load_script_entropy_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
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
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[script entropy] Missing columns: {missing}")

    df = df.copy()
    df["pca_dim"] = df["pca_dim"].astype(int)
    df["mean_H"] = df["mean_H"].astype(float)
    df["std_H"] = df["std_H"].astype(float)
    df["num_clusters"] = df["num_clusters"].astype(int)

    df = df.sort_values(["model_name", "pca_dim"]).reset_index(drop=True)
    return df


def sanity_check_script_summary_uniqueness(df: pd.DataFrame) -> None:
    dup = df.duplicated(subset=["pca_dim"], keep=False)
    if dup.any():
        bad = df.loc[dup].sort_values("pca_dim")
        raise ValueError(
            "script entropy summary has non-unique pca_dim rows within this model.\n"
            "You likely mixed multiple parameter settings.\n"
            f"{bad}"
        )


def merge_partition_summaries(
    morphology_df: pd.DataFrame,
    script_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = morphology_df.merge(
        script_df[["pca_dim", "mean_H", "std_H"]],
        on="pca_dim",
        how="inner",
    ).sort_values("pca_dim").reset_index(drop=True)

    required_cols = [
        "pca_dim",
        "global_mean",
        "global_std",
        "mean_H",
        "std_H",
    ]
    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"[merged] Missing columns after merge: {missing}")

    return merged


def sanity_check_merged_partition_df(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Merged dataframe is empty.")

    assert df["pca_dim"].is_monotonic_increasing, "pca_dim is not sorted ascending"

    assert (df["global_std"] >= 0).all(), "global_std must be >= 0"
    assert (df["std_H"] >= 0).all(), "std_H must be >= 0"

    if df["pca_dim"].nunique() != len(df):
        raise ValueError("Merged dataframe has duplicate pca_dim rows.")


def plot_partition_level_morphology_vs_script_entropy(
    df: pd.DataFrame,
    model_name: str,
    figsize=(8, 5),
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=11,
    legend_fontsize=11,
    legend_loc="lower right",
    save_path=None,
    show=True,
):
    x = df["pca_dim"].to_numpy()

    morph_mean = df["global_mean"].to_numpy()
    morph_std = df["global_std"].to_numpy()

    script_mean = df["mean_H"].to_numpy()
    script_std = df["std_H"].to_numpy()

    plt.figure(figsize=figsize)

    plt.plot(
        x,
        morph_mean,
        marker="o",
        linewidth=2,
        label="Orthographic Similarity",
    )
    plt.fill_between(
        x,
        morph_mean - morph_std,
        morph_mean + morph_std,
        alpha=0.1,
    )

    plt.plot(
        x,
        script_mean,
        marker="s",
        linewidth=2,
        label="Multi-script entropy",
    )
    plt.fill_between(
        x,
        script_mean - script_std,
        script_mean + script_std,
        alpha=0.1,
    )

    for xi in x:
        plt.axvline(
            x=xi,
            linestyle="--",
            linewidth=0.6,
            alpha=0.4,
            color="black",
        )

    plt.xlabel("PCA Dimension", fontsize=label_fontsize)
    plt.ylabel("Score", fontsize=label_fontsize)
    plt.title(model_name, fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.tight_layout()

    if save_path is None:
        save_dir = "./res/linguistic"
        os.makedirs(save_dir, exist_ok=True)
        safe_model_name = model_name.replace("/", "_")
        save_path = f"{save_dir}/{safe_model_name}.pdf"
    else:
        save_path = os.fspath(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path)
    print(f"Saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
    return save_path


def build_seed_morphology_csv_path(
    out_root: str,
    model_name: str,
    space_name: str,
    seed: int,
) -> str:
    return (
        f"{out_root}/{model_name}/{space_name}/"
        f"random/seed_{seed}/kondrak_global_summary.csv"
    )


def build_seed_script_entropy_summary_path(
    out_root: str,
    model_name: str,
    space_name: str,
    seed: int,
) -> str:
    return (
        f"{out_root}/{model_name}/{space_name}/"
        f"random/seed_{seed}/summary_random_pca_seed_{seed}_script_entropy.csv"
    )


def build_seed_plot_save_path(
    model_name: str,
    seed: int,
    save_dir: str = "res/linguistic",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    return f"{save_dir}/seed_{seed}_{safe_model_name}.pdf"


def plot_partition_level_morphology_vs_script_entropy_seed(
    df: pd.DataFrame,
    model_name: str,
    seed: int,
    figsize=(8, 5),
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=11,
    legend_fontsize=11,
    legend_loc="upper right",
    save_path=None,
    show=True,
):
    x = df["pca_dim"].to_numpy()

    morph_mean = df["global_mean"].to_numpy()
    morph_std = df["global_std"].to_numpy()

    script_mean = df["mean_H"].to_numpy()
    script_std = df["std_H"].to_numpy()

    plt.figure(figsize=figsize)

    plt.plot(
        x,
        morph_mean,
        marker="o",
        linewidth=2,
        label="Orthographic Similarity",
    )
    plt.fill_between(
        x,
        morph_mean - morph_std,
        morph_mean + morph_std,
        alpha=0.18,
    )

    plt.plot(
        x,
        script_mean,
        marker="s",
        linewidth=2,
        label="Multi-script entropy (unnormalized)",
    )
    plt.fill_between(
        x,
        script_mean - script_std,
        script_mean + script_std,
        alpha=0.1,
    )

    for xi in x:
        plt.axvline(
            x=xi,
            linestyle="--",
            linewidth=0.6,
            alpha=0.4,
            color="black",
        )

    plt.xlabel("PCA Dimension", fontsize=label_fontsize)
    plt.ylabel("Score", fontsize=label_fontsize)
    plt.title(f"{model_name} | seed={seed}", fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.tight_layout()

    if save_path is None:
        save_path = build_seed_plot_save_path(
            model_name=model_name,
            seed=seed,
        )
    else:
        save_path = os.fspath(save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
    return save_path
