"""Regression tests that reproduce the published paper numbers.

The expected values are transcribed from "From Local Density to Lexical
Features: A Cross-Scale Study of Unembedding Spaces" (MIWAI 2026, LNCS 17147,
pp. 160-175). Every value is recomputed from the compact tables under
res/paper_results/ and compared at the precision printed in the paper.
"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from crossscale.kondrak_cluster_morphology import compute_global_cluster_stats
from crossscale.script_entropy import (
    entropy_from_distribution,
    normalized_entropy,
    partition_entropy_stats,
)
from crossscale.statistics_pipeline import (
    MAEStatistic,
    R2Statistic,
    SpearmanStatistic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "res" / "paper_results"
SPECTRUM_ROOT = REPO_ROOT / "res" / "spectrum_analysis"

MISTRAL = "mistralai_Mistral-7B-v0.1"
MIXTRAL = "mistralai_Mixtral-8x7B-v0.1"
GPT_OSS = "gpt-oss"
MODELS = (MISTRAL, MIXTRAL, GPT_OSS)

OS = "O-S"
MSE = "MS-E"

# Table 1: model-specific candidate scales.
TABLE_1 = {
    MISTRAL: {
        "elbow": [5, 142, 3938, 4088],
        "energy": [997, 2084, 3031, 3459],
        "full_dimension": [4096],
    },
    MIXTRAL: {
        "elbow": [8, 158, 3957, 4093],
        "energy": [1111, 2156, 3052, 3457],
        "full_dimension": [4096],
    },
    GPT_OSS: {
        "elbow": [6, 182, 466, 2868],
        "energy": [739, 1591, 2264, 2532],
        "full_dimension": [2880],
    },
}

# Table 2: (Full, Random, Permutation) as (mean, sd) strings, plus Gap.
TABLE_2 = {
    (MISTRAL, OS): (("0.500", "0.117"), ("0.502", "0.002"), ("0.142", "0.018"), "0.360"),
    (MISTRAL, MSE): (("0.044", "0.018"), ("0.044", "0.0005"), ("0.102", "0.049"), "-0.058"),
    (MIXTRAL, OS): (("0.484", "0.114"), ("0.486", "0.002"), ("0.146", "0.015"), "0.340"),
    (MIXTRAL, MSE): (("0.046", "0.010"), ("0.046", "0.0005"), ("0.095", "0.031"), "-0.049"),
    (GPT_OSS, OS): (("0.497", "0.099"), ("0.499", "0.002"), ("0.101", "0.016"), "0.398"),
    (GPT_OSS, MSE): (("0.070", "0.027"), ("0.069", "0.001"), ("0.280", "0.126"), "-0.210"),
}

# Tables 4-7: (rho, p_value, mae, r2), each as (mean, sd) strings.
TABLE_4_CROSS_SEED = {
    (MISTRAL, OS): (("0.965", "0.033"), ("0.000", "0.000"), ("0.004", "0.001"), ("0.996", "0.002")),
    (MISTRAL, MSE): (("0.960", "0.031"), ("0.000", "0.000"), ("0.004", "0.001"), ("0.985", "0.010")),
    (MIXTRAL, OS): (("0.956", "0.053"), ("0.001", "0.001"), ("0.003", "0.001"), ("0.998", "0.002")),
    (MIXTRAL, MSE): (("0.929", "0.053"), ("0.001", "0.001"), ("0.003", "0.001"), ("0.981", "0.020")),
    (GPT_OSS, OS): (("0.996", "0.008"), ("0.000", "0.000"), ("0.003", "0.001"), ("0.997", "0.003")),
    (GPT_OSS, MSE): (("0.965", "0.055"), ("0.001", "0.002"), ("0.004", "0.001"), ("0.993", "0.006")),
}

TABLE_5_TOKEN_PERMUTATION = {
    (MISTRAL, OS): (("0.328", "0.008"), ("0.388", "0.012"), ("0.359", "0.001"), ("-9.136", "0.042")),
    (MISTRAL, MSE): (("-0.183", "0.065"), ("0.639", "0.126"), ("0.198", "0.004"), ("-15.065", "1.018")),
    (MIXTRAL, OS): (("-0.073", "0.272"), ("0.520", "0.152"), ("0.338", "0.001"), ("-8.809", "0.025")),
    (MIXTRAL, MSE): (("-0.410", "0.099"), ("0.287", "0.124"), ("0.168", "0.003"), ("-36.972", "1.440")),
    (GPT_OSS, OS): (("-0.417", "0.053"), ("0.269", "0.068"), ("0.396", "0.000"), ("-16.151", "0.006")),
    (GPT_OSS, MSE): (("-0.865", "0.062"), ("0.005", "0.006"), ("0.158", "0.001"), ("-2.944", "0.060")),
}

TABLE_6_HDBSCAN_PARAMETERS = {
    (MISTRAL, OS): (("0.942", "0.049"), ("0.001", "0.001"), ("0.013", "0.005"), ("0.984", "0.011")),
    (MISTRAL, MSE): (("0.797", "0.211"), ("0.048", "0.075"), ("0.011", "0.004"), ("0.915", "0.102")),
    (MIXTRAL, OS): (("0.686", "0.158"), ("0.070", "0.092"), ("0.013", "0.005"), ("0.981", "0.013")),
    (MIXTRAL, MSE): (("0.889", "0.141"), ("0.011", "0.023"), ("0.010", "0.004"), ("0.878", "0.065")),
    (GPT_OSS, OS): (("0.981", "0.025"), ("0.000", "0.000"), ("0.017", "0.006"), ("0.966", "0.026")),
    (GPT_OSS, MSE): (("0.919", "0.081"), ("0.002", "0.004"), ("0.022", "0.008"), ("0.923", "0.052")),
}

TABLE_7_LENGTH_BUCKET = {
    (MISTRAL, OS): (("0.545", "0.118"), ("0.149", "0.095"), ("0.351", "0.001"), ("-8.770", "0.030")),
    (MIXTRAL, OS): (("0.675", "0.111"), ("0.061", "0.054"), ("0.335", "0.000"), ("-8.416", "0.013")),
    (GPT_OSS, OS): (("-0.373", "0.052"), ("0.326", "0.073"), ("0.390", "0.000"), ("-15.618", "0.018")),
}

# Appendix A.4: mean joint Spearman over the six HDBSCAN settings.
JOINT_SPEARMAN = {MISTRAL: "0.9542", MIXTRAL: "0.5710", GPT_OSS: "0.8000"}

# Printed values that the archived results do not reproduce at the printed
# precision. Each entry maps a cell to (printed value, reproduced value).
KNOWN_DEVIATIONS = {
    # The printed sd uses ddof=1; every other Table 2 sd uses ddof=0 (0.0144).
    ("table2", MIXTRAL, OS, "permutation_sd"): ("0.015", "0.014"),
    # The archived mean MAE is 0.01648.
    ("table6", GPT_OSS, OS, "mae_mean"): ("0.017", "0.016"),
    # The archived mean joint rho is 0.79742.
    ("a4", GPT_OSS, "joint", "rho"): ("0.8000", "0.7974"),
}

STATISTIC_NAMES = ("rho", "p_value", "mae", "r2")

# Erratum: GPT-oss permutation controls recomputed with decoded token strings.
CORRECTIONS_ROOT = RESULTS_ROOT / "corrections" / GPT_OSS
CORRECTED_TABLE_2 = {
    OS: (("0.103", "0.018"), "0.396"),
    MSE: (("0.232", "0.091"), "-0.162"),
}
CORRECTED_TABLE_5 = {
    OS: (("-0.433", "0.061"), ("0.249", "0.077"), ("0.394", "0.000"), ("-15.971", "0.013")),
    MSE: (("-0.552", "0.033"), ("0.125", "0.028"), ("0.546", "0.001"), ("-53.287", "0.246")),
}
CORRECTED_TABLE_7 = (("-0.415", "0.060"), ("0.272", "0.078"), ("0.385", "0.000"), ("-15.216", "0.015"))


def read_table(*parts: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_ROOT.joinpath(*parts))


def curve(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.sort_values("pca_dim").set_index("pca_dim")[column].astype(float)


def os_curve(frame: pd.DataFrame) -> pd.Series:
    return curve(frame, "global_mean")


def compare_curves(reference: pd.Series, target: pd.Series) -> dict[str, float]:
    if list(reference.index) != list(target.index):
        raise AssertionError(
            f"curves are not aligned: {list(reference.index)} vs {list(target.index)}"
        )
    values = {}
    for statistic in (SpearmanStatistic(), MAEStatistic(), R2Statistic()):
        result = statistic.compute(reference.to_numpy(), target.to_numpy())
        for observation in result.observations:
            values[observation.component] = observation.value
    return values


def summarize_comparisons(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    frame = pd.DataFrame(rows)
    return {
        name: (float(frame[name].mean()), float(frame[name].std(ddof=1)))
        for name in STATISTIC_NAMES
    }


def compare_seeded(
    reference: pd.Series,
    frame: pd.DataFrame,
    column: str,
    seed_column: str,
) -> dict[str, tuple[float, float]]:
    rows = [
        compare_curves(reference, curve(group, column))
        for _, group in frame.groupby(seed_column)
    ]
    return summarize_comparisons(rows)


def hdbscan_config_dirs() -> list[Path]:
    return sorted(
        path for path in (RESULTS_ROOT / "hdbscan_params").iterdir() if path.is_dir()
    )


class PaperValueAssertions(unittest.TestCase):
    def assert_printed(self, actual: float, printed: str, key: tuple) -> None:
        if key in KNOWN_DEVIATIONS:
            printed = KNOWN_DEVIATIONS[key][1]
        decimals = len(printed.split(".")[1])
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        self.assertLessEqual(
            abs(actual - float(printed)),
            tolerance,
            msg=f"{key}: reproduced {actual:.6f}, printed {printed}",
        )

    def assert_statistic_table(
        self,
        table_name: str,
        key: tuple[str, str],
        summary: dict[str, tuple[float, float]],
        printed: tuple[tuple[str, str], ...],
    ) -> None:
        for name, (printed_mean, printed_sd) in zip(STATISTIC_NAMES, printed):
            mean, sd = summary[name]
            self.assert_printed(mean, printed_mean, (table_name, *key, f"{name}_mean"))
            self.assert_printed(sd, printed_sd, (table_name, *key, f"{name}_sd"))


class Table1CandidateScalesTest(unittest.TestCase):
    def test_candidate_scales_match_table_1(self) -> None:
        for model, expected in TABLE_1.items():
            with self.subTest(model=model):
                frame = pd.read_csv(SPECTRUM_ROOT / model / "candidate_scales.csv")
                observed = {
                    source: sorted(group["scale"].astype(int))
                    for source, group in frame.groupby("sources")
                }
                self.assertEqual(observed, expected)

    def test_result_tables_use_paper_scales(self) -> None:
        for model, expected in TABLE_1.items():
            scales = sorted(scale for values in expected.values() for scale in values)
            with self.subTest(model=model):
                frame = read_table("baseline", model, "kondrak_global_summary.csv")
                self.assertEqual(sorted(frame["pca_dim"].astype(int)), scales)


class Table2AggregatedStatisticsTest(PaperValueAssertions):
    def test_full_random_permutation_and_gap(self) -> None:
        sources = {
            OS: {
                "baseline": ("kondrak_global_summary.csv", "global_mean"),
                "cross_seed": ("kondrak_global_summary_all_seeds.csv", "global_mean"),
                "permutation": ("kondrak_global_summary_all_seeds.csv", "global_mean"),
            },
            MSE: {
                "baseline": ("script_entropy_summary.csv", "mean_H_norm"),
                "cross_seed": ("script_entropy_all_seeds.csv", "mean_H_norm"),
                "permutation": ("script_entropy_summary_all_seeds.csv", "mean_H_norm"),
            },
        }
        for (model, metric), printed in TABLE_2.items():
            (full, random, permutation, gap) = printed
            files = sources[metric]
            with self.subTest(model=model, metric=metric):
                baseline = curve(read_table("baseline", model, files["baseline"][0]), files["baseline"][1])
                self.assert_printed(baseline.mean(), full[0], ("table2", model, metric, "full_mean"))
                self.assert_printed(baseline.std(ddof=0), full[1], ("table2", model, metric, "full_sd"))

                seeds = read_table("cross_seed", model, files["cross_seed"][0])
                seed_means = seeds.groupby("pca_seed")[files["cross_seed"][1]].mean()
                self.assertEqual(len(seed_means), 8)
                self.assert_printed(seed_means.mean(), random[0], ("table2", model, metric, "random_mean"))
                self.assert_printed(seed_means.std(ddof=1), random[1], ("table2", model, metric, "random_sd"))

                permuted = read_table("token_permutation", model, files["permutation"][0])
                permuted_curve = permuted.groupby("pca_dim")[files["permutation"][1]].mean()
                self.assert_printed(permuted_curve.mean(), permutation[0], ("table2", model, metric, "permutation_mean"))
                self.assert_printed(permuted_curve.std(ddof=0), permutation[1], ("table2", model, metric, "permutation_sd"))

                printed_gap = max(float(full[0]), float(random[0])) - float(permutation[0])
                self.assert_printed(printed_gap, gap, ("table2", model, metric, "gap"))


class CurveComparisonTablesTest(PaperValueAssertions):
    def test_table_4_cross_seed(self) -> None:
        for (model, metric), printed in TABLE_4_CROSS_SEED.items():
            with self.subTest(model=model, metric=metric):
                if metric == OS:
                    reference = os_curve(read_table("baseline", model, "kondrak_global_summary.csv"))
                    frame = read_table("cross_seed", model, "kondrak_global_summary_all_seeds.csv")
                    column = "global_mean"
                else:
                    reference = curve(read_table("baseline", model, "script_entropy_summary.csv"), "mean_H")
                    frame = read_table("cross_seed", model, "script_entropy_all_seeds.csv")
                    column = "mean_H"
                self.assertEqual(frame["pca_seed"].nunique(), 8)
                summary = compare_seeded(reference, frame, column, "pca_seed")
                self.assert_statistic_table("table4", (model, metric), summary, printed)

    def test_table_5_token_permutation(self) -> None:
        for (model, metric), printed in TABLE_5_TOKEN_PERMUTATION.items():
            with self.subTest(model=model, metric=metric):
                if metric == OS:
                    reference = os_curve(read_table("baseline", model, "kondrak_global_summary.csv"))
                    frame = read_table("token_permutation", model, "kondrak_global_summary_all_seeds.csv")
                    column = "global_mean"
                else:
                    reference = curve(read_table("baseline", model, "script_entropy_summary.csv"), "mean_H")
                    frame = read_table("token_permutation", model, "script_entropy_summary_all_seeds.csv")
                    column = "mean_H"
                self.assertEqual(frame["perm_seed"].nunique(), 10)
                summary = compare_seeded(reference, frame, column, "perm_seed")
                self.assert_statistic_table("table5", (model, metric), summary, printed)

    def test_table_6_hdbscan_parameters(self) -> None:
        configs = hdbscan_config_dirs()
        self.assertEqual(
            [path.name for path in configs],
            ["mcs=4_ms=4", "mcs=4_ms=5", "mcs=5_ms=4", "mcs=5_ms=6", "mcs=6_ms=5", "mcs=6_ms=6"],
        )
        for (model, metric), printed in TABLE_6_HDBSCAN_PARAMETERS.items():
            with self.subTest(model=model, metric=metric):
                if metric == OS:
                    filename, column = "kondrak_global_summary.csv", "global_mean"
                else:
                    filename, column = "script_entropy_summary.csv", "mean_H"
                reference = curve(read_table("baseline", model, filename), column)
                rows = [
                    compare_curves(reference, curve(pd.read_csv(config / model / filename), column))
                    for config in configs
                ]
                summary = summarize_comparisons(rows)
                self.assert_statistic_table("table6", (model, metric), summary, printed)

    def test_table_7_length_bucket_permutation(self) -> None:
        for (model, metric), printed in TABLE_7_LENGTH_BUCKET.items():
            with self.subTest(model=model):
                reference = os_curve(read_table("baseline", model, "kondrak_global_summary.csv"))
                frame = read_table("length_bucket_permutation", model, "kondrak_global_summary_all_seeds.csv")
                self.assertEqual(frame["perm_seed"].nunique(), 10)
                summary = compare_seeded(reference, frame, "global_mean", "perm_seed")
                self.assert_statistic_table("table7", (model, metric), summary, printed)

    def test_appendix_a4_joint_spearman(self) -> None:
        spearman = SpearmanStatistic()
        for model, printed in JOINT_SPEARMAN.items():
            with self.subTest(model=model):
                reference_os = os_curve(read_table("baseline", model, "kondrak_global_summary.csv"))
                reference_mse = curve(read_table("baseline", model, "script_entropy_summary.csv"), "mean_H")
                rhos = []
                for config in hdbscan_config_dirs():
                    target_os = os_curve(pd.read_csv(config / model / "kondrak_global_summary.csv"))
                    target_mse = curve(pd.read_csv(config / model / "script_entropy_summary.csv"), "mean_H")
                    # Both curves are z-scored with the baseline mean and sd.
                    reference = np.concatenate([
                        (reference_os - reference_os.mean()) / reference_os.std(ddof=0),
                        (reference_mse - reference_mse.mean()) / reference_mse.std(ddof=0),
                    ])
                    target = np.concatenate([
                        (target_os - reference_os.mean()) / reference_os.std(ddof=0),
                        (target_mse - reference_mse.mean()) / reference_mse.std(ddof=0),
                    ])
                    result = spearman.compute(reference, target)
                    rhos.append(result.observations[0].value)
                self.assert_printed(float(np.mean(rhos)), printed, ("a4", model, "joint", "rho"))

    def test_appendix_a4_matches_archived_stability_table(self) -> None:
        archived = read_table("hdbscan_params", "stability_vs_full_spearman_all_configs.csv")
        means = archived.groupby("model_name")["joint_mean_rho"].mean()
        self.assertAlmostEqual(means["mistralai/Mistral-7B-v0.1"], 0.9542, places=4)
        self.assertAlmostEqual(means["mistralai/Mixtral-8x7B-v0.1"], 0.5710, places=4)
        self.assertAlmostEqual(means["gpt-oss"], 0.7974, places=4)


class ErratumCorrectedValuesTest(PaperValueAssertions):
    """Check the corrected GPT-oss permutation values listed in the README erratum."""

    def corrected(self, name: str) -> pd.DataFrame:
        frame = pd.read_csv(CORRECTIONS_ROOT / name)
        self.assertEqual(frame["perm_seed"].nunique(), 10)
        self.assertEqual(sorted(frame["pca_dim"].unique()), sorted(
            scale for values in TABLE_1[GPT_OSS].values() for scale in values
        ))
        return frame

    def test_corrected_table_2(self) -> None:
        sources = {
            OS: ("token_permutation_kondrak_global_summary_all_seeds.csv", "global_mean"),
            MSE: ("token_permutation_script_entropy_summary_all_seeds.csv", "mean_H_norm"),
        }
        for metric, ((mean, sd), gap) in CORRECTED_TABLE_2.items():
            with self.subTest(metric=metric):
                filename, column = sources[metric]
                permuted_curve = self.corrected(filename).groupby("pca_dim")[column].mean()
                self.assert_printed(permuted_curve.mean(), mean, ("erratum2", metric, "mean"))
                self.assert_printed(permuted_curve.std(ddof=0), sd, ("erratum2", metric, "sd"))
                full, random = TABLE_2[(GPT_OSS, metric)][:2]
                corrected_gap = max(float(full[0]), float(random[0])) - permuted_curve.mean()
                self.assert_printed(corrected_gap, gap, ("erratum2", metric, "gap"))

    def test_corrected_table_5(self) -> None:
        for metric, printed in CORRECTED_TABLE_5.items():
            with self.subTest(metric=metric):
                if metric == OS:
                    reference = os_curve(read_table("baseline", GPT_OSS, "kondrak_global_summary.csv"))
                    frame = self.corrected("token_permutation_kondrak_global_summary_all_seeds.csv")
                    column = "global_mean"
                else:
                    reference = curve(read_table("baseline", GPT_OSS, "script_entropy_summary.csv"), "mean_H")
                    frame = self.corrected("token_permutation_script_entropy_summary_all_seeds.csv")
                    column = "mean_H"
                summary = compare_seeded(reference, frame, column, "perm_seed")
                self.assert_statistic_table("erratum5", (GPT_OSS, metric), summary, printed)

    def test_corrected_table_7(self) -> None:
        reference = os_curve(read_table("baseline", GPT_OSS, "kondrak_global_summary.csv"))
        frame = self.corrected("length_bucket_kondrak_global_summary_all_seeds.csv")
        summary = compare_seeded(reference, frame, "global_mean", "perm_seed")
        self.assert_statistic_table("erratum7", (GPT_OSS, OS), summary, CORRECTED_TABLE_7)

    def test_corrected_runs_keep_published_partitions(self) -> None:
        for corrected_name, published_dir, published_name in (
            ("token_permutation_kondrak_global_summary_all_seeds.csv", "token_permutation", "kondrak_global_summary_all_seeds.csv"),
            ("length_bucket_kondrak_global_summary_all_seeds.csv", "length_bucket_permutation", "kondrak_global_summary_all_seeds.csv"),
        ):
            with self.subTest(corrected=corrected_name):
                corrected = self.corrected(corrected_name).set_index(["perm_seed", "pca_dim"]).sort_index()
                published = read_table(published_dir, GPT_OSS, published_name).set_index(["perm_seed", "pca_dim"]).sort_index()
                self.assertEqual(list(corrected.index), list(published.index))
                self.assertEqual(corrected["n_clusters"].tolist(), published["n_clusters"].tolist())


class ClusterLevelAggregationTest(unittest.TestCase):
    """Rebuild scale-level statistics from the cluster-level baseline tables."""

    def test_orthographic_similarity_aggregation(self) -> None:
        for model in MODELS:
            summary = read_table("baseline", model, "kondrak_global_summary.csv")
            for row in summary.itertuples():
                with self.subTest(model=model, pca_dim=row.pca_dim):
                    clusters = read_table("baseline", model, "morph", f"kondrak_clusters_pca_{row.pca_dim}.csv")
                    stats = compute_global_cluster_stats(clusters, score_col="M(c)")
                    self.assertEqual(stats["n_clusters"], row.n_clusters)
                    self.assertAlmostEqual(stats["global_mean"], row.global_mean, places=12)
                    self.assertAlmostEqual(stats["global_std"], row.global_std, places=12)

    def test_script_entropy_recomputed_from_counts(self) -> None:
        for model in MODELS:
            summary = read_table("baseline", model, "script_entropy_summary.csv")
            for row in summary.itertuples():
                with self.subTest(model=model, pca_dim=row.pca_dim):
                    clusters = read_table(
                        "baseline", model, "script_entropy", f"pca_{row.pca_dim}", "cluster_script_entropy.csv"
                    )
                    scripts = [c for c in clusters.columns if c not in {"cluster_id", "H", "H_norm"}]
                    recomputed_h = []
                    recomputed_h_norm = []
                    for counts in clusters[scripts].itertuples(index=False):
                        total = sum(counts)
                        distribution = {s: c / total for s, c in zip(scripts, counts)}
                        h = entropy_from_distribution(distribution)
                        recomputed_h.append(h)
                        recomputed_h_norm.append(
                            normalized_entropy(h, global_script_set_size=len(scripts))
                        )
                    np.testing.assert_allclose(recomputed_h, clusters["H"], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(recomputed_h_norm, clusters["H_norm"], rtol=0, atol=1e-12)

                    stats = partition_entropy_stats(clusters)
                    self.assertEqual(stats["num_clusters"], row.num_clusters)
                    self.assertAlmostEqual(stats["mean_H"], row.mean_H, places=12)
                    self.assertAlmostEqual(stats["std_H"], row.std_H, places=12)
                    self.assertAlmostEqual(stats["mean_H_norm"], row.mean_H_norm, places=12)
                    self.assertAlmostEqual(stats["std_H_norm"], row.std_H_norm, places=12)


if __name__ == "__main__":
    unittest.main()
