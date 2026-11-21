# Paper Results

Compact result tables for the experiments reported in *From Local Density to Lexical Features: A Cross-Scale Study of Unembedding Spaces* (MIWAI 2026). Every table is restricted to the nine paper scales listed in `res/spectrum_analysis/<model>/candidate_scales.csv`. Large artifacts such as PCA projections and token-level partitions are not included.

Model directories use the same names as `res/spectrum_analysis/`: `mistralai_Mistral-7B-v0.1`, `mistralai_Mixtral-8x7B-v0.1`, and `gpt-oss` (GPT-oss-20B).

## Layout

| Directory | Experiment | Paper reference |
|---|---|---|
| `baseline/<model>/` | Full PCA, HDBSCAN `(min_cluster_size, min_samples) = (5, 5)` | Fig. 2, Fig. 3, Table 2 (Full) |
| `cross_seed/<model>/` | Randomized PCA, eight seeds | Table 2 (Random), Table 4 |
| `hdbscan_params/mcs=<a>_ms=<b>/<model>/` | Six perturbed HDBSCAN settings | Table 6 |
| `token_permutation/<model>/` | Token permutation within non-noise clusters, ten seeds | Table 2 (Permutation), Table 5 |
| `length_bucket_permutation/<model>/` | Permutation within character-length buckets, ten seeds (O-S only) | Table 7 |
| `corrections/gpt-oss/` | GPT-oss permutation controls rescored with decoded token strings | [Erratum](../../README.md#erratum) |

The GPT-oss tables in `token_permutation/` and `length_bucket_permutation/` are the published values, kept for reference. Use `corrections/gpt-oss/` for corrected GPT-oss permutation results.

## Files

| File | Content |
|---|---|
| `summary.csv`, `summary_all_seeds.csv` | Per-scale partition statistics: cluster count, noise count, noise ratio |
| `kondrak_global_summary*.csv` | Per-scale Orthographic Similarity (O-S): `global_mean`, `global_std`, `n_clusters` |
| `script_entropy_summary*.csv`, `script_entropy_all_seeds.csv` | Per-scale Multi-Script Entropy (MS-E): raw (`mean_H`) and normalized (`mean_H_norm`) |
| `baseline/<model>/morph/kondrak_clusters_pca_<k>.csv` | Cluster-level O-S at scale `k` |
| `baseline/<model>/script_entropy/pca_<k>/cluster_script_entropy.csv` | Cluster-level MS-E at scale `k` |
| `length_bucket_permutation/length_bucket_token_counts.csv` | Token counts per character-length bucket |
| `hdbscan_params/stability_vs_full_spearman_all_configs.csv` | Per-setting Spearman statistics, including the joint Spearman of Appendix A.4 |

The `*_path` columns in summary tables record where each artifact was written during the original run. Those artifacts are not part of this directory.

## Notes on reported values

`tests/test_paper_results.py` recomputes Tables 1, 2, and 4–7 and the Appendix A.4 joint Spearman from these files.

- Table 2 reports MS-E as the normalized entropy `mean_H_norm` = H(c) / log|S|. Tables 4–7 and Appendix A.4 compare the raw entropy `mean_H`.
- Table 2 standard deviations for Full and Permutation are population standard deviations (`ddof=0`) across the nine scale-level values. The Random column averages each seed's cross-scale mean, then reports the mean and sample standard deviation across seeds.
- Tables 4–7 report the mean and sample standard deviation across seeds or parameter settings of Spearman's ρ, its p-value, MAE, and R² against the baseline curve.
- The Appendix A.4 joint Spearman concatenates the O-S and MS-E curves after z-scoring both the baseline and perturbed curves with the baseline mean and standard deviation.

Three printed values differ from these files beyond rounding:

| Location | Printed | Reproduced | Note |
|---|---|---|---|
| Table 2, Mixtral-8×7B O-S, Permutation sd | 0.015 | 0.014 | The printed value uses `ddof=1` (0.0153); the other cells use `ddof=0` (0.0144) |
| Table 6, GPT-oss-20B O-S, MAE mean | 0.017 | 0.016 | Archived mean is 0.01648 |
| Appendix A.4, GPT-oss-20B joint Spearman | 0.8000 | 0.7974 | Archived mean is 0.79742 |
