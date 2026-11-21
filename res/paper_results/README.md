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

## Files

| File | Content |
|---|---|
| `summary.csv`, `summary_all_seeds.csv` | Per-scale partition statistics: cluster count, noise count, noise ratio |
| `kondrak_global_summary*.csv` | Per-scale Orthographic Similarity (O-S): `global_mean`, `global_std`, `n_clusters` |
| `script_entropy_summary*.csv`, `script_entropy_all_seeds.csv` | Per-scale Multi-Script Entropy (MS-E): raw (`mean_H`) and normalized (`mean_H_norm`) |
| `baseline/<model>/morph/kondrak_clusters_pca_<k>.csv` | Cluster-level O-S at scale `k` |
| `baseline/<model>/script_entropy/pca_<k>/cluster_script_entropy.csv` | Cluster-level MS-E at scale `k` |
| `length_bucket_permutation/length_bucket_token_counts.csv` | Token counts per character-length bucket |

The `*_path` columns in summary tables record where each artifact was written during the original run. Those artifacts are not part of this directory.

## Notes on reported values

- MS-E values in the paper's tables and figures are the normalized entropy `mean_H_norm` = H(c) / log|S|.
- The scale-level standard deviations reported in Table 2 (Full) are population standard deviations (`ddof=0`) across the nine scale-level means.
- The Random column in Table 2 averages each seed's cross-scale mean, then reports the mean and sample standard deviation across seeds.
