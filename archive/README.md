# Archived Notebooks

These notebooks produced the results published in the MIWAI 2026 paper. They are kept as a record of the original experiment sequence and are not maintained.

Most of them import modules that were replaced by the packaged pipeline in `crossscale/` (`pca_hdbscan`, `kondrak_cluster_morphology`, `script_entropy`, `spearman`, `partition_permutation_control`, `permutation_multiseed_pipeline`), and all of them use paths relative to the repository root, so they do not run from this directory against the current code. To run them as they were, check out the Git tag `pre-legacy-removal`, where they sit at the repository root next to the modules they import.

For current usage, see [`walkthrough.ipynb`](../walkthrough.ipynb).

| Stage | Notebook | Purpose |
|---:|---|---|
| 0 | [`00_init.ipynb`](00_init.ipynb) | Model, tokenizer, and tensor preparation |
| 0 | [`0_tool_fig.ipynb`](0_tool_fig.ipynb), [`0_tool_mess.ipynb`](0_tool_mess.ipynb), [`0_tool_sum.ipynb`](0_tool_sum.ipynb) | Figure, cleanup, and summary utilities |
| 1 | [`1_spectrum_anlys.ipynb`](1_spectrum_anlys.ipynb) | Main spectrum analysis and scale selection |
| 1R | [`1r_low_dimension_spectrum_anlys.ipynb`](1r_low_dimension_spectrum_anlys.ipynb) | End-to-end Low12 supplement |
| 2 | [`2_full_pca.ipynb`](2_full_pca.ipynb) | Deterministic full PCA |
| 3 | [`3_hdbscan_gpu.ipynb`](3_hdbscan_gpu.ipynb) | Baseline GPU HDBSCAN |
| 3R | [`3r_hdbscan_gpu.ipynb`](3r_hdbscan_gpu.ipynb) | HDBSCAN parameter stability |
| 4 | [`4_random PCA_hdbscan.ipynb`](4_random%20PCA_hdbscan.ipynb) | Randomized-PCA stability |
| 4R | [`4r_random_projection_hdbscan.ipynb`](4r_random_projection_hdbscan.ipynb) | Random orthogonal projection |
| 5 / 5R | [`5_morphology.ipynb`](5_morphology.ipynb), [`5r_morphology.ipynb`](5r_morphology.ipynb) | Orthographic analyses |
| 6 / 6R | [`6_multi-ligual.ipynb`](6_multi-ligual.ipynb), [`6r_multi-ligual.ipynb`](6r_multi-ligual.ipynb) | Multi-script analyses |
| 7 / 7R | [`7_spearman.ipynb`](7_spearman.ipynb), [`7r_pearson.ipynb`](7r_pearson.ipynb) | Cross-scale association statistics |
| 8 | [`8_permutation.ipynb`](8_permutation.ipynb) | Token permutation controls |
| 9 | [`9_case study.ipynb`](9_case%20study.ipynb) | Cluster-level case studies |
| 10 | [`10_statistic.ipynb`](10_statistic.ipynb), [`10p_statistic.ipynb`](10p_statistic.ipynb), [`10r_statistic.ipynb`](10r_statistic.ipynb) | Baseline, permutation, and stability summaries |

The `R` suffix is historical and has no single meaning. Use the descriptions rather than inferring an experiment from a filename.

The GPT-oss-20B permutation-control scores computed in these notebooks used undecoded byte-level tokens; see the [erratum](../README.md#erratum).
