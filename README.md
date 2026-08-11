# CrossScaleGeometry

This repository accompanies the paper **From Local Density to Lexical Features: A Cross-Scale Study of Unembedding Spaces**. It provides the analysis code, experiment notebooks, validation tests, compact result artifacts, and repository-only extensions behind the study.

> The pipeline can also be adapted to other matrices whose rows are independently meaningful vectors with associated token or string attributes, enabling cross-scale analysis of the relationship between local geometry and lexical features.
>
> The pipeline is designed for extensibility, with modular interfaces intended to support additional geometric, lexical, and linguistic analysis methods in future work.

The project asks how lexical properties of local token neighborhoods change across PCA subspaces of an LLM unembedding matrix. It studies three untied models:

- Mistral-7B-v0.1
- Mixtral-8x7B-v0.1
- GPT-oss-20B

The two primary lexical measurements are:

- **Orthographic Similarity (O-S):** mean pairwise similarity of token strings within a cluster.
- **Multi-Script Entropy (MS-E):** diversity of Unicode writing systems within a cluster.

The repository is a supplement to the paper rather than a model distribution. Model weights, exported tensors, cluster partitions, and other large generated artifacts are intentionally excluded from version control.

[Paper PDF](docs/paper/from-local-density-to-lexical-features.pdf) · [Compact results](res/) · [License](LICENSE)

## Overview

<p align="center">
  <img src="docs/assets/pipeline.png" alt="Analysis pipeline" width="100%">
</p>

The main analysis follows six stages:

1. Export the unembedding matrix and tokenizer for each model.
2. Analyze the singular-value spectrum and select model-specific scales.
3. Extract nested full-PCA subspaces and identify local density clusters with HDBSCAN.
4. Measure cluster-level O-S and MS-E.
5. Aggregate the lexical measurements into cross-scale curves.
6. Evaluate stability with randomized PCA, HDBSCAN parameter perturbations, and permutation controls.

Across the three models, the paper reports a shared pattern: low-dimensional subspaces tend to show lower O-S and higher MS-E. O-S rises sharply and then decreases slightly, while MS-E initially falls and later increases. These trends remain close under randomized PCA and mild HDBSCAN parameter perturbations, but are disrupted by token permutation controls.

## Paper Scope and Repository Extensions

| Analysis | Paper | Repository |
|---|:---:|:---:|
| Nine-scale full-PCA baseline | Yes | Yes |
| Randomized-PCA cross-seed stability | Summary | Full workflow and curves |
| HDBSCAN parameter stability | Summary | Full workflow and curves |
| Token permutation | Summary | Full workflow and curves |
| Length-bucket permutation | Summary | Full workflow and curves |
| Baseline cluster counts, sizes, and noise fractions | Limited | Full results |
| Random orthogonal projection control | No | Yes |
| Twelve-scale low-dimensional supplement | No | Yes |

The last two experiments are repository-only extensions. They use the same clustering and lexical metrics as the paper but address questions that could not be included in the paper-length presentation.

## Main-Study Results

The paper defines the main-study candidate scales as

$$
\mathcal{D}
=
\{d_{\mathrm{model}}\}
\cup
\mathcal{D}_{\mathrm{energy}}
\cup
\mathcal{D}_{\mathrm{elbow}}.
$$

Here, \(d_{\mathrm{model}}\) is the full unembedding dimension, \(\mathcal{D}_{\mathrm{energy}}\) contains the dimensions selected at the cumulative spectral-energy thresholds \(\{50\%, 75\%, 90\%, 95\%\}\), and \(\mathcal{D}_{\mathrm{elbow}}\) contains the top four log-spectrum elbow dimensions. Their union produces nine model-specific candidate scales.

| Model | Selected PCA dimensions |
|---|---|
| Mistral-7B | 5, 142, 997, 2084, 3031, 3459, 3938, 4088, 4096 |
| Mixtral-8x7B | 8, 158, 1111, 2156, 3052, 3457, 3957, 4093, 4096 |
| GPT-oss-20B | 6, 182, 466, 739, 1591, 2264, 2532, 2868, 2880 |

The following curves show the scale-level means and standard deviations of O-S and MS-E under the deterministic full-PCA baseline.

<p align="center">
  <img src="docs/assets/main_mistral.png" alt="Mistral main-study lexical curves" width="32%">
  <img src="docs/assets/main_mixtral.png" alt="Mixtral main-study lexical curves" width="32%">
  <img src="docs/assets/main_gpt_oss.png" alt="GPT-oss main-study lexical curves" width="32%">
</p>

High-resolution versions and control curves are available in [`res/linguistic`](res/linguistic/).

## Paper Appendix Results

This section presents the detailed appendix material that supports the compressed results in the paper. These analyses use the same nine model-specific scales, deterministic full-PCA baseline, randomized-PCA stability runs, and permutation controls described above.

### Scale-level aggregation

For a scale containing \(K\) valid non-noise clusters, let \(m(c)\) be the cluster-level lexical measurement for cluster \(c\). Each cluster contributes equally to the scale-level mean and standard deviation:

$$
\overline{m} = \frac{1}{K}\sum_{c=1}^{K} m(c),
\qquad
\sigma_m = \sqrt{\frac{1}{K}\sum_{c=1}^{K}\left(m(c)-\overline{m}\right)^2}.
$$

For O-S, \(m(c)\) is the mean pairwise orthographic similarity in a cluster. For MS-E, \(m(c)\) is the entropy of the cluster's token-level Unicode script distribution. Noise points are excluded from both metrics.

Exploratory non-pairwise cluster metrics, such as the cross-language ratio \(R(c)\), use the same equal-cluster mean and standard-deviation aggregation. They are reported as auxiliary analyses rather than primary paper claims.

### Baseline clustering statistics

The paper focuses on lexical measurements. The appendix additionally exposes how the underlying HDBSCAN partitions change across the same nine scales.

<p align="center">
  <img src="docs/assets/cluster_counts.png" alt="Number of clusters across scales" width="48%">
  <img src="docs/assets/cluster_sizes.png" alt="Cluster sizes across scales" width="48%">
</p>

The first two scales are structurally unstable: cluster counts and size distributions change sharply. From the middle scales onward, cluster sizes stabilize while the number of identified clusters generally continues to increase.

The fraction of tokens labeled as HDBSCAN noise is:

| Model | SI=1 | SI=2 | SI=3 | SI=4 | SI=5 | SI=6 | SI=7 | SI=8 | SI=9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mistral-7B | 88.4% | 91.3% | 77.1% | 68.9% | 68.0% | 68.3% | 67.9% | 66.5% | 66.9% |
| Mixtral-8x7B | 73.2% | 87.6% | 63.5% | 60.5% | 60.0% | 60.0% | 59.5% | 59.5% | 59.5% |
| GPT-oss-20B | 74.8% | 84.4% | 84.0% | 76.6% | 67.4% | 65.9% | 65.8% | 65.8% | 65.8% |

Here, SI denotes the ascending scale index rather than a shared PCA dimension. Each model has its own dimensions listed in the main-study table.

### Randomized-PCA and permutation rank correlations

Randomized-PCA curves remain strongly aligned with the deterministic full-PCA baseline. Values below are the mean and standard deviation across eight PCA seeds.

| Model | O-S rho | O-S p-value | MS-E rho | MS-E p-value |
|---|---:|---:|---:|---:|
| Mistral-7B | 0.965 ± 0.033 | 0.000 ± 0.000 | 0.960 ± 0.031 | 0.000 ± 0.000 |
| Mixtral-8x7B | 0.956 ± 0.053 | 0.001 ± 0.001 | 0.929 ± 0.053 | 0.001 ± 0.001 |
| GPT-oss-20B | 0.996 ± 0.008 | 0.000 ± 0.000 | 0.985 ± 0.011 | 0.001 ± 0.002 |

The global token-permutation control preserves cluster sizes and non-noise positions but reassigns token IDs. Its mean curves no longer reproduce the baseline trend:

| Model | O-S rho | O-S p-value | MS-E rho | MS-E p-value |
|---|---:|---:|---:|---:|
| Mistral-7B | 0.333 | 0.381 | -0.217 | 0.576 |
| Mixtral-8x7B | -0.233 | 0.546 | -0.283 | 0.460 |
| GPT-oss-20B | -0.450 | 0.224 | -0.900 | 0.001 |

The displayed p-values are rounded to three decimal places. Rank correlation is interpreted together with MAE and R2 because the analysis contains only nine scale observations per curve.

### HDBSCAN parameter stability

The baseline uses `min_cluster_size=5` and `min_samples=5`. The appendix evaluates six nearby configurations: `(6, 5)`, `(5, 6)`, `(6, 6)`, `(4, 5)`, `(5, 4)`, and `(4, 4)`.

Mean absolute error against the baseline curves:

| Model | O-S MAE | MS-E MAE | Joint MAE |
|---|---:|---:|---:|
| Mistral-7B | 0.0126 | 0.0110 | 0.0118 |
| Mixtral-8x7B | 0.0130 | 0.0102 | 0.0116 |
| GPT-oss-20B | 0.0165 | 0.0223 | 0.0194 |

Coefficient of determination against the baseline curves:

| Model | O-S R2 | MS-E R2 | Joint R2 |
|---|---:|---:|---:|
| Mistral-7B | 0.9845 | 0.9151 | 0.9931 |
| Mixtral-8x7B | 0.9813 | 0.8781 | 0.9942 |
| GPT-oss-20B | 0.9655 | 0.9235 | 0.9815 |

The joint Spearman correlations remain significant under these local perturbations. Their mean values are 0.9542 for Mistral-7B, 0.5710 for Mixtral-8x7B, and 0.8000 for GPT-oss-20B. These checks support robustness at a fixed observation resolution; they are not a search for optimal HDBSCAN parameters.

### Length-bucket permutation

The stricter O-S control allows token exchange only within the character-length buckets `1-2`, `3-5`, `6-10`, and `11+`. It preserves the valid token set, cluster sizes, and coarse token-length composition while disrupting the original token-cluster assignment.

| Model | Spearman p-value | MAE | R2 |
|---|---:|---:|---:|
| Mistral-7B | 0.1491 | 0.3514 | -8.7701 |
| Mixtral-8x7B | 0.0610 | 0.3345 | -8.4158 |
| GPT-oss-20B | 0.3260 | 0.3896 | -15.6177 |

All three controls lose conventional rank-correlation significance and differ from the baseline by roughly one third of the full O-S range at each matched scale. The strongly negative R2 values show that the permuted curves reproduce less baseline variation than a constant baseline-mean predictor. This control is applied only to O-S because token length is a direct confound for orthographic similarity, while MS-E is defined from within-cluster script distributions.

### Scale-level control curves

The following appendix plots show two representative randomized-PCA seeds and the ten-seed token-permutation mean. The blue line is O-S, the orange line is MS-E, and the shaded regions show the corresponding cluster-level standard deviations.

<details>
<summary>Randomized PCA: seed 0</summary>

<p align="center">
  <img src="docs/assets/appendix_seed_0_mistral.png" alt="Mistral randomized PCA seed 0" width="32%">
  <img src="docs/assets/appendix_seed_0_mixtral.png" alt="Mixtral randomized PCA seed 0" width="32%">
  <img src="docs/assets/appendix_seed_0_gpt_oss.png" alt="GPT-oss randomized PCA seed 0" width="32%">
</p>

</details>

<details>
<summary>Randomized PCA: seed 42</summary>

<p align="center">
  <img src="docs/assets/appendix_seed_42_mistral.png" alt="Mistral randomized PCA seed 42" width="32%">
  <img src="docs/assets/appendix_seed_42_mixtral.png" alt="Mixtral randomized PCA seed 42" width="32%">
  <img src="docs/assets/appendix_seed_42_gpt_oss.png" alt="GPT-oss randomized PCA seed 42" width="32%">
</p>

</details>

<details>
<summary>Global token permutation</summary>

<p align="center">
  <img src="docs/assets/appendix_permutation_mistral.png" alt="Mistral global token permutation" width="32%">
  <img src="docs/assets/appendix_permutation_mixtral.png" alt="Mixtral global token permutation" width="32%">
  <img src="docs/assets/appendix_permutation_gpt_oss.png" alt="GPT-oss global token permutation" width="32%">
</p>

</details>

## Repository-Only Extension: Random Orthogonal Projection

Randomized PCA and random orthogonal projection are different controls:

- **Randomized PCA** approximates data-aligned principal directions with a stochastic solver. It tests numerical and seed stability.
- **Random orthogonal projection** replaces the PCA directions with random orthogonal directions. It tests whether the lexical curves depend on PCA-aligned subspaces rather than dimension alone.

For each model and seed, the control constructs one full orthogonal basis and takes nested prefixes at the nine main-study dimensions. The experiment uses ten fixed seeds and the same L2 normalization and HDBSCAN configuration as the baseline. At the full model dimension, the orthogonal transform preserves pairwise geometry; the implementation checks partition invariance and reports AMI and ARI against the baseline.

<p align="center">
  <img src="docs/assets/random_projection_mistral.png" alt="Mistral full PCA versus random projection" width="100%">
</p>

<p align="center">
  <img src="docs/assets/random_projection_mixtral.png" alt="Mixtral full PCA versus random projection" width="100%">
</p>

<p align="center">
  <img src="docs/assets/random_projection_gpt_oss.png" alt="GPT-oss full PCA versus random projection" width="100%">
</p>

The random-projection curves differ most strongly from the PCA baseline at low dimensions and converge at the full dimension. This control is implemented in [`4r_random_projection_hdbscan.ipynb`](4r_random_projection_hdbscan.ipynb) and [`cluster_anlys/random_projection.py`](cluster_anlys/random_projection.py).

## Repository-Only Extension: Low12

The low12 experiment defines its candidate scales as

$$
\mathcal{D}_{\mathrm{low12}}
=
\mathcal{D}_{\mathrm{energy}},
$$

where \(\mathcal{D}_{\mathrm{energy}}\) contains the dimensions selected at twelve cumulative spectral-energy thresholds \(\{12\%, 15\%, 18\%, 21\%, 24\%, 27\%, 30\%, 33\%, 36\%, 39\%, 42\%, 45\%\}\).

| Model | Low12 PCA dimensions |
|---|---|
| Mistral-7B | 6, 44, 95, 155, 223, 296, 373, 455, 541, 631, 725, 824 |
| Mixtral-8x7B | 76, 134, 198, 267, 339, 415, 494, 577, 663, 752, 845, 942 |
| GPT-oss-20B | 37, 60, 88, 122, 162, 209, 262, 320, 384, 452, 525, 602 |

Low12 reuses the fixed spectrum and deterministic full-PCA matrices from the main study. Its workflow includes:

- 36 deterministic baseline partitions;
- randomized-PCA cross-seed stability;
- six HDBSCAN parameter perturbations;
- ten-seed global token permutation;
- ten-seed length-bucket permutation;
- O-S, normalized MS-E, Spearman correlation, MAE, and R2;
- persisted plotting-data tables for every published curve.

<p align="center">
  <img src="docs/assets/low12_mistral.png" alt="Mistral low12 baseline" width="32%">
  <img src="docs/assets/low12_mixtral.png" alt="Mixtral low12 baseline" width="32%">
  <img src="docs/assets/low12_gpt_oss.png" alt="GPT-oss low12 baseline" width="32%">
</p>

The complete workflow and its persistent stage contracts are documented in [`1r_low_dimension_spectrum_anlys.ipynb`](1r_low_dimension_spectrum_anlys.ipynb).

Low12 refers to a low-dimensional observation range, not to a lightweight computation. Full reproduction includes hundreds of clustering and metric tasks and requires substantial local storage.

## Repository Map

The [`1r_low_dimension_spectrum_anlys.ipynb`](1r_low_dimension_spectrum_anlys.ipynb) notebook provides an end-to-end demonstration of the packaged analysis pipeline over the low12 domain, which spans cumulative spectral-energy thresholds up to 45%. It makes the complete workflow explicit, from scale selection and representation construction through clustering, lexical metrics, robustness controls, statistics, and reproducible figures, while also demonstrating the pipeline interfaces and persistent artifact contracts used by the project.

### Core modules

| Path | Responsibility |
|---|---|
| [`cluster_anlys/spectrum_pipeline.py`](cluster_anlys/spectrum_pipeline.py) | Spectrum loading, scale selection, and diagnostic outputs |
| [`cluster_anlys/representation_pipeline.py`](cluster_anlys/representation_pipeline.py) | Reusable representation and projection stages |
| [`cluster_anlys/clustering_pipeline.py`](cluster_anlys/clustering_pipeline.py) | Scale tasks, PCA prefixes, HDBSCAN, and partition manifests |
| [`cluster_anlys/metric_pipeline.py`](cluster_anlys/metric_pipeline.py) | O-S, MS-E, result tables, and resumable metric execution |
| [`cluster_anlys/permutation_pipeline.py`](cluster_anlys/permutation_pipeline.py) | Global and length-bucket permutation controls |
| [`cluster_anlys/statistics_pipeline.py`](cluster_anlys/statistics_pipeline.py) | Curve alignment, Spearman, Pearson, MAE, and R2 |
| [`cluster_anlys/plotting_pipeline.py`](cluster_anlys/plotting_pipeline.py) | Reproducible figures and sibling plotting-data tables |
| [`cluster_anlys/experiment_orchestrator.py`](cluster_anlys/experiment_orchestrator.py) | Multi-stage experiment orchestration |

### Experiment notebooks

| Stage | Notebook | Purpose |
|---:|---|---|
| 0 | [`00_init.ipynb`](00_init.ipynb) | Model, tokenizer, and tensor preparation |
| 1 | [`1_spectrum_anlys.ipynb`](1_spectrum_anlys.ipynb) | Main-study spectrum analysis and scale selection |
| 1R | [`1r_low_dimension_spectrum_anlys.ipynb`](1r_low_dimension_spectrum_anlys.ipynb) | Independent low12 supplement |
| 2 | [`2_full_pca.ipynb`](2_full_pca.ipynb) | Deterministic full PCA |
| 3 | [`3_hdbscan_gpu.ipynb`](3_hdbscan_gpu.ipynb) | Baseline GPU HDBSCAN |
| 3R | [`3r_hdbscan_gpu.ipynb`](3r_hdbscan_gpu.ipynb) | HDBSCAN parameter stability |
| 4 | [`4_random PCA_hdbscan.ipynb`](4_random%20PCA_hdbscan.ipynb) | Randomized-PCA cross-seed control |
| 4R | [`4r_random_projection_hdbscan.ipynb`](4r_random_projection_hdbscan.ipynb) | Random orthogonal projection control |
| 5 / 5R | [`5_morphology.ipynb`](5_morphology.ipynb), [`5r_morphology.ipynb`](5r_morphology.ipynb) | Orthographic similarity analyses |
| 6 / 6R | [`6_multi-ligual.ipynb`](6_multi-ligual.ipynb), [`6r_multi-ligual.ipynb`](6r_multi-ligual.ipynb) | Multi-script entropy analyses |
| 7 / 7R | [`7_spearman.ipynb`](7_spearman.ipynb), [`7r_pearson.ipynb`](7r_pearson.ipynb) | Cross-scale association statistics |
| 8 | [`8_permutation.ipynb`](8_permutation.ipynb) | Token permutation controls |
| 9 | [`9_case study.ipynb`](9_case%20study.ipynb) | Cluster-level case studies |
| 10 | [`10_statistic.ipynb`](10_statistic.ipynb), [`10p_statistic.ipynb`](10p_statistic.ipynb), [`10r_statistic.ipynb`](10r_statistic.ipynb) | Baseline, permutation, and stability summaries |

The `R` suffix is historical and does not have one universal meaning. Use the descriptions above rather than inferring an experiment from its filename.

## Installation

Python 3.10 is the reference environment. The package metadata supports Python 3.10 through 3.12.

### Standard analysis and notebook environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### CUDA 12 and RAPIDS environment

The full clustering workflow requires an NVIDIA GPU and RAPIDS cuML. The recommended environment is:

```bash
conda env create -f environment.yml
conda activate cross-scale-geometry
```

An existing CUDA 12 environment can instead use:

```bash
python -m pip install -r requirements-gpu.txt
```

The reference experiments used Python 3.10.19, PyTorch 2.10.0 with CUDA 12.9, and RAPIDS cuML 25.12 on an NVIDIA RTX 4070 Ti SUPER with 16 GB of VRAM.

### Verify the installation

```bash
python -m unittest discover -s tests -q
```

## Data Preparation

No model weights or exported tensor matrices are included. The export utility downloads the requested Hugging Face files and writes the tokenizer, configuration, and selected tensors beneath an output root.

For example:

```bash
python dump_hf_tensors.py \
  --model_name mistralai/Mistral-7B-v0.1 \
  --out_dir . \
  --spaces output_proj
```

Repeat the export for the required models. Authentication may be required by the upstream model repository. The notebooks expect the following local roots:

| Model source | Expected local root |
|---|---|
| `mistralai/Mistral-7B-v0.1` | `mistralai/Mistral-7B-v0.1/` |
| `mistralai/Mixtral-8x7B-v0.1` | `mistralai/Mixtral-8x7B-v0.1/` |
| `openai/gpt-oss-20b` | `gpt-oss/` |

Each root must contain a `tokenizer/` directory and `tensors/output_proj.pt`. See [`00_init.ipynb`](00_init.ipynb) and [`dump_hf_tensors.py`](dump_hf_tensors.py) for the complete export workflow.

## Running the Experiments

The main-study dependency order is:

1. `00_init.ipynb`
2. `1_spectrum_anlys.ipynb`
3. `2_full_pca.ipynb`
4. `3_hdbscan_gpu.ipynb`
5. `5_morphology.ipynb` and `6_multi-ligual.ipynb`
6. `7_spearman.ipynb` and the `10*_statistic.ipynb` notebooks

The stability and control branches start after the baseline PCA and partitions are available:

- `3r_hdbscan_gpu.ipynb` for HDBSCAN parameter perturbations;
- `4_random PCA_hdbscan.ipynb` for randomized PCA;
- `4r_random_projection_hdbscan.ipynb` for random orthogonal projection;
- `8_permutation.ipynb` for permutation controls;
- `1r_low_dimension_spectrum_anlys.ipynb` for the isolated low12 workflow.

Large generated files are written beneath ignored directories such as `comp/` and `comp_supplement/`. A complete reproduction may require hundreds of gigabytes of local storage.

## Result and Persistence Conventions

- `res/spectrum_analysis/` contains compact spectrum arrays, scale tables, and provenance metadata.
- `res/linguistic/` contains paper-scale lexical curves and selected controls.
- `res/clusters/` contains baseline cluster statistics.
- `docs/assets/` contains PNG previews used by this README.
- Generated partitions, PCA matrices, and model tensors remain local and are excluded by `.gitignore`.

The modular metric pipeline persists token IDs and aggregate numerical results. It does not write tokenizer-decoded token strings to metric output tables. Token decoding is performed through the tokenizer only when a metric evaluation requires it.

## Notes on Interpretation

- HDBSCAN is used as a fixed-resolution probe, not as a search for an optimal clustering configuration.
- O-S and MS-E describe surface-level lexical organization, not semantic equivalence.
- Cross-scale agreement of aggregate curves does not imply that individual cluster identities are preserved across seeds or projections.
- AMI, probability-silhouette correlation, local-dimension, and cross-language-ratio figures produced during exploration are not treated as primary paper claims unless explicitly identified as such.

## Citation

Archival citation metadata will be added when it becomes available. Until then, cite the accompanying paper by title:

> From Local Density to Lexical Features: A Cross-Scale Study of Unembedding Spaces.

## License

The source code in this repository is licensed under the [Apache License 2.0](LICENSE). Model weights, tokenizers, and downloaded artifacts remain subject to their upstream licenses and are not redistributed by this repository.
