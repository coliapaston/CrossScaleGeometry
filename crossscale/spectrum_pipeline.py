from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, savgol_filter
from sklearn.utils.extmath import randomized_svd


ArrayLike = np.ndarray | torch.Tensor


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class SpectrumResult:
    method: str
    values: np.ndarray
    scale_axis: np.ndarray
    ordering: str
    measures: dict[str, np.ndarray] = field(default_factory=dict)
    directions: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.scale_axis = np.asarray(self.scale_axis)
        self.measures = {
            str(name): np.asarray(values) for name, values in self.measures.items()
        }
        if self.directions is not None:
            self.directions = np.asarray(self.directions)
        if self.values.ndim != 1:
            raise ValueError("Spectrum values must be one-dimensional")
        if self.scale_axis.ndim != 1:
            raise ValueError("Scale axis must be one-dimensional")
        if len(self.values) != len(self.scale_axis):
            raise ValueError("Spectrum values and scale axis must have equal length")
        if len(np.unique(self.scale_axis)) != len(self.scale_axis):
            raise ValueError("Scale axis values must be unique")
        for name, values in self.measures.items():
            if values.ndim != 1 or len(values) != len(self.values):
                raise ValueError(
                    f"Measure '{name}' must be one-dimensional and match spectrum length"
                )

    @property
    def singular_values(self) -> np.ndarray:
        if self.method != "svd":
            raise AttributeError("singular_values are only available for SVD spectra")
        return self.values

    @property
    def energy(self) -> np.ndarray:
        return self.require_measure("energy")

    @property
    def energy_ratio(self) -> np.ndarray:
        return self.require_measure("energy_ratio")

    @property
    def cum_energy_ratio(self) -> np.ndarray:
        return self.require_measure("cumulative_energy_ratio")

    def require_measure(self, name: str) -> np.ndarray:
        if name not in self.measures:
            raise ValueError(
                f"Spectrum method '{self.method}' does not provide measure '{name}'"
            )
        return self.measures[name]

    def scale_to_index(self, scale: int) -> int:
        matches = np.flatnonzero(self.scale_axis == scale)
        if len(matches) != 1:
            raise ValueError(f"Scale {scale} is not present exactly once in scale_axis")
        return int(matches[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "values": self.values,
            "scale_axis": self.scale_axis,
            "ordering": self.ordering,
            "measures": self.measures,
            "directions": self.directions,
            "metadata": self.metadata,
        }


@dataclass
class ScaleSelectionResult:
    selector: str
    selected_indices: np.ndarray
    selected_scales: list[int]
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.selected_indices = np.asarray(self.selected_indices, dtype=int)
        self.selected_scales = [int(scale) for scale in self.selected_scales]
        self.scores = {
            str(name): np.asarray(values) for name, values in self.scores.items()
        }
        if len(self.selected_indices) != len(self.selected_scales):
            raise ValueError("Selected indices and selected scales must have equal length")

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "selected_indices": self.selected_indices.tolist(),
            "selected_scales": self.selected_scales,
            "parameters": _json_safe(self.parameters),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class CandidateScaleSet:
    scales: list[int]
    provenance: dict[int, list[str]]
    full_dimension: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scales = sorted(set(int(scale) for scale in self.scales))
        normalized: dict[int, list[str]] = {}
        for scale, sources in self.provenance.items():
            normalized[int(scale)] = sorted(set(str(source) for source in sources))
        self.provenance = normalized
        if set(self.scales) != set(self.provenance):
            raise ValueError("Candidate scales and provenance keys must match")
        if self.full_dimension is not None:
            self.full_dimension = int(self.full_dimension)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scales": self.scales,
            "provenance": {
                str(scale): sources for scale, sources in self.provenance.items()
            },
            "full_dimension": self.full_dimension,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateScaleSet":
        return cls(
            scales=[int(scale) for scale in data["scales"]],
            provenance={
                int(scale): [str(source) for source in sources]
                for scale, sources in data["provenance"].items()
            },
            full_dimension=data.get("full_dimension"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "scale": scale,
                    "sources": ";".join(self.provenance[scale]),
                    "is_full_dimension": scale == self.full_dimension,
                }
                for scale in self.scales
            ]
        )


@dataclass
class SpectralAnalysisResult:
    spectrum: SpectrumResult
    selections: dict[str, ScaleSelectionResult]
    candidate_scales: CandidateScaleSet

    def selection(self, name: str) -> ScaleSelectionResult:
        if name not in self.selections:
            raise KeyError(f"Selection '{name}' is not available")
        return self.selections[name]

    @property
    def singular_values(self) -> np.ndarray:
        return self.spectrum.singular_values

    @property
    def energy(self) -> np.ndarray:
        return self.spectrum.energy

    @property
    def energy_ratio(self) -> np.ndarray:
        return self.spectrum.energy_ratio

    @property
    def cum_energy_ratio(self) -> np.ndarray:
        return self.spectrum.cum_energy_ratio

    @property
    def log_spectrum(self) -> np.ndarray:
        return self.selection("elbow").scores["log_spectrum"]

    @property
    def log_spectrum_smooth(self) -> np.ndarray:
        return self.selection("elbow").scores["smoothing"]

    @property
    def curvature(self) -> np.ndarray:
        return self.selection("elbow").scores["curvature"]

    @property
    def inflection_dims(self) -> list[int]:
        return self.selection("elbow").selected_scales

    @property
    def peak_indices(self) -> np.ndarray:
        return self.selection("elbow").scores["curvature_peak_indices"]

    @property
    def peak_scores(self) -> np.ndarray:
        return self.selection("elbow").scores["peak_scores"]

    @property
    def energy_dims(self) -> list[int]:
        return self.selection("energy").selected_scales

    @property
    def energy_threshold_map(self) -> dict[float, int]:
        return self.selection("energy").metadata["threshold_to_scale"]

    @property
    def candidate_dims(self) -> list[int]:
        return self.candidate_scales.scales

    @property
    def svd_mode(self) -> str:
        return str(self.spectrum.metadata.get("svd_mode", "unknown"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spectrum": self.spectrum.to_dict(),
            "selections": {
                name: selection.to_summary_dict()
                for name, selection in self.selections.items()
            },
            "candidate_scales": self.candidate_scales.to_dict(),
        }


class SpectrumAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        W: ArrayLike,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SpectrumResult:
        raise NotImplementedError


@dataclass
class SVDAnalyzer(SpectrumAnalyzer):
    svd_mode: Literal["full", "randomized"] = "full"
    randomized_n_components: Optional[int] = None
    randomized_n_iter: int = 4
    random_state: int = 0
    device: Optional[str] = None

    def analyze(
        self,
        W: ArrayLike,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SpectrumResult:
        if len(W.shape) != 2:
            raise ValueError(f"Expected a two-dimensional matrix, got shape {tuple(W.shape)}")
        spectrum = compute_spectrum(
            W=W,
            svd_mode=self.svd_mode,
            randomized_n_components=self.randomized_n_components,
            randomized_n_iter=self.randomized_n_iter,
            random_state=self.random_state,
            device=self.device,
        )
        values = spectrum["singular_values"]
        input_shape = [int(size) for size in W.shape]
        result_metadata = dict(metadata or {})
        result_metadata.update(
            {
                "svd_mode": self.svd_mode,
                "input_shape": input_shape,
                "feature_dimension": input_shape[1],
                "spectrum_length": len(values),
                "randomized_n_components": self.randomized_n_components,
                "randomized_n_iter": self.randomized_n_iter,
                "random_state": self.random_state,
                "device": self.device,
            }
        )
        return SpectrumResult(
            method="svd",
            values=values,
            scale_axis=np.arange(1, len(values) + 1, dtype=int),
            ordering="descending",
            measures={
                "energy": spectrum["energy"],
                "energy_ratio": spectrum["energy_ratio"],
                "cumulative_energy_ratio": spectrum["cum_energy_ratio"],
            },
            directions=None,
            metadata=result_metadata,
        )


class ScaleSelector(ABC):
    @abstractmethod
    def select(self, spectrum: SpectrumResult) -> ScaleSelectionResult:
        raise NotImplementedError


@dataclass
class ElbowSelector(ScaleSelector):
    use_squared: bool = True
    smoothing: Optional[Literal["savgol", "median"]] = None
    smooth_window: int = 51
    smooth_polyorder: int = 3
    take_abs_curvature: bool = False
    top_k: int = 3
    min_gap: int = 32
    use_abs_for_peak_ranking: bool = True
    prominence: Optional[float] = None
    name: str = "elbow"

    def select(self, spectrum: SpectrumResult) -> ScaleSelectionResult:
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.min_gap < 0:
            raise ValueError("min_gap must be non-negative")
        log_spectrum = compute_log_spectrum(
            spectrum.values,
            use_squared=self.use_squared,
        )
        smoothed = smooth_spectrum(
            log_spectrum,
            method=self.smoothing,
            window=self.smooth_window,
            polyorder=self.smooth_polyorder,
        )
        curvature = compute_curvature(
            smoothed,
            take_abs=self.take_abs_curvature,
        )
        _, curvature_peak_indices, peak_scores = find_inflection_dims(
            curvature,
            top_k=self.top_k,
            min_gap=self.min_gap,
            use_abs_for_ranking=self.use_abs_for_peak_ranking,
            prominence=self.prominence,
        )
        selected_indices = curvature_peak_indices + 1
        selected_scales = [
            int(spectrum.scale_axis[index]) for index in selected_indices
        ]
        return ScaleSelectionResult(
            selector=self.name,
            selected_indices=selected_indices,
            selected_scales=selected_scales,
            scores={
                "log_spectrum": log_spectrum,
                "smoothing": smoothed,
                "curvature": curvature,
                "curvature_peak_indices": curvature_peak_indices,
                "peak_scores": peak_scores,
            },
            parameters={
                "use_squared": self.use_squared,
                "smoothing": self.smoothing,
                "smooth_window": self.smooth_window,
                "smooth_polyorder": self.smooth_polyorder,
                "take_abs_curvature": self.take_abs_curvature,
                "top_k": self.top_k,
                "min_gap": self.min_gap,
                "gap_mode": "curvature_index",
                "use_abs_for_peak_ranking": self.use_abs_for_peak_ranking,
                "prominence": self.prominence,
            },
        )


@dataclass
class EnergyThresholdSelector(ScaleSelector):
    thresholds: Iterable[float] = (0.5, 0.7, 0.8, 0.9, 0.95)
    name: str = "energy"

    def select(self, spectrum: SpectrumResult) -> ScaleSelectionResult:
        thresholds = tuple(float(value) for value in self.thresholds)
        if not thresholds:
            return ScaleSelectionResult(
                selector=self.name,
                selected_indices=np.array([], dtype=int),
                selected_scales=[],
                parameters={"thresholds": []},
                metadata={"threshold_to_index": {}, "threshold_to_scale": {}},
            )
        if any(not 0.0 < threshold <= 1.0 for threshold in thresholds):
            raise ValueError("Energy thresholds must satisfy 0 < threshold <= 1")
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("Energy thresholds must be unique")
        cumulative_energy = spectrum.require_measure("cumulative_energy_ratio")
        _, threshold_to_dimension = find_energy_dims(
            cumulative_energy,
            thresholds=thresholds,
        )
        threshold_to_index = {
            threshold: int(dimension - 1)
            for threshold, dimension in threshold_to_dimension.items()
        }
        threshold_to_scale = {
            threshold: int(spectrum.scale_axis[index])
            for threshold, index in threshold_to_index.items()
        }
        selected_indices = np.array(
            sorted(set(threshold_to_index.values())),
            dtype=int,
        )
        selected_scales = [
            int(spectrum.scale_axis[index]) for index in selected_indices
        ]
        return ScaleSelectionResult(
            selector=self.name,
            selected_indices=selected_indices,
            selected_scales=selected_scales,
            scores={"cumulative_energy_ratio": cumulative_energy},
            parameters={"thresholds": list(thresholds)},
            metadata={
                "threshold_to_index": threshold_to_index,
                "threshold_to_scale": threshold_to_scale,
            },
        )


@dataclass
class ManualScaleSelector(ScaleSelector):
    scales: Iterable[int] = ()
    name: str = "manual"

    def select(self, spectrum: SpectrumResult) -> ScaleSelectionResult:
        scales = sorted(set(int(scale) for scale in self.scales))
        selected_indices = np.array(
            [spectrum.scale_to_index(scale) for scale in scales],
            dtype=int,
        )
        return ScaleSelectionResult(
            selector=self.name,
            selected_indices=selected_indices,
            selected_scales=scales,
            parameters={"scales": scales},
        )


@dataclass
class FullDimensionSelector(ScaleSelector):
    full_dimension: Optional[int] = None
    name: str = "full_dimension"

    def select(self, spectrum: SpectrumResult) -> ScaleSelectionResult:
        dimension = self.full_dimension
        if dimension is None:
            dimension = spectrum.metadata.get("feature_dimension")
        if dimension is None:
            raise ValueError("Full dimension is not available in spectrum metadata")
        dimension = int(dimension)
        index = spectrum.scale_to_index(dimension)
        return ScaleSelectionResult(
            selector=self.name,
            selected_indices=np.array([index], dtype=int),
            selected_scales=[dimension],
            parameters={"full_dimension": dimension},
        )


def _to_torch_float32(W: ArrayLike, device: Optional[str] = None) -> torch.Tensor:
    if isinstance(W, np.ndarray):
        x = torch.from_numpy(W)
    elif isinstance(W, torch.Tensor):
        x = W.detach()
    else:
        raise TypeError(f"Unsupported type for W: {type(W)}")

    x = x.to(dtype=torch.float32)
    if device is not None:
        x = x.to(device)
    return x


def compute_spectrum(
    W: ArrayLike,
    svd_mode: Literal["full", "randomized"] = "full",
    randomized_n_components: Optional[int] = None,
    randomized_n_iter: int = 4,
    random_state: int = 0,
    device: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """
    Compute a singular value spectrum.

    Full mode computes all singular values with torch.linalg.svdvals.
    Randomized mode computes a leading truncated spectrum. Energy ratios in
    randomized mode are relative to that truncated spectrum.
    """
    if svd_mode == "full":
        x = _to_torch_float32(W, device=device)
        singular_values = torch.linalg.svdvals(x).detach().cpu().numpy()
    elif svd_mode == "randomized":
        x = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else np.asarray(W)
        max_rank = min(x.shape)
        n_components = randomized_n_components or max_rank
        n_components = min(n_components, max_rank)
        _, singular_values, _ = randomized_svd(
            x,
            n_components=n_components,
            n_iter=randomized_n_iter,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown svd_mode: {svd_mode}")

    energy = singular_values**2
    energy_ratio = energy / energy.sum()
    cumulative_energy_ratio = np.cumsum(energy_ratio)

    return {
        "singular_values": singular_values,
        "energy": energy,
        "energy_ratio": energy_ratio,
        "cum_energy_ratio": cumulative_energy_ratio,
    }


def compute_log_spectrum(
    singular_values: np.ndarray,
    use_squared: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    if use_squared:
        return np.log((singular_values**2) + eps)
    return np.log(singular_values + eps)


def smooth_spectrum(
    y: np.ndarray,
    method: Optional[Literal["savgol", "median"]] = None,
    window: int = 51,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply optional smoothing to a one-dimensional spectrum."""
    if method is None:
        return y.copy()

    if len(y) < 5:
        return y.copy()

    if method == "savgol":
        if window >= len(y):
            window = len(y) - 1
        if window < 5:
            return y.copy()
        if window % 2 == 0:
            window += 1
        polyorder = min(polyorder, window - 1)
        return savgol_filter(y, window_length=window, polyorder=polyorder)

    if method == "median":
        return (
            pd.Series(y)
            .rolling(window=window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )

    raise ValueError(f"Unknown smoothing method: {method}")


def compute_curvature(y: np.ndarray, take_abs: bool = False) -> np.ndarray:
    curvature = np.diff(y, n=2)
    if take_abs:
        curvature = np.abs(curvature)
    return curvature


def find_inflection_dims(
    curvature: np.ndarray,
    top_k: int = 10,
    min_gap: int = 32,
    use_abs_for_ranking: bool = True,
    prominence: Optional[float] = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Find top curvature peaks with a minimum index spacing constraint."""
    if len(curvature) == 0:
        return [], np.array([], dtype=int), np.array([], dtype=float)

    peaks, _ = find_peaks(curvature, prominence=prominence)
    if len(peaks) == 0:
        return [], np.array([], dtype=int), np.array([], dtype=float)

    ranking_scores = np.abs(curvature[peaks]) if use_abs_for_ranking else curvature[peaks]
    order = np.argsort(ranking_scores)[::-1]

    selected: list[int] = []
    for index in order:
        peak = int(peaks[index])
        if all(abs(peak - existing) >= min_gap for existing in selected):
            selected.append(peak)
        if len(selected) >= top_k:
            break

    selected = sorted(selected)
    inflection_dims = [int(peak + 2) for peak in selected]
    peak_scores = np.array([curvature[peak] for peak in selected], dtype=float)
    return inflection_dims, np.array(selected, dtype=int), peak_scores


def find_energy_dims(
    cum_energy_ratio: np.ndarray,
    thresholds: Iterable[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> tuple[list[int], dict[float, int]]:
    dims: list[int] = []
    threshold_map: dict[float, int] = {}
    for threshold in thresholds:
        dimension = min(
            int(np.searchsorted(cum_energy_ratio, threshold) + 1),
            len(cum_energy_ratio),
        )
        dims.append(dimension)
        threshold_map[float(threshold)] = dimension
    return sorted(set(dims)), threshold_map


def select_candidate_dims(
    inflection_dims: Iterable[int],
    energy_dims: Optional[Iterable[int]] = None,
    combine_mode: Literal["union", "inflection_only"] = "union",
) -> list[int]:
    inflection_dims = [int(value) for value in inflection_dims]
    energy_dims = [] if energy_dims is None else [int(value) for value in energy_dims]

    if combine_mode == "inflection_only":
        return sorted(set(inflection_dims))
    if combine_mode == "union":
        return sorted(set(inflection_dims) | set(energy_dims))
    raise ValueError(f"Unknown combine_mode: {combine_mode}")


def merge_scale_selections(
    selections: Iterable[ScaleSelectionResult],
    full_dimension: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CandidateScaleSet:
    provenance: dict[int, list[str]] = {}
    for selection in selections:
        for scale in selection.selected_scales:
            provenance.setdefault(int(scale), []).append(selection.selector)
    return CandidateScaleSet(
        scales=sorted(provenance),
        provenance=provenance,
        full_dimension=full_dimension,
        metadata=dict(metadata or {}),
    )


def run_spectrum_pipeline(
    W: ArrayLike,
    svd_mode: Literal["full", "randomized"] = "full",
    randomized_n_components: Optional[int] = None,
    randomized_n_iter: int = 4,
    random_state: int = 0,
    device: Optional[str] = None,
    use_squared: bool = True,
    smoothing: Optional[Literal["savgol", "median"]] = None,
    smooth_window: int = 51,
    smooth_polyorder: int = 3,
    take_abs_curvature: bool = False,
    top_k_inflections: int = 3,
    min_gap: int = 32,
    use_abs_for_peak_ranking: bool = True,
    peak_prominence: Optional[float] = None,
    energy_thresholds: Iterable[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
    combine_mode: Literal["union", "inflection_only"] = "union",
    manual_scales: Iterable[int] = (),
    include_full_dimension: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SpectralAnalysisResult:
    analyzer = SVDAnalyzer(
        svd_mode=svd_mode,
        randomized_n_components=randomized_n_components,
        randomized_n_iter=randomized_n_iter,
        random_state=random_state,
        device=device,
    )
    spectrum = analyzer.analyze(W, metadata=metadata)
    elbow_selection = ElbowSelector(
        use_squared=use_squared,
        smoothing=smoothing,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        take_abs_curvature=take_abs_curvature,
        top_k=top_k_inflections,
        min_gap=min_gap,
        use_abs_for_peak_ranking=use_abs_for_peak_ranking,
        prominence=peak_prominence,
    ).select(spectrum)
    energy_selection = EnergyThresholdSelector(
        thresholds=energy_thresholds,
    ).select(spectrum)
    manual_selection = ManualScaleSelector(scales=manual_scales).select(spectrum)

    selections = {
        "elbow": elbow_selection,
        "energy": energy_selection,
        "manual": manual_selection,
    }
    merged_selections = [elbow_selection]
    if combine_mode == "union":
        merged_selections.append(energy_selection)
    elif combine_mode != "inflection_only":
        raise ValueError(f"Unknown combine_mode: {combine_mode}")
    if manual_selection.selected_scales:
        merged_selections.append(manual_selection)

    full_dimension = int(W.shape[1])
    if include_full_dimension:
        full_selection = FullDimensionSelector(full_dimension).select(spectrum)
        selections["full_dimension"] = full_selection
        merged_selections.append(full_selection)

    candidate_scales = merge_scale_selections(
        merged_selections,
        full_dimension=full_dimension,
        metadata={"combine_mode": combine_mode},
    )
    return SpectralAnalysisResult(
        spectrum=spectrum,
        selections=selections,
        candidate_scales=candidate_scales,
    )


def save_spectrum_result(result: SpectrumResult, path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.suffix != ".npz":
        raise ValueError("Spectrum result path must use the .npz suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "values": result.values,
        "scale_axis": result.scale_axis,
    }
    for name, values in result.measures.items():
        payload[f"measure__{name}"] = values
    if result.directions is not None:
        payload["directions"] = result.directions
    header = {
        "format_version": 1,
        "method": result.method,
        "ordering": result.ordering,
        "metadata": _json_safe(result.metadata),
        "measure_names": sorted(result.measures),
        "has_directions": result.directions is not None,
    }
    payload["header_json"] = np.asarray(json.dumps(header, sort_keys=True))
    np.savez_compressed(output_path, **payload)
    return output_path


def load_spectrum_result(path: str | Path) -> SpectrumResult:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        header = json.loads(str(data["header_json"].item()))
        measures = {
            name: np.array(data[f"measure__{name}"], copy=True)
            for name in header["measure_names"]
        }
        directions = (
            np.array(data["directions"], copy=True)
            if header["has_directions"]
            else None
        )
        return SpectrumResult(
            method=str(header["method"]),
            values=np.array(data["values"], copy=True),
            scale_axis=np.array(data["scale_axis"], copy=True),
            ordering=str(header["ordering"]),
            measures=measures,
            directions=directions,
            metadata=dict(header.get("metadata", {})),
        )


def save_analysis_summary(
    result: SpectralAnalysisResult,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    if output_path.suffix != ".json":
        raise ValueError("Analysis summary path must use the .json suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "format_version": 1,
        "spectrum": {
            "method": result.spectrum.method,
            "ordering": result.spectrum.ordering,
            "metadata": _json_safe(result.spectrum.metadata),
        },
        "selections": {
            name: selection.to_summary_dict()
            for name, selection in result.selections.items()
        },
        "candidate_scales": result.candidate_scales.to_dict(),
    }
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_analysis_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_candidate_scale_set(
    candidate_scales: CandidateScaleSet,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    if output_path.suffix != ".json":
        raise ValueError("Candidate scale path must use the .json suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(candidate_scales.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_candidate_scale_set(path: str | Path) -> CandidateScaleSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CandidateScaleSet.from_dict(data)


def energy_threshold_frame(
    spectrum: SpectrumResult,
    selection: ScaleSelectionResult,
) -> pd.DataFrame:
    threshold_to_index = selection.metadata.get("threshold_to_index", {})
    threshold_to_scale = selection.metadata.get("threshold_to_scale", {})
    cumulative_energy = spectrum.require_measure("cumulative_energy_ratio")
    records = []
    for threshold in sorted(threshold_to_scale):
        index = int(threshold_to_index[threshold])
        records.append(
            {
                "threshold": float(threshold),
                "index": index,
                "scale": int(threshold_to_scale[threshold]),
                "cumulative_energy": float(cumulative_energy[index]),
            }
        )
    return pd.DataFrame.from_records(records)


def print_spectrum_summary(
    result: SpectralAnalysisResult,
    top_n: int = 10,
) -> None:
    singular_values = result.singular_values
    energy_ratio = result.energy_ratio
    cumulative_energy = result.cum_energy_ratio

    print("=== Spectrum Summary ===\n")
    print("Top singular values:")
    for index in range(min(top_n, len(singular_values))):
        print(
            f"{index + 1:4d}  singular_value={singular_values[index]:.6f}  "
            f"energy_ratio={energy_ratio[index]:.6f}"
        )

    print("\nEnergy thresholds:")
    for threshold, scale in sorted(result.energy_threshold_map.items()):
        index = result.spectrum.scale_to_index(scale)
        print(
            f"threshold={threshold:.4f}  scale={scale:4d}  "
            f"cumulative_energy={cumulative_energy[index]:.4f}"
        )

    print("\nElbow scales:")
    print(result.inflection_dims)
    print("\nCandidate scales:")
    print(result.candidate_dims)
    print("\nScale provenance:")
    print(result.candidate_scales.to_frame().to_string(index=False))

    if result.svd_mode == "randomized":
        print(
            "\n[Warning] Randomized SVD energy thresholds are relative to the "
            "truncated spectrum, not the full matrix energy."
        )


def plot_spectrum_diagnostics(
    result: SpectralAnalysisResult,
    model_name: str,
    xscale: Literal["linear", "log"] = "linear",
    figsize: tuple[int, int] = (11, 8),
    fontsize: int = 16,
    out_dir: str | Path = "res/spectrum",
) -> Path:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = str(model_name).replace("/", "_")
    output_path = output_dir / f"{safe_model_name}_spectrum_diagnostics.pdf"

    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    scale_axis = result.spectrum.scale_axis
    axes[0].plot(scale_axis, result.log_spectrum, label="log spectrum", alpha=0.8)
    if not np.allclose(result.log_spectrum, result.log_spectrum_smooth):
        axes[0].plot(
            scale_axis,
            result.log_spectrum_smooth,
            label="smoothed spectrum",
            linewidth=2,
        )

    for scale in result.inflection_dims:
        axes[0].axvline(scale, color="red", linestyle="--", alpha=0.6)
    for scale in result.energy_dims:
        axes[0].axvline(scale, color="green", linestyle=":", alpha=0.6)

    curvature_axis = scale_axis[1:-1]
    axes[1].plot(curvature_axis, result.curvature, label="curvature", alpha=0.8)
    for scale in result.inflection_dims:
        axes[1].axvline(scale, color="red", linestyle="--", alpha=0.6)

    use_squared = bool(result.selection("elbow").parameters["use_squared"])
    axes[0].set_title(f"Spectrum of {safe_model_name}", fontsize=fontsize + 2)
    axes[0].set_ylabel(
        "log(sigma^2)" if use_squared else "log(sigma)",
        fontsize=fontsize,
    )
    axes[1].set_xlabel("scale", fontsize=fontsize)
    axes[1].set_ylabel("second difference", fontsize=fontsize)
    axes[1].set_xscale(xscale)
    for axis in axes:
        axis.tick_params(axis="both", labelsize=fontsize - 2)
        axis.legend(fontsize=fontsize - 2)

    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()
    return output_path
