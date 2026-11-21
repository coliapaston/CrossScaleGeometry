"""Typed plotting specifications without rendering side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .metric_pipeline import MetricResultTable
from .pipeline_core import stable_config_hash
from .statistics_pipeline import CurveAligner, CurveQuery


PLOT_AXES = ("primary", "secondary")
PLOT_SCALES = ("linear", "log", "symlog", "logit")
PLOT_DATA_COLUMNS = (
    "line_index",
    "label",
    "axis",
    "curve_id",
    "uncertainty_curve_id",
    "scale",
    "value",
    "uncertainty",
)


@dataclass(frozen=True)
class AxisSpec:
    """Describe one labeled plot axis independently of a backend."""

    label: str
    limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.limits is None:
            return
        if not isinstance(self.limits, tuple) or len(self.limits) != 2:
            raise TypeError("limits must be a two-value tuple when set")
        lower, upper = self.limits
        if isinstance(lower, bool) or isinstance(upper, bool):
            raise TypeError("axis limits must be numeric")
        try:
            lower_value = float(lower)
            upper_value = float(upper)
        except (TypeError, ValueError) as error:
            raise TypeError("axis limits must be numeric") from error
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            raise ValueError("axis limits must be finite")
        if not lower_value < upper_value:
            raise ValueError("axis limits must be strictly increasing")
        object.__setattr__(self, "limits", (lower_value, upper_value))


@dataclass(frozen=True)
class LegendSpec:
    """Describe backend-independent legend behavior."""

    visible: bool = True
    location: str = "best"
    title: str | None = None
    columns: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.visible, bool):
            raise TypeError("visible must be a boolean")
        if not isinstance(self.location, str) or not self.location.strip():
            raise ValueError("location must be a non-empty string")
        if self.title is not None and (
            not isinstance(self.title, str) or not self.title.strip()
        ):
            raise ValueError("title must be a non-empty string when set")
        if isinstance(self.columns, bool) or not isinstance(self.columns, int):
            raise TypeError("columns must be an integer")
        if self.columns <= 0:
            raise ValueError("columns must be positive")


@dataclass(frozen=True)
class LineSpec:
    """Select one curve and define its backend-independent visual identity."""

    curve_query: CurveQuery
    label: str
    color: str | None = None
    marker: str | None = None
    uncertainty_query: CurveQuery | None = None
    axis: str = "primary"

    def __post_init__(self) -> None:
        if not isinstance(self.curve_query, CurveQuery):
            raise TypeError("curve_query must be a CurveQuery")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        for field_name in ("color", "marker"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be a non-empty string when set")
        if self.uncertainty_query is not None:
            if not isinstance(self.uncertainty_query, CurveQuery):
                raise TypeError("uncertainty_query must be a CurveQuery when set")
            if self.uncertainty_query.curve_id == self.curve_query.curve_id:
                raise ValueError("uncertainty_query must differ from curve_query")
        if self.axis not in PLOT_AXES:
            raise ValueError(f"axis must be one of {PLOT_AXES}")


@dataclass(frozen=True)
class FigureSpec:
    """Describe a complete figure request without loading or plotting data."""

    lines: tuple[LineSpec, ...]
    x_axis: AxisSpec
    y_axis: AxisSpec
    output_path: Path
    title: str | None = None
    x_scale: str = "linear"
    y_scale: str = "linear"
    legend: LegendSpec = field(default_factory=LegendSpec)
    secondary_y_axis: AxisSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lines, tuple) or not self.lines:
            raise ValueError("lines must be a non-empty tuple")
        if any(not isinstance(line, LineSpec) for line in self.lines):
            raise TypeError("lines must contain LineSpec objects")
        labels = tuple(line.label for line in self.lines)
        if len(labels) != len(set(labels)):
            raise ValueError("line labels must be unique")
        if not isinstance(self.x_axis, AxisSpec):
            raise TypeError("x_axis must be an AxisSpec")
        if not isinstance(self.y_axis, AxisSpec):
            raise TypeError("y_axis must be an AxisSpec")
        if self.secondary_y_axis is not None and not isinstance(
            self.secondary_y_axis,
            AxisSpec,
        ):
            raise TypeError("secondary_y_axis must be an AxisSpec when set")
        has_secondary_line = any(line.axis == "secondary" for line in self.lines)
        if has_secondary_line and self.secondary_y_axis is None:
            raise ValueError("secondary lines require secondary_y_axis")
        if self.title is not None and (
            not isinstance(self.title, str) or not self.title.strip()
        ):
            raise ValueError("title must be a non-empty string when set")
        for field_name in ("x_scale", "y_scale"):
            value = getattr(self, field_name)
            if value not in PLOT_SCALES:
                raise ValueError(f"{field_name} must be one of {PLOT_SCALES}")
        if not isinstance(self.legend, LegendSpec):
            raise TypeError("legend must be a LegendSpec")
        if not isinstance(self.output_path, (str, Path)):
            raise TypeError("output_path must be a path-like string or Path")
        output_path = Path(self.output_path)
        if not output_path.name or not output_path.suffix:
            raise ValueError("output_path must identify a file with a suffix")
        object.__setattr__(self, "output_path", output_path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the figure request."""

        def axis_dict(axis: AxisSpec | None) -> dict[str, Any] | None:
            if axis is None:
                return None
            return {
                "label": axis.label,
                "limits": None if axis.limits is None else list(axis.limits),
            }

        return {
            "schema_version": 1,
            "lines": [
                {
                    "curve_query": line.curve_query.to_dict(),
                    "label": line.label,
                    "color": line.color,
                    "marker": line.marker,
                    "uncertainty_query": (
                        None
                        if line.uncertainty_query is None
                        else line.uncertainty_query.to_dict()
                    ),
                    "axis": line.axis,
                }
                for line in self.lines
            ],
            "x_axis": axis_dict(self.x_axis),
            "y_axis": axis_dict(self.y_axis),
            "secondary_y_axis": axis_dict(self.secondary_y_axis),
            "title": self.title,
            "x_scale": self.x_scale,
            "y_scale": self.y_scale,
            "legend": {
                "visible": self.legend.visible,
                "location": self.legend.location,
                "title": self.legend.title,
                "columns": self.legend.columns,
            },
            "output_path": str(self.output_path),
        }


@dataclass(frozen=True)
class PlotLineData:
    """Hold the exact ordered observations prepared for one plotted line."""

    line_spec: LineSpec
    curve_id: str
    scales: tuple[int, ...]
    values: tuple[float, ...]
    uncertainty_curve_id: str | None = None
    uncertainty_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.line_spec, LineSpec):
            raise TypeError("line_spec must be a LineSpec")
        if not isinstance(self.curve_id, str) or not self.curve_id.strip():
            raise ValueError("curve_id must be a non-empty string")
        if not isinstance(self.scales, tuple) or not self.scales:
            raise ValueError("scales must be a non-empty tuple")
        if any(
            isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0
            for scale in self.scales
        ):
            raise ValueError("scales must contain positive integers")
        if len(self.scales) != len(set(self.scales)):
            raise ValueError("scales must be unique")
        if tuple(sorted(self.scales)) != self.scales:
            raise ValueError("scales must be sorted")
        if not isinstance(self.values, tuple) or len(self.values) != len(self.scales):
            raise ValueError("values must be a tuple matching scales length")
        if not np.all(np.isfinite(np.asarray(self.values, dtype=float))):
            raise ValueError("values must be finite")
        has_uncertainty_id = self.uncertainty_curve_id is not None
        has_uncertainty_values = self.uncertainty_values is not None
        if has_uncertainty_id != has_uncertainty_values:
            raise ValueError(
                "uncertainty_curve_id and uncertainty_values must be set together"
            )
        if has_uncertainty_id:
            if (
                not isinstance(self.uncertainty_curve_id, str)
                or not self.uncertainty_curve_id.strip()
            ):
                raise ValueError("uncertainty_curve_id must be non-empty when set")
            if self.uncertainty_curve_id == self.curve_id:
                raise ValueError("uncertainty curve must differ from the main curve")
            if (
                not isinstance(self.uncertainty_values, tuple)
                or len(self.uncertainty_values) != len(self.scales)
            ):
                raise ValueError(
                    "uncertainty_values must be a tuple matching scales length"
                )
            uncertainty = np.asarray(self.uncertainty_values, dtype=float)
            if not np.all(np.isfinite(uncertainty)):
                raise ValueError("uncertainty values must be finite")
            if np.any(uncertainty < 0.0):
                raise ValueError("uncertainty values must be non-negative")


@dataclass(frozen=True)
class PlotData:
    """Collect all observations actually selected for one figure request."""

    figure_spec: FigureSpec
    lines: tuple[PlotLineData, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.figure_spec, FigureSpec):
            raise TypeError("figure_spec must be a FigureSpec")
        if not isinstance(self.lines, tuple) or not self.lines:
            raise ValueError("lines must be a non-empty tuple")
        if any(not isinstance(line, PlotLineData) for line in self.lines):
            raise TypeError("lines must contain PlotLineData objects")
        prepared_specs = tuple(line.line_spec for line in self.lines)
        if prepared_specs != self.figure_spec.lines:
            raise ValueError("prepared lines must match figure line order exactly")

    def to_frame(self) -> pd.DataFrame:
        """Return the exact plotting observations as a long-form table."""

        rows = []
        for line_index, line in enumerate(self.lines):
            uncertainty_values = line.uncertainty_values
            for point_index, (scale, value) in enumerate(
                zip(line.scales, line.values)
            ):
                rows.append({
                    "line_index": line_index,
                    "label": line.line_spec.label,
                    "axis": line.line_spec.axis,
                    "curve_id": line.curve_id,
                    "uncertainty_curve_id": line.uncertainty_curve_id,
                    "scale": scale,
                    "value": float(value),
                    "uncertainty": (
                        None
                        if uncertainty_values is None
                        else float(uncertainty_values[point_index])
                    ),
                })
        return pd.DataFrame(rows, columns=PLOT_DATA_COLUMNS)


class PlotDataPreparer:
    """Resolve figure queries and enforce exact uncertainty alignment."""

    def prepare(
        self,
        results: MetricResultTable,
        figure_spec: FigureSpec,
    ) -> PlotData:
        """Select every requested curve without rendering or writing files."""

        if not isinstance(results, MetricResultTable):
            raise TypeError("results must be a MetricResultTable")
        if not isinstance(figure_spec, FigureSpec):
            raise TypeError("figure_spec must be a FigureSpec")

        prepared_lines = tuple(
            self._prepare_line(results, line_spec)
            for line_spec in figure_spec.lines
        )
        return PlotData(figure_spec=figure_spec, lines=prepared_lines)

    @staticmethod
    def _prepare_line(
        results: MetricResultTable,
        line_spec: LineSpec,
    ) -> PlotLineData:
        curve = line_spec.curve_query.select(results)
        curve_id = curve.curve_ids[0]
        if line_spec.uncertainty_query is None:
            return PlotLineData(
                line_spec=line_spec,
                curve_id=curve_id,
                scales=tuple(row.scale for row in curve.rows),
                values=tuple(float(row.value) for row in curve.rows),
            )

        uncertainty_curve = line_spec.uncertainty_query.select(results)
        aligned = CurveAligner("exact").align(curve, uncertainty_curve)
        return PlotLineData(
            line_spec=line_spec,
            curve_id=aligned.reference_curve_id,
            scales=aligned.scales,
            values=aligned.reference_values,
            uncertainty_curve_id=aligned.target_curve_id,
            uncertainty_values=aligned.target_values,
        )


@runtime_checkable
class PlotDataProvider(Protocol):
    """Prepare plot data from a metric result store and figure request."""

    def prepare(
        self,
        results: MetricResultTable,
        figure_spec: FigureSpec,
    ) -> PlotData:
        """Return validated observations in figure line order."""


@runtime_checkable
class PlotBackend(Protocol):
    """Render already prepared plot data without selecting curves."""

    name: str

    def config(self) -> Mapping[str, Any]:
        """Return the effective rendering configuration."""

    def render(self, plot_data: PlotData) -> Path:
        """Render one prepared figure and return its final output path."""


class MatplotlibBackend:
    """Render PlotData with a headless Matplotlib figure canvas."""

    name = "matplotlib"

    def __init__(
        self,
        *,
        figure_size: tuple[float, float] = (7.0, 4.5),
        dpi: int = 150,
        line_width: float = 1.5,
        uncertainty_alpha: float = 0.2,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(figure_size, tuple) or len(figure_size) != 2:
            raise TypeError("figure_size must be a two-value tuple")
        width, height = figure_size
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (width, height)
        ):
            raise ValueError("figure_size values must be finite and positive")
        if isinstance(dpi, bool) or not isinstance(dpi, int):
            raise TypeError("dpi must be an integer")
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        if (
            isinstance(line_width, bool)
            or not isinstance(line_width, (int, float))
            or not math.isfinite(float(line_width))
            or float(line_width) <= 0.0
        ):
            raise ValueError("line_width must be finite and positive")
        if (
            isinstance(uncertainty_alpha, bool)
            or not isinstance(uncertainty_alpha, (int, float))
            or not math.isfinite(float(uncertainty_alpha))
            or not 0.0 <= float(uncertainty_alpha) <= 1.0
        ):
            raise ValueError("uncertainty_alpha must be between zero and one")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        self.figure_size = (float(width), float(height))
        self.dpi = dpi
        self.line_width = float(line_width)
        self.uncertainty_alpha = float(uncertainty_alpha)
        self.overwrite = overwrite

    def config(self) -> Mapping[str, Any]:
        """Return deterministic rendering parameters."""

        return {
            "schema_version": 1,
            "figure_size": list(self.figure_size),
            "dpi": self.dpi,
            "line_width": self.line_width,
            "uncertainty_alpha": self.uncertainty_alpha,
            "overwrite": self.overwrite,
        }

    def render(self, plot_data: PlotData) -> Path:
        """Render and atomically commit one figure to its requested path."""

        if not isinstance(plot_data, PlotData):
            raise TypeError("plot_data must be PlotData")
        destination = plot_data.figure_spec.output_path
        if destination.exists() and not self.overwrite:
            raise FileExistsError(f"Figure already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=self.figure_size, dpi=self.dpi)
        FigureCanvasAgg(figure)
        primary_axis = figure.add_subplot(1, 1, 1)
        secondary_axis = None
        if plot_data.figure_spec.secondary_y_axis is not None:
            secondary_axis = primary_axis.twinx()

        temporary_path = self._temporary_path(destination)
        try:
            for line_data in plot_data.lines:
                axis = (
                    primary_axis
                    if line_data.line_spec.axis == "primary"
                    else secondary_axis
                )
                if axis is None:
                    raise ValueError("secondary plot data requires a secondary axis")
                line_kwargs: dict[str, Any] = {
                    "label": line_data.line_spec.label,
                    "linewidth": self.line_width,
                }
                if line_data.line_spec.color is not None:
                    line_kwargs["color"] = line_data.line_spec.color
                if line_data.line_spec.marker is not None:
                    line_kwargs["marker"] = line_data.line_spec.marker
                plotted_line = axis.plot(
                    line_data.scales,
                    line_data.values,
                    **line_kwargs,
                )[0]
                if line_data.uncertainty_values is not None:
                    values = np.asarray(line_data.values, dtype=float)
                    uncertainty = np.asarray(
                        line_data.uncertainty_values,
                        dtype=float,
                    )
                    axis.fill_between(
                        line_data.scales,
                        values - uncertainty,
                        values + uncertainty,
                        color=plotted_line.get_color(),
                        alpha=self.uncertainty_alpha,
                    )

            self._configure_axes(primary_axis, secondary_axis, plot_data.figure_spec)
            figure.tight_layout()
            figure.savefig(temporary_path, dpi=self.dpi)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination

    @staticmethod
    def _temporary_path(destination: Path) -> Path:
        temporary_file = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
        return temporary_path

    @staticmethod
    def _configure_axes(primary_axis, secondary_axis, figure_spec: FigureSpec) -> None:
        primary_axis.set_xlabel(figure_spec.x_axis.label)
        primary_axis.set_ylabel(figure_spec.y_axis.label)
        primary_axis.set_xscale(figure_spec.x_scale)
        primary_axis.set_yscale(figure_spec.y_scale)
        if figure_spec.x_axis.limits is not None:
            primary_axis.set_xlim(*figure_spec.x_axis.limits)
        if figure_spec.y_axis.limits is not None:
            primary_axis.set_ylim(*figure_spec.y_axis.limits)
        if figure_spec.title is not None:
            primary_axis.set_title(figure_spec.title)

        axes = [primary_axis]
        if secondary_axis is not None:
            secondary_spec = figure_spec.secondary_y_axis
            secondary_axis.set_ylabel(secondary_spec.label)
            secondary_axis.set_yscale(figure_spec.y_scale)
            if secondary_spec.limits is not None:
                secondary_axis.set_ylim(*secondary_spec.limits)
            axes.append(secondary_axis)

        if figure_spec.legend.visible:
            handles = []
            labels = []
            for axis in axes:
                axis_handles, axis_labels = axis.get_legend_handles_labels()
                handles.extend(axis_handles)
                labels.extend(axis_labels)
            primary_axis.legend(
                handles,
                labels,
                loc=figure_spec.legend.location,
                title=figure_spec.legend.title,
                ncols=figure_spec.legend.columns,
            )


@dataclass(frozen=True)
class PlotPipelineRun:
    """Return prepared data, committed outputs, and rendering provenance."""

    plot_data: PlotData
    figure_path: Path
    data_path: Path
    backend_name: str
    config_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.plot_data, PlotData):
            raise TypeError("plot_data must be PlotData")
        for field_name in ("figure_path", "data_path"):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise TypeError(f"{field_name} must be a Path")
            if not value.is_file():
                raise ValueError(f"{field_name} must identify a committed file")
        if self.figure_path != self.plot_data.figure_spec.output_path:
            raise ValueError("figure_path must match the requested output path")
        if not isinstance(self.backend_name, str) or not self.backend_name.strip():
            raise ValueError("backend_name must be a non-empty string")
        if (
            not isinstance(self.config_hash, str)
            or len(self.config_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.config_hash)
        ):
            raise ValueError("config_hash must be a lowercase SHA-256 hex digest")

    @property
    def plotting_frame(self) -> pd.DataFrame:
        """Return the exact long-form data committed beside the figure."""

        return self.plot_data.to_frame()


class PlotPipeline:
    """Coordinate curve preparation, rendering, and plot-data provenance."""

    def __init__(
        self,
        *,
        backend: PlotBackend,
        preparer: PlotDataProvider | None = None,
    ) -> None:
        if not isinstance(backend, PlotBackend):
            raise TypeError("backend must satisfy PlotBackend")
        if not isinstance(backend.name, str) or not backend.name.strip():
            raise ValueError("backend name must be a non-empty string")
        selected_preparer = PlotDataPreparer() if preparer is None else preparer
        if not isinstance(selected_preparer, PlotDataProvider):
            raise TypeError("preparer must satisfy PlotDataProvider")
        self.backend = backend
        self.preparer = selected_preparer

    @staticmethod
    def data_path_for(figure_path: str | Path) -> Path:
        """Return the canonical sibling path for actual plotting data."""

        path = Path(figure_path)
        return path.with_name(f"{path.stem}.plot_data.csv")

    def run(
        self,
        results: MetricResultTable,
        figure_spec: FigureSpec,
    ) -> PlotPipelineRun:
        """Prepare, render, and commit one figure with its exact input table."""

        if not isinstance(results, MetricResultTable):
            raise TypeError("results must be a MetricResultTable")
        if not isinstance(figure_spec, FigureSpec):
            raise TypeError("figure_spec must be a FigureSpec")
        figure_path = figure_spec.output_path
        data_path = self.data_path_for(figure_path)
        existing = tuple(
            path for path in (figure_path, data_path) if path.exists()
        )
        if existing:
            if len(existing) == 1:
                raise FileExistsError(
                    f"Incomplete plot output exists: {existing[0]}"
                )
            raise FileExistsError(
                f"Plot outputs already exist: {figure_path}, {data_path}"
            )

        raw_backend_config = self.backend.config()
        if not isinstance(raw_backend_config, Mapping):
            raise TypeError("backend config must be a mapping")
        backend_config = dict(raw_backend_config)
        config_hash = stable_config_hash({
            "schema_version": 1,
            "figure": figure_spec.to_dict(),
            "backend": {
                "name": self.backend.name,
                "parameters": backend_config,
            },
        })
        plot_data = self.preparer.prepare(results, figure_spec)
        plotting_frame = plot_data.to_frame()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_data_path = self._temporary_data_path(data_path)
        try:
            plotting_frame.to_csv(
                temporary_data_path,
                index=False,
                encoding="utf-8",
            )
            rendered_path = Path(self.backend.render(plot_data))
            if rendered_path != figure_path:
                raise ValueError(
                    "backend returned a path different from the requested output path"
                )
            if not rendered_path.is_file():
                raise ValueError("backend did not commit the requested figure file")
            os.replace(temporary_data_path, data_path)
        except Exception:
            temporary_data_path.unlink(missing_ok=True)
            raise

        return PlotPipelineRun(
            plot_data=plot_data,
            figure_path=figure_path,
            data_path=data_path,
            backend_name=self.backend.name,
            config_hash=config_hash,
        )

    @staticmethod
    def _temporary_data_path(destination: Path) -> Path:
        temporary_file = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
        return temporary_path
