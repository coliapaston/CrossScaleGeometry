from pathlib import Path
import unittest
import tempfile

import pandas as pd

from cluster_anlys.metric_pipeline import MetricResultRow, MetricResultTable
from cluster_anlys.pipeline_core import ScaleTask
from cluster_anlys.plotting_pipeline import (
    AxisSpec,
    FigureSpec,
    LegendSpec,
    LineSpec,
    MatplotlibBackend,
    PLOT_AXES,
    PLOT_DATA_COLUMNS,
    PLOT_SCALES,
    PlotData,
    PlotDataPreparer,
    PlotDataProvider,
    PlotBackend,
    PlotLineData,
    PlotPipeline,
    PlotPipelineRun,
)
from cluster_anlys.statistics_pipeline import CurveQuery


def make_query(**overrides):
    values = {
        "analysis_id": "analysis_a",
        "model_name": "model_a",
        "source": "main_pca",
        "metric": "orthographic_similarity",
        "component": "score",
        "aggregation": "mean",
        "space_name": "output_proj",
        "partition_method": "hdbscan",
    }
    values.update(overrides)
    return CurveQuery(**values)


def make_line(label="O-S", **overrides):
    values = {
        "curve_query": make_query(),
        "label": label,
        "color": "tab:blue",
        "marker": "o",
    }
    values.update(overrides)
    return LineSpec(**values)


def make_figure(lines=None, **overrides):
    values = {
        "lines": (make_line(),) if lines is None else lines,
        "x_axis": AxisSpec("PCA dimension"),
        "y_axis": AxisSpec("Metric value"),
        "output_path": Path("figures/curve.png"),
    }
    values.update(overrides)
    return FigureSpec(**values)


def make_metric_row(scale=10, **overrides):
    task_values = {
        "analysis_id": "analysis_a",
        "model_name": "model_a",
        "space_name": "output_proj",
        "scale": scale,
        "source": "main_pca",
        "seed_type": None,
        "seed": None,
    }
    task_values.update(overrides.pop("task_overrides", {}))
    values = {
        "task": ScaleTask(**task_values),
        "partition_method": "hdbscan",
        "metric": "orthographic_similarity",
        "component": "score",
        "aggregation": "mean",
        "value": scale / 100.0,
        "scope": "partition",
        "config_hash": "a" * 64,
        "reused": False,
    }
    values.update(overrides)
    return MetricResultRow(**values)


class AxisSpecTests(unittest.TestCase):
    def test_normalizes_numeric_limits(self):
        spec = AxisSpec("Dimension", (1, 100.5))
        self.assertEqual(spec.limits, (1.0, 100.5))

    def test_rejects_invalid_label_and_limits(self):
        with self.assertRaises(ValueError):
            AxisSpec(" ")
        with self.assertRaises(ValueError):
            AxisSpec("Dimension", (10.0, 10.0))
        with self.assertRaises(TypeError):
            AxisSpec("Dimension", [1.0, 2.0])


class LegendSpecTests(unittest.TestCase):
    def test_accepts_backend_independent_configuration(self):
        spec = LegendSpec(visible=True, location="upper left", title="Series", columns=2)
        self.assertEqual(spec.columns, 2)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(TypeError):
            LegendSpec(visible=1)
        with self.assertRaises(ValueError):
            LegendSpec(location="")
        with self.assertRaises(ValueError):
            LegendSpec(columns=0)


class LineSpecTests(unittest.TestCase):
    def test_accepts_distinct_uncertainty_query(self):
        uncertainty = make_query(component="standard_deviation")
        spec = make_line(uncertainty_query=uncertainty, axis="secondary")
        self.assertEqual(spec.uncertainty_query, uncertainty)
        self.assertEqual(spec.axis, "secondary")

    def test_rejects_invalid_query_style_and_axis(self):
        with self.assertRaises(TypeError):
            make_line(curve_query="not-a-query")
        with self.assertRaises(ValueError):
            make_line(color="")
        with self.assertRaises(ValueError):
            make_line(axis="tertiary")

    def test_rejects_main_curve_as_its_own_uncertainty(self):
        query = make_query()
        with self.assertRaises(ValueError):
            make_line(curve_query=query, uncertainty_query=query)


class FigureSpecTests(unittest.TestCase):
    def test_accepts_any_three_explicit_lines(self):
        lines = (
            make_line("O-S"),
            make_line(
                "MS-E",
                curve_query=make_query(metric="script_entropy"),
                color="tab:orange",
            ),
            make_line(
                "Cluster count",
                curve_query=make_query(metric="cluster_count"),
                color="tab:green",
                axis="secondary",
            ),
        )
        spec = make_figure(
            lines,
            title="Selected curves",
            x_scale="log",
            secondary_y_axis=AxisSpec("Cluster count"),
        )
        self.assertEqual(tuple(line.label for line in spec.lines), (
            "O-S",
            "MS-E",
            "Cluster count",
        ))
        self.assertEqual(spec.output_path, Path("figures/curve.png"))

    def test_rejects_empty_lines_and_duplicate_labels(self):
        with self.assertRaises(ValueError):
            make_figure(lines=())
        with self.assertRaises(ValueError):
            make_figure(lines=(make_line(), make_line()))

    def test_requires_axis_configuration_for_secondary_lines(self):
        with self.assertRaises(ValueError):
            make_figure(lines=(make_line(axis="secondary"),))

    def test_rejects_invalid_scale_and_output_path(self):
        self.assertEqual(PLOT_AXES, ("primary", "secondary"))
        self.assertIn("log", PLOT_SCALES)
        with self.assertRaises(ValueError):
            make_figure(x_scale="categorical")
        with self.assertRaises(ValueError):
            make_figure(output_path="figures/no_suffix")


class PlotLineDataTests(unittest.TestCase):
    def test_validates_values_and_uncertainty(self):
        line = make_line(uncertainty_query=make_query(component="standard_deviation"))
        data = PlotLineData(
            line_spec=line,
            curve_id="curve_a",
            scales=(10, 20),
            values=(0.1, 0.2),
            uncertainty_curve_id="curve_b",
            uncertainty_values=(0.01, 0.02),
        )
        self.assertEqual(data.uncertainty_values, (0.01, 0.02))
        with self.assertRaises(ValueError):
            PlotLineData(
                line_spec=line,
                curve_id="curve_a",
                scales=(20, 10),
                values=(0.2, 0.1),
            )
        with self.assertRaises(ValueError):
            PlotLineData(
                line_spec=line,
                curve_id="curve_a",
                scales=(10, 20),
                values=(0.1, 0.2),
                uncertainty_curve_id="curve_b",
                uncertainty_values=(0.01, -0.02),
            )


class PlotDataPreparerTests(unittest.TestCase):
    def test_selects_any_three_lines_and_returns_exact_plotting_table(self):
        rows = []
        for scale in (10, 20, 40):
            rows.extend((
                make_metric_row(scale=scale, value=scale / 100.0),
                make_metric_row(
                    scale=scale,
                    component="standard_deviation",
                    value=scale / 1000.0,
                ),
                make_metric_row(
                    scale=scale,
                    metric="script_entropy",
                    component="normalized",
                    value=scale / 200.0,
                ),
                make_metric_row(
                    scale=scale,
                    metric="cluster_count",
                    component="count",
                    aggregation="count",
                    value=float(scale),
                ),
            ))
        lines = (
            make_line(
                "O-S",
                uncertainty_query=make_query(component="standard_deviation"),
            ),
            make_line(
                "MS-E",
                curve_query=make_query(
                    metric="script_entropy",
                    component="normalized",
                ),
                color="tab:orange",
            ),
            make_line(
                "Cluster count",
                curve_query=make_query(
                    metric="cluster_count",
                    component="count",
                    aggregation="count",
                ),
                axis="secondary",
                color="tab:green",
            ),
        )
        figure = make_figure(
            lines,
            secondary_y_axis=AxisSpec("Cluster count"),
        )
        prepared = PlotDataPreparer().prepare(MetricResultTable(tuple(rows)), figure)

        self.assertIsInstance(prepared, PlotData)
        self.assertEqual(tuple(line.line_spec.label for line in prepared.lines), (
            "O-S",
            "MS-E",
            "Cluster count",
        ))
        self.assertEqual(prepared.lines[0].scales, (10, 20, 40))
        self.assertEqual(prepared.lines[0].values, (0.1, 0.2, 0.4))
        self.assertEqual(prepared.lines[0].uncertainty_values, (0.01, 0.02, 0.04))
        frame = prepared.to_frame()
        self.assertEqual(tuple(frame.columns), PLOT_DATA_COLUMNS)
        self.assertEqual(len(frame), 9)
        pd.testing.assert_series_equal(
            frame.loc[frame["label"] == "Cluster count", "value"].reset_index(drop=True),
            pd.Series([10.0, 20.0, 40.0], name="value"),
        )

    def test_rejects_missing_main_curve(self):
        results = MetricResultTable((make_metric_row(scale=10),))
        line = make_line(curve_query=make_query(metric="missing_metric"))
        with self.assertRaises(ValueError):
            PlotDataPreparer().prepare(results, make_figure(lines=(line,)))

    def test_rejects_misaligned_uncertainty_scales(self):
        results = MetricResultTable((
            make_metric_row(scale=10),
            make_metric_row(scale=20),
            make_metric_row(
                scale=10,
                component="standard_deviation",
                value=0.01,
            ),
        ))
        line = make_line(
            uncertainty_query=make_query(component="standard_deviation")
        )
        with self.assertRaises(ValueError):
            PlotDataPreparer().prepare(results, make_figure(lines=(line,)))

    def test_rejects_invalid_inputs_and_prepared_line_order(self):
        with self.assertRaises(TypeError):
            PlotDataPreparer().prepare("not-results", make_figure())
        first = make_line("First")
        second = make_line("Second", curve_query=make_query(metric="second"))
        figure = make_figure(lines=(first, second))
        first_data = PlotLineData(first, "first", (10,), (0.1,))
        second_data = PlotLineData(second, "second", (10,), (0.2,))
        with self.assertRaises(ValueError):
            PlotData(figure, (second_data, first_data))


class DummyPlotBackend:
    name = "dummy"

    def __init__(self):
        self.calls = []

    def config(self):
        return {"format": "none"}

    def render(self, plot_data):
        self.calls.append(plot_data)
        return plot_data.figure_spec.output_path


class PlotBackendTests(unittest.TestCase):
    def test_dummy_backend_satisfies_narrow_protocol(self):
        backend = DummyPlotBackend()
        figure = make_figure()
        plot_data = PlotData(
            figure,
            (PlotLineData(figure.lines[0], "curve", (10,), (0.1,)),),
        )
        self.assertIsInstance(backend, PlotBackend)
        self.assertEqual(backend.render(plot_data), figure.output_path)
        self.assertEqual(backend.calls, [plot_data])

    def test_matplotlib_backend_renders_atomic_png_with_all_features(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested" / "figure.png"
            uncertainty_query = make_query(component="standard_deviation")
            primary = make_line("Primary", uncertainty_query=uncertainty_query)
            secondary = make_line(
                "Secondary",
                curve_query=make_query(metric="secondary_metric"),
                axis="secondary",
                color="tab:orange",
            )
            figure = make_figure(
                lines=(primary, secondary),
                output_path=output_path,
                title="Rendered figure",
                x_scale="log",
                x_axis=AxisSpec("Dimension", (10, 100)),
                y_axis=AxisSpec("Primary value", (0.0, 1.0)),
                secondary_y_axis=AxisSpec("Secondary value", (0.0, 10.0)),
                legend=LegendSpec(location="upper left", columns=2),
            )
            plot_data = PlotData(figure, (
                PlotLineData(
                    primary,
                    "primary_curve",
                    (10, 20, 40),
                    (0.2, 0.4, 0.6),
                    "uncertainty_curve",
                    (0.02, 0.03, 0.04),
                ),
                PlotLineData(
                    secondary,
                    "secondary_curve",
                    (10, 20, 40),
                    (2.0, 4.0, 6.0),
                ),
            ))

            backend = MatplotlibBackend(dpi=80)
            rendered_path = backend.render(plot_data)

            self.assertEqual(rendered_path, output_path)
            self.assertTrue(rendered_path.is_file())
            self.assertGreater(rendered_path.stat().st_size, 1000)
            self.assertEqual(rendered_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(list(output_path.parent.glob(".figure.*.png")), [])

    def test_matplotlib_backend_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "figure.png"
            output_path.write_bytes(b"existing")
            figure = make_figure(output_path=output_path)
            plot_data = PlotData(
                figure,
                (PlotLineData(figure.lines[0], "curve", (10,), (0.1,)),),
            )
            with self.assertRaises(FileExistsError):
                MatplotlibBackend().render(plot_data)
            self.assertEqual(output_path.read_bytes(), b"existing")

    def test_matplotlib_backend_validates_configuration(self):
        with self.assertRaises(ValueError):
            MatplotlibBackend(figure_size=(0.0, 4.0))
        with self.assertRaises(ValueError):
            MatplotlibBackend(dpi=0)
        with self.assertRaises(ValueError):
            MatplotlibBackend(line_width=float("nan"))
        with self.assertRaises(ValueError):
            MatplotlibBackend(uncertainty_alpha=1.1)
        backend = MatplotlibBackend(overwrite=True)
        self.assertTrue(backend.config()["overwrite"])


class FileWritingDummyBackend:
    name = "file_writing_dummy"

    def __init__(self):
        self.calls = []

    def config(self):
        return {"payload": "dummy-figure"}

    def render(self, plot_data):
        self.calls.append(plot_data)
        path = plot_data.figure_spec.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy-figure")
        return path


class CountingPreparer:
    def __init__(self):
        self.calls = []
        self.delegate = PlotDataPreparer()

    def prepare(self, results, figure_spec):
        self.calls.append((results, figure_spec))
        return self.delegate.prepare(results, figure_spec)


class PlotPipelineTests(unittest.TestCase):
    @staticmethod
    def make_three_curve_results():
        rows = []
        for scale in (10, 20, 40):
            rows.extend((
                make_metric_row(scale=scale, value=scale / 100.0),
                make_metric_row(
                    scale=scale,
                    component="standard_deviation",
                    value=scale / 1000.0,
                ),
                make_metric_row(
                    scale=scale,
                    metric="script_entropy",
                    component="normalized",
                    value=scale / 200.0,
                ),
                make_metric_row(
                    scale=scale,
                    metric="cluster_count",
                    component="count",
                    aggregation="count",
                    value=float(scale),
                ),
            ))
        return MetricResultTable(tuple(rows))

    @staticmethod
    def make_three_curve_figure(output_path):
        lines = (
            make_line(
                "O-S",
                uncertainty_query=make_query(component="standard_deviation"),
            ),
            make_line(
                "MS-E",
                curve_query=make_query(
                    metric="script_entropy",
                    component="normalized",
                ),
                color="tab:orange",
            ),
            make_line(
                "Cluster count",
                curve_query=make_query(
                    metric="cluster_count",
                    component="count",
                    aggregation="count",
                ),
                axis="secondary",
                color="tab:green",
            ),
        )
        return make_figure(
            lines=lines,
            output_path=output_path,
            secondary_y_axis=AxisSpec("Cluster count"),
        )

    def test_three_curve_end_to_end_with_replaceable_components(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = Path(temporary_directory) / "three_curves.bin"
            figure = self.make_three_curve_figure(figure_path)
            results = self.make_three_curve_results()
            backend = FileWritingDummyBackend()
            preparer = CountingPreparer()

            self.assertIsInstance(preparer, PlotDataProvider)
            run = PlotPipeline(backend=backend, preparer=preparer).run(
                results,
                figure,
            )

            self.assertIsInstance(run, PlotPipelineRun)
            self.assertEqual(run.figure_path.read_bytes(), b"dummy-figure")
            self.assertEqual(run.data_path, figure_path.with_name(
                "three_curves.plot_data.csv"
            ))
            self.assertEqual(len(preparer.calls), 1)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(run.backend_name, backend.name)
            self.assertEqual(len(run.config_hash), 64)
            committed = pd.read_csv(run.data_path)
            self.assertEqual(tuple(committed.columns), PLOT_DATA_COLUMNS)
            self.assertEqual(len(committed), 9)
            self.assertEqual(tuple(committed["label"].drop_duplicates()), (
                "O-S",
                "MS-E",
                "Cluster count",
            ))
            expected = run.plotting_frame
            committed["uncertainty_curve_id"] = committed[
                "uncertainty_curve_id"
            ].astype("string")
            expected["uncertainty_curve_id"] = expected[
                "uncertainty_curve_id"
            ].astype("string")
            pd.testing.assert_frame_equal(
                committed,
                expected,
                check_dtype=False,
            )

    def test_matplotlib_pipeline_commits_figure_and_sibling_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = Path(temporary_directory) / "figure.png"
            figure = make_figure(output_path=figure_path)
            results = MetricResultTable((
                make_metric_row(scale=10, value=0.1),
                make_metric_row(scale=20, value=0.2),
            ))
            run = PlotPipeline(backend=MatplotlibBackend(dpi=80)).run(
                results,
                figure,
            )
            self.assertEqual(run.figure_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(run.data_path.is_file())

    def test_preflight_refuses_complete_and_incomplete_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = Path(temporary_directory) / "figure.bin"
            figure = make_figure(output_path=figure_path)
            results = MetricResultTable((make_metric_row(scale=10),))
            backend = FileWritingDummyBackend()
            pipeline = PlotPipeline(backend=backend)
            data_path = pipeline.data_path_for(figure_path)

            figure_path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                pipeline.run(results, figure)
            self.assertEqual(backend.calls, [])

            data_path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                pipeline.run(results, figure)
            self.assertEqual(backend.calls, [])

    def test_rejects_backend_that_does_not_commit_requested_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            figure_path = Path(temporary_directory) / "figure.bin"
            figure = make_figure(output_path=figure_path)
            results = MetricResultTable((make_metric_row(scale=10),))
            backend = DummyPlotBackend()
            with self.assertRaises(ValueError):
                PlotPipeline(backend=backend).run(results, figure)
            self.assertFalse(PlotPipeline.data_path_for(figure_path).exists())


if __name__ == "__main__":
    unittest.main()
