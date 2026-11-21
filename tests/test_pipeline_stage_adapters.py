from pathlib import Path
import tempfile
import unittest

from cluster_anlys.clustering_pipeline import (
    ClusteringPipelineRun,
    PartitionManifest,
)
from cluster_anlys.experiment_orchestrator import (
    DataProduct,
    DataProductSpec,
    DataProductStore,
    ExperimentOrchestrator,
    ExperimentStage,
    StageRegistry,
)
from cluster_anlys.pipeline_core import ScaleTask, StageRunReport
from cluster_anlys.metric_pipeline import (
    MetricResultRow,
    MetricResultTable,
    PartitionMetricPipelineRun,
)
from cluster_anlys.pipeline_stage_adapters import (
    ClusteringPipelineRunner,
    PartitionMetricPipelineRunner,
    ComposedStatisticsPipelineRunner,
    PlotPipelineRunner,
    StatisticsPipelineRunner,
    make_clustering_stage,
    make_composed_statistics_stage,
    make_metric_table_combine_stage,
    make_partition_metric_stage,
    make_plot_stage,
    make_statistics_stage,
)
from cluster_anlys.statistics_pipeline import (
    ComposedStatisticsPipeline,
    ComposedStatisticsPipelineRun,
    CurveQuery,
    CurveTable,
    MAEStatistic,
    StatisticResultTable,
    StatisticsPipeline,
    StatisticsPipelineRun,
    ZScoreConcatenationComposer,
)
from cluster_anlys.plotting_pipeline import (
    AxisSpec,
    FigureSpec,
    LineSpec,
    PlotPipeline,
)


def make_task(scale=2, **overrides):
    values = dict(
        analysis_id="adapter-test",
        model_name="model",
        space_name="space",
        scale=scale,
        source="synthetic",
    )
    values.update(overrides)
    return ScaleTask(**values)


class FakeClusteringPipeline:
    def __init__(self, run):
        self.pipeline_run = run
        self.calls = []

    def run(self, tasks):
        self.calls.append(tasks)
        return self.pipeline_run


class FakePartitionMetricPipeline:
    def __init__(self, run):
        self.pipeline_run = run
        self.calls = []

    def run(self, manifest):
        self.calls.append(manifest)
        return self.pipeline_run


class FakeStatisticsPipeline:
    def __init__(self, run):
        self.pipeline_run = run
        self.calls = []

    def run(self, metric_results, queries):
        self.calls.append((metric_results, queries))
        return self.pipeline_run


class FakeComposedStatisticsPipeline:
    def __init__(self, run):
        self.pipeline_run = run
        self.calls = []

    def run(self, metric_results, *, reference_queries, target_queries):
        self.calls.append((metric_results, reference_queries, target_queries))
        return self.pipeline_run


class FakePlotPipeline:
    def __init__(self, run):
        self.pipeline_run = run
        self.calls = []

    def run(self, metric_results, figure_spec):
        self.calls.append((metric_results, figure_spec))
        return self.pipeline_run


class FilePlotBackend:
    name = "file-backend"

    def config(self):
        return {"format": "binary"}

    def render(self, plot_data):
        destination = plot_data.figure_spec.output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"plot")
        return destination


class ClusteringStageAdapterTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline_run = ClusteringPipelineRun(
            manifest=PartitionManifest(),
            report=StageRunReport(stage="clustering", records=()),
            manifest_path=Path("partition_manifest.csv"),
        )
        return FakeClusteringPipeline(pipeline_run), pipeline_run

    def test_binds_pipeline_and_exposes_manifest_product(self):
        pipeline, pipeline_run = self.make_pipeline()
        tasks = (make_task(),)
        stage = make_clustering_stage(
            stage_id="clustering",
            pipeline=pipeline,
            tasks=tasks,
        )
        execution = stage.execute(DataProductStore())
        self.assertIsInstance(pipeline, ClusteringPipelineRunner)
        self.assertIsInstance(stage, ExperimentStage)
        self.assertEqual(pipeline.calls, [tasks])
        self.assertIs(execution.products[0].value, pipeline_run.manifest)
        self.assertIs(execution.native_report, pipeline_run.report)

    def test_runs_through_orchestrator_with_native_report(self):
        pipeline, pipeline_run = self.make_pipeline()
        tasks = (make_task(), make_task(3))
        stage = make_clustering_stage(
            stage_id="low-dim-clustering",
            pipeline=pipeline,
            tasks=tasks,
            product_key="partitions.low-dim",
        )
        report = ExperimentOrchestrator(StageRegistry((stage,))).run(
            ("low-dim-clustering",)
        )
        self.assertTrue(report.succeeded)
        self.assertEqual(pipeline.calls, [tasks])
        product = report.final_products.product("partitions.low-dim")
        self.assertIs(product.value, pipeline_run.manifest)
        self.assertEqual(product.producer_stage, "low-dim-clustering")
        self.assertIs(report.records[0].native_report, pipeline_run.report)

    def test_rejects_invalid_runner_tasks_and_pipeline_result(self):
        pipeline, _ = self.make_pipeline()
        with self.assertRaises(TypeError):
            make_clustering_stage(
                stage_id="clustering",
                pipeline=object(),
                tasks=(make_task(),),
            )
        with self.assertRaises(ValueError):
            make_clustering_stage(
                stage_id="clustering",
                pipeline=pipeline,
                tasks=(),
            )
        with self.assertRaises(ValueError):
            make_clustering_stage(
                stage_id="clustering",
                pipeline=pipeline,
                tasks=(make_task(), make_task()),
            )

        bad_pipeline = FakeClusteringPipeline("not-a-run")
        stage = make_clustering_stage(
            stage_id="clustering",
            pipeline=bad_pipeline,
            tasks=(make_task(),),
        )
        report = ExperimentOrchestrator(StageRegistry((stage,))).run(
            ("clustering",)
        )
        self.assertFalse(report.succeeded)
        self.assertEqual(report.final_products.keys, ())
        self.assertIn("TypeError", report.records[0].error_message)


class PartitionMetricStageAdapterTests(unittest.TestCase):
    def make_pipeline(self, table=None):
        results = table or MetricResultTable()
        pipeline_run = PartitionMetricPipelineRun(
            results=results,
            report=StageRunReport(stage="partition_metric", records=()),
            result_table_path=Path("metric_results.csv"),
        )
        return FakePartitionMetricPipeline(pipeline_run), pipeline_run

    def test_maps_manifest_input_to_metric_result_product(self):
        pipeline, pipeline_run = self.make_pipeline()
        stage = make_partition_metric_stage(
            stage_id="os-stage",
            pipeline=pipeline,
            dependencies=(),
            manifest_key="partitions.external",
            result_key="metrics.os",
        )
        manifest = PartitionManifest()
        store = DataProductStore((DataProduct(
            stage.required_products[0],
            manifest,
        ),))
        execution = stage.execute(store)
        self.assertIsInstance(pipeline, PartitionMetricPipelineRunner)
        self.assertEqual(pipeline.calls, [manifest])
        self.assertIs(execution.products[0].value, pipeline_run.results)
        self.assertIs(execution.native_report, pipeline_run.report)

    def test_sibling_metric_stages_share_manifest_without_overwrite(self):
        clustering_pipeline, clustering_run = (
            ClusteringStageAdapterTests().make_pipeline()
        )
        clustering_stage = make_clustering_stage(
            stage_id="clustering",
            pipeline=clustering_pipeline,
            tasks=(make_task(),),
        )
        os_pipeline, os_run = self.make_pipeline()
        mse_pipeline, mse_run = self.make_pipeline()
        os_stage = make_partition_metric_stage(
            stage_id="os-stage",
            pipeline=os_pipeline,
            dependencies=("clustering",),
            result_key="metrics.os",
        )
        mse_stage = make_partition_metric_stage(
            stage_id="mse-stage",
            pipeline=mse_pipeline,
            dependencies=("clustering",),
            result_key="metrics.mse",
        )
        report = ExperimentOrchestrator(StageRegistry((
            clustering_stage,
            os_stage,
            mse_stage,
        ))).run(("os-stage", "mse-stage"))
        self.assertTrue(report.succeeded)
        self.assertEqual(report.final_products.keys, (
            "partitions.main",
            "metrics.os",
            "metrics.mse",
        ))
        self.assertEqual(os_pipeline.calls, [clustering_run.manifest])
        self.assertEqual(mse_pipeline.calls, [clustering_run.manifest])
        self.assertIs(
            report.final_products.product("metrics.os").value,
            os_run.results,
        )
        self.assertIs(
            report.final_products.product("metrics.mse").value,
            mse_run.results,
        )

    def test_rejects_invalid_runner_and_wrong_pipeline_result(self):
        with self.assertRaises(TypeError):
            make_partition_metric_stage(
                stage_id="os-stage",
                pipeline=object(),
                dependencies=(),
            )
        bad_pipeline = FakePartitionMetricPipeline("not-a-run")
        stage = make_partition_metric_stage(
            stage_id="os-stage",
            pipeline=bad_pipeline,
            dependencies=(),
            manifest_key="partitions.external",
            result_key="metrics.os",
        )
        manifest = PartitionManifest()
        store = DataProductStore((DataProduct(
            stage.required_products[0],
            manifest,
        ),))
        with self.assertRaises(TypeError):
            stage.execute(store)


class MetricTableCombineStageTests(unittest.TestCase):
    def make_result_table(self, metric, value):
        return MetricResultTable((MetricResultRow(
            task=make_task(),
            partition_method="hdbscan",
            metric=metric,
            component="score",
            aggregation="mean",
            value=value,
            scope="partition",
            config_hash=f"{metric}-config",
        ),))

    def make_external_store(self, os_table, mse_table):
        return DataProductStore((
            DataProduct(
                DataProductSpec("metrics.os", MetricResultTable),
                os_table,
            ),
            DataProduct(
                DataProductSpec("metrics.mse", MetricResultTable),
                mse_table,
            ),
        ))

    def test_combines_named_inputs_in_declared_order(self):
        os_table = self.make_result_table("orthographic_similarity", 0.4)
        mse_table = self.make_result_table("multi_script_entropy", 0.2)
        stage = make_metric_table_combine_stage(
            stage_id="combine-metrics",
            dependencies=(),
            input_keys=("metrics.mse", "metrics.os"),
        )
        store = self.make_external_store(os_table, mse_table)
        execution = stage.execute(store)
        combined = execution.products[0].value
        self.assertEqual(
            tuple(row.metric for row in combined.rows),
            ("multi_script_entropy", "orthographic_similarity"),
        )
        self.assertIsNone(execution.native_report)

    def test_runs_after_sibling_metric_stages(self):
        clustering_pipeline, _ = ClusteringStageAdapterTests().make_pipeline()
        clustering_stage = make_clustering_stage(
            stage_id="clustering",
            pipeline=clustering_pipeline,
            tasks=(make_task(),),
        )
        os_table = self.make_result_table("orthographic_similarity", 0.4)
        mse_table = self.make_result_table("multi_script_entropy", 0.2)
        os_pipeline, _ = PartitionMetricStageAdapterTests().make_pipeline(os_table)
        mse_pipeline, _ = PartitionMetricStageAdapterTests().make_pipeline(mse_table)
        os_stage = make_partition_metric_stage(
            stage_id="os-stage",
            pipeline=os_pipeline,
            dependencies=("clustering",),
            result_key="metrics.os",
        )
        mse_stage = make_partition_metric_stage(
            stage_id="mse-stage",
            pipeline=mse_pipeline,
            dependencies=("clustering",),
            result_key="metrics.mse",
        )
        combine_stage = make_metric_table_combine_stage(
            stage_id="combine-metrics",
            dependencies=("os-stage", "mse-stage"),
            input_keys=("metrics.os", "metrics.mse"),
        )
        report = ExperimentOrchestrator(StageRegistry((
            clustering_stage,
            os_stage,
            mse_stage,
            combine_stage,
        ))).run(("combine-metrics",))
        self.assertTrue(report.succeeded)
        combined = report.final_products.product("metrics.combined").value
        self.assertEqual(combined.rows, os_table.rows + mse_table.rows)

    def test_duplicate_or_missing_input_fails_without_combined_product(self):
        duplicate = self.make_result_table("orthographic_similarity", 0.4)
        stage = make_metric_table_combine_stage(
            stage_id="combine-metrics",
            dependencies=(),
            input_keys=("metrics.os", "metrics.copy"),
        )
        duplicate_store = DataProductStore((
            DataProduct(stage.required_products[0], duplicate),
            DataProduct(stage.required_products[1], duplicate),
        ))
        with self.assertRaises(ValueError):
            stage.execute(duplicate_store)
        with self.assertRaises(KeyError):
            stage.execute(DataProductStore((DataProduct(
                stage.required_products[0],
                duplicate,
            ),)))

    def test_rejects_insufficient_duplicate_or_overlapping_keys(self):
        with self.assertRaises(ValueError):
            make_metric_table_combine_stage(
                stage_id="combine-metrics",
                dependencies=(),
                input_keys=("metrics.os",),
            )
        with self.assertRaises(ValueError):
            make_metric_table_combine_stage(
                stage_id="combine-metrics",
                dependencies=(),
                input_keys=("metrics.os", "metrics.os"),
            )
        with self.assertRaises(ValueError):
            make_metric_table_combine_stage(
                stage_id="combine-metrics",
                dependencies=(),
                input_keys=("metrics.os", "metrics.mse"),
                output_key="metrics.os",
            )


class StatisticsStageAdapterTests(unittest.TestCase):
    def make_query(self, source):
        return CurveQuery(
            analysis_id="adapter-test",
            model_name="model",
            source=source,
            metric="orthographic_similarity",
            component="score",
            aggregation="mean",
        )

    def make_metric_table(self):
        rows = []
        for source, values in (
            ("baseline", (0.2, 0.4)),
            ("random", (0.3, 0.5)),
        ):
            for scale, value in zip((2, 3), values):
                rows.append(MetricResultRow(
                    task=make_task(scale, source=source),
                    partition_method="hdbscan",
                    metric="orthographic_similarity",
                    component="score",
                    aggregation="mean",
                    value=value,
                    scope="partition",
                    config_hash=f"{source}-config",
                ))
        return MetricResultTable(tuple(rows))

    def test_maps_run_curves_and_results_to_separate_products(self):
        pipeline_run = StatisticsPipelineRun(
            curves=CurveTable(),
            alignments=(),
            results=StatisticResultTable(),
        )
        pipeline = FakeStatisticsPipeline(pipeline_run)
        queries = (self.make_query("baseline"), self.make_query("random"))
        stage = make_statistics_stage(
            stage_id="statistics",
            pipeline=pipeline,
            dependencies=(),
            queries=queries,
            metric_key="metrics.external",
        )
        metric_results = self.make_metric_table()
        execution = stage.execute(DataProductStore((DataProduct(
            stage.required_products[0],
            metric_results,
        ),)))
        self.assertIsInstance(pipeline, StatisticsPipelineRunner)
        self.assertEqual(pipeline.calls, [(metric_results, queries)])
        self.assertIs(execution.products[0].value, pipeline_run.curves)
        self.assertIs(execution.products[1].value, pipeline_run.results)
        self.assertIsNone(execution.native_report)

    def test_real_statistics_pipeline_runs_through_orchestrator(self):
        metric_results = self.make_metric_table()
        queries = (self.make_query("baseline"), self.make_query("random"))
        stage = make_statistics_stage(
            stage_id="statistics",
            pipeline=StatisticsPipeline(statistics=(MAEStatistic(),)),
            dependencies=(),
            queries=queries,
            metric_key="metrics.external",
            curves_key="curves.comparison",
            results_key="statistics.mae",
        )
        registry = StageRegistry(
            (stage,),
            external_products=(stage.required_products[0],),
        )
        initial_store = DataProductStore((DataProduct(
            stage.required_products[0],
            metric_results,
        ),))
        report = ExperimentOrchestrator(registry).run(
            ("statistics",),
            initial_store,
        )
        self.assertTrue(report.succeeded)
        curves = report.final_products.product("curves.comparison").value
        results = report.final_products.product("statistics.mae").value
        self.assertEqual(len(curves.curve_ids), 2)
        self.assertEqual(len(results.rows), 1)
        self.assertEqual(results.rows[0].statistic, "mae")

    def test_rejects_invalid_runner_queries_and_wrong_return(self):
        queries = (self.make_query("baseline"), self.make_query("random"))
        with self.assertRaises(TypeError):
            make_statistics_stage(
                stage_id="statistics",
                pipeline=object(),
                dependencies=(),
                queries=queries,
            )
        pipeline_run = StatisticsPipelineRun(
            curves=CurveTable(),
            alignments=(),
            results=StatisticResultTable(),
        )
        pipeline = FakeStatisticsPipeline(pipeline_run)
        with self.assertRaises(ValueError):
            make_statistics_stage(
                stage_id="statistics",
                pipeline=pipeline,
                dependencies=(),
                queries=(queries[0],),
            )
        bad_pipeline = FakeStatisticsPipeline("not-a-run")
        stage = make_statistics_stage(
            stage_id="statistics",
            pipeline=bad_pipeline,
            dependencies=(),
            queries=queries,
            metric_key="metrics.external",
        )
        with self.assertRaises(TypeError):
            stage.execute(DataProductStore((DataProduct(
                stage.required_products[0],
                self.make_metric_table(),
            ),)))


class ComposedStatisticsStageAdapterTests(unittest.TestCase):
    def make_queries(self):
        reference = tuple(
            CurveQuery(
                analysis_id="adapter-test",
                model_name="model",
                source="baseline",
                metric=metric,
                component=component,
                aggregation="mean",
            )
            for metric, component in (
                ("orthographic_similarity", "score"),
                ("multi_script_entropy", "normalized"),
            )
        )
        target = tuple(
            CurveQuery(
                analysis_id="adapter-test",
                model_name="model",
                source="random",
                metric=metric,
                component=component,
                aggregation="mean",
            )
            for metric, component in (
                ("orthographic_similarity", "score"),
                ("multi_script_entropy", "normalized"),
            )
        )
        return reference, target

    def make_metric_table(self):
        rows = []
        for source, offset in (("baseline", 0.0), ("random", 0.1)):
            for metric, component, values in (
                ("orthographic_similarity", "score", (0.2, 0.5)),
                ("multi_script_entropy", "normalized", (0.3, 0.7)),
            ):
                for scale, value in zip((2, 3), values):
                    rows.append(MetricResultRow(
                        task=make_task(scale, source=source),
                        partition_method="hdbscan",
                        metric=metric,
                        component=component,
                        aggregation="mean",
                        value=value + offset,
                        scope="partition",
                        config_hash=f"{source}-{metric}-config",
                    ))
        return MetricResultTable(tuple(rows))

    def make_real_run(self):
        reference_queries, target_queries = self.make_queries()
        pipeline = ComposedStatisticsPipeline(
            composer=ZScoreConcatenationComposer(
                feature_slots=("orthographic", "script"),
            ),
            statistics=(MAEStatistic(),),
        )
        metric_results = self.make_metric_table()
        run = pipeline.run(
            metric_results,
            reference_queries=reference_queries,
            target_queries=target_queries,
        )
        return metric_results, reference_queries, target_queries, run

    def test_maps_all_four_public_outputs_without_copying(self):
        metric_results, reference_queries, target_queries, pipeline_run = (
            self.make_real_run()
        )
        pipeline = FakeComposedStatisticsPipeline(pipeline_run)
        stage = make_composed_statistics_stage(
            stage_id="composed-statistics",
            pipeline=pipeline,
            dependencies=(),
            reference_queries=reference_queries,
            target_queries=target_queries,
            metric_key="metrics.external",
        )
        execution = stage.execute(DataProductStore((DataProduct(
            stage.required_products[0],
            metric_results,
        ),)))
        self.assertIsInstance(pipeline, ComposedStatisticsPipelineRunner)
        self.assertEqual(
            pipeline.calls,
            [(metric_results, reference_queries, target_queries)],
        )
        expected = (
            pipeline_run.reference_curves,
            pipeline_run.target_curves,
            pipeline_run.composed_pair,
            pipeline_run.results,
        )
        self.assertTrue(all(
            product.value is value
            for product, value in zip(execution.products, expected)
        ))

    def test_real_zscore_pipeline_runs_through_orchestrator(self):
        metric_results, reference_queries, target_queries, _ = self.make_real_run()
        pipeline = ComposedStatisticsPipeline(
            composer=ZScoreConcatenationComposer(
                feature_slots=("orthographic", "script"),
            ),
            statistics=(MAEStatistic(),),
        )
        stage = make_composed_statistics_stage(
            stage_id="composed-statistics",
            pipeline=pipeline,
            dependencies=(),
            reference_queries=reference_queries,
            target_queries=target_queries,
            metric_key="metrics.external",
        )
        registry = StageRegistry(
            (stage,),
            external_products=(stage.required_products[0],),
        )
        report = ExperimentOrchestrator(registry).run(
            ("composed-statistics",),
            DataProductStore((DataProduct(
                stage.required_products[0],
                metric_results,
            ),)),
        )
        self.assertTrue(report.succeeded)
        pair = report.final_products.product("curves.composed-pair").value
        results = report.final_products.product("statistics.composed").value
        self.assertEqual(len(pair.observation_keys), 4)
        self.assertEqual(results.rows[0].n_points, 4)

    def test_rejects_mismatched_queries_and_wrong_return(self):
        metric_results, reference_queries, target_queries, pipeline_run = (
            self.make_real_run()
        )
        pipeline = FakeComposedStatisticsPipeline(pipeline_run)
        with self.assertRaises(ValueError):
            make_composed_statistics_stage(
                stage_id="composed-statistics",
                pipeline=pipeline,
                dependencies=(),
                reference_queries=reference_queries,
                target_queries=target_queries[:1],
            )
        bad_pipeline = FakeComposedStatisticsPipeline("not-a-run")
        stage = make_composed_statistics_stage(
            stage_id="composed-statistics",
            pipeline=bad_pipeline,
            dependencies=(),
            reference_queries=reference_queries,
            target_queries=target_queries,
            metric_key="metrics.external",
        )
        with self.assertRaises(TypeError):
            stage.execute(DataProductStore((DataProduct(
                stage.required_products[0],
                metric_results,
            ),)))


class PlotStageAdapterTests(unittest.TestCase):
    def make_inputs(self, output_path):
        query = CurveQuery(
            analysis_id="adapter-test",
            model_name="model",
            source="baseline",
            metric="orthographic_similarity",
            component="score",
            aggregation="mean",
        )
        figure_spec = FigureSpec(
            lines=(LineSpec(query, "O-S"),),
            x_axis=AxisSpec("Dimension"),
            y_axis=AxisSpec("Score"),
            output_path=output_path,
        )
        metric_results = MetricResultTable(tuple(
            MetricResultRow(
                task=make_task(scale, source="baseline"),
                partition_method="hdbscan",
                metric="orthographic_similarity",
                component="score",
                aggregation="mean",
                value=value,
                scope="partition",
                config_hash="plot-config",
            )
            for scale, value in ((2, 0.2), (3, 0.4))
        ))
        return metric_results, figure_spec

    def test_maps_complete_plot_run_as_one_terminal_product(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metric_results, figure_spec = self.make_inputs(
                Path(temporary_directory) / "source.bin"
            )
            pipeline_run = PlotPipeline(backend=FilePlotBackend()).run(
                metric_results,
                figure_spec,
            )
            pipeline = FakePlotPipeline(pipeline_run)
            stage = make_plot_stage(
                stage_id="plotting",
                pipeline=pipeline,
                dependencies=(),
                figure_spec=figure_spec,
                metric_key="metrics.external",
            )
            execution = stage.execute(DataProductStore((DataProduct(
                stage.required_products[0],
                metric_results,
            ),)))
            self.assertIsInstance(pipeline, PlotPipelineRunner)
            self.assertEqual(pipeline.calls, [(metric_results, figure_spec)])
            self.assertIs(execution.products[0].value, pipeline_run)
            self.assertIsNone(execution.native_report)

    def test_real_plot_pipeline_runs_through_orchestrator(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metric_results, figure_spec = self.make_inputs(
                Path(temporary_directory) / "orchestrated.bin"
            )
            stage = make_plot_stage(
                stage_id="plotting",
                pipeline=PlotPipeline(backend=FilePlotBackend()),
                dependencies=(),
                figure_spec=figure_spec,
                metric_key="metrics.external",
                plot_run_key="plots.curves",
            )
            registry = StageRegistry(
                (stage,),
                external_products=(stage.required_products[0],),
            )
            report = ExperimentOrchestrator(registry).run(
                ("plotting",),
                DataProductStore((DataProduct(
                    stage.required_products[0],
                    metric_results,
                ),)),
            )
            self.assertTrue(report.succeeded)
            run = report.final_products.product("plots.curves").value
            self.assertEqual(run.figure_path.read_bytes(), b"plot")
            self.assertTrue(run.data_path.is_file())

    def test_rejects_invalid_runner_wrong_return_and_mismatched_figure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metric_results, figure_spec = self.make_inputs(root / "first.bin")
            with self.assertRaises(TypeError):
                make_plot_stage(
                    stage_id="plotting",
                    pipeline=object(),
                    dependencies=(),
                    figure_spec=figure_spec,
                )

            bad_pipeline = FakePlotPipeline("not-a-run")
            stage = make_plot_stage(
                stage_id="plotting",
                pipeline=bad_pipeline,
                dependencies=(),
                figure_spec=figure_spec,
                metric_key="metrics.external",
            )
            store = DataProductStore((DataProduct(
                stage.required_products[0],
                metric_results,
            ),))
            with self.assertRaises(TypeError):
                stage.execute(store)

            _, other_figure = self.make_inputs(root / "other.bin")
            other_run = PlotPipeline(backend=FilePlotBackend()).run(
                metric_results,
                other_figure,
            )
            mismatched = make_plot_stage(
                stage_id="plotting",
                pipeline=FakePlotPipeline(other_run),
                dependencies=(),
                figure_spec=figure_spec,
                metric_key="metrics.external",
            )
            with self.assertRaises(ValueError):
                mismatched.execute(store)


class FullOrchestrationGraphTests(unittest.TestCase):
    def make_metric_table(self, metric, component):
        rows = []
        for source, offset in (("baseline", 0.0), ("random", 0.1)):
            for scale, value in ((2, 0.2), (3, 0.4)):
                rows.append(MetricResultRow(
                    task=make_task(scale, source=source),
                    partition_method="hdbscan",
                    metric=metric,
                    component=component,
                    aggregation="mean",
                    value=value + offset,
                    scope="partition",
                    config_hash=f"{metric}-{source}-config",
                ))
        return MetricResultTable(tuple(rows))

    def make_metric_pipeline(self, table):
        return FakePartitionMetricPipeline(PartitionMetricPipelineRun(
            results=table,
            report=StageRunReport(stage="partition_metric", records=()),
            result_table_path=Path("synthetic_metric_results.csv"),
        ))

    def test_full_branch_and_single_stage_request_use_correct_closures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            clustering_pipeline, _ = ClusteringStageAdapterTests().make_pipeline()
            clustering = make_clustering_stage(
                stage_id="clustering",
                pipeline=clustering_pipeline,
                tasks=(make_task(),),
            )
            os_pipeline = self.make_metric_pipeline(self.make_metric_table(
                "orthographic_similarity",
                "score",
            ))
            mse_pipeline = self.make_metric_pipeline(self.make_metric_table(
                "multi_script_entropy",
                "normalized",
            ))
            os_stage = make_partition_metric_stage(
                stage_id="os-stage",
                pipeline=os_pipeline,
                dependencies=("clustering",),
                result_key="metrics.os",
            )
            mse_stage = make_partition_metric_stage(
                stage_id="mse-stage",
                pipeline=mse_pipeline,
                dependencies=("clustering",),
                result_key="metrics.mse",
            )
            combine = make_metric_table_combine_stage(
                stage_id="combine-metrics",
                dependencies=("os-stage", "mse-stage"),
                input_keys=("metrics.os", "metrics.mse"),
            )
            baseline_query = CurveQuery(
                analysis_id="adapter-test",
                model_name="model",
                source="baseline",
                metric="orthographic_similarity",
                component="score",
                aggregation="mean",
            )
            random_query = CurveQuery(
                analysis_id="adapter-test",
                model_name="model",
                source="random",
                metric="orthographic_similarity",
                component="score",
                aggregation="mean",
            )
            statistics = make_statistics_stage(
                stage_id="statistics",
                pipeline=StatisticsPipeline(statistics=(MAEStatistic(),)),
                dependencies=("combine-metrics",),
                queries=(baseline_query, random_query),
            )
            figure_spec = FigureSpec(
                lines=(LineSpec(baseline_query, "Baseline O-S"),),
                x_axis=AxisSpec("Dimension"),
                y_axis=AxisSpec("Score"),
                output_path=Path(temporary_directory) / "full-graph.bin",
            )
            plotting = make_plot_stage(
                stage_id="plotting",
                pipeline=PlotPipeline(backend=FilePlotBackend()),
                dependencies=("combine-metrics",),
                figure_spec=figure_spec,
            )
            registry = StageRegistry((
                clustering,
                os_stage,
                mse_stage,
                combine,
                statistics,
                plotting,
            ))

            single_report = ExperimentOrchestrator(registry).run(("mse-stage",))
            self.assertEqual(single_report.execution_order, (
                "clustering",
                "mse-stage",
            ))
            self.assertEqual(single_report.final_products.keys, (
                "partitions.main",
                "metrics.mse",
            ))
            self.assertTrue(single_report.succeeded)

            full_report = ExperimentOrchestrator(registry).run((
                "statistics",
                "plotting",
            ))
            self.assertEqual(full_report.execution_order, (
                "clustering",
                "os-stage",
                "mse-stage",
                "combine-metrics",
                "statistics",
                "plotting",
            ))
            self.assertTrue(full_report.succeeded)
            self.assertEqual(full_report.final_products.keys, (
                "partitions.main",
                "metrics.os",
                "metrics.mse",
                "metrics.combined",
                "curves.selected",
                "statistics.main",
                "plots.main",
            ))
            self.assertEqual(
                tuple(record.native_report is not None for record in full_report.records),
                (True, True, True, False, False, False),
            )
            statistic_results = full_report.final_products.product(
                "statistics.main"
            ).value
            plot_run = full_report.final_products.product("plots.main").value
            self.assertEqual(statistic_results.rows[0].statistic, "mae")
            self.assertEqual(plot_run.figure_path.read_bytes(), b"plot")
            self.assertTrue(plot_run.data_path.is_file())


if __name__ == "__main__":
    unittest.main()
