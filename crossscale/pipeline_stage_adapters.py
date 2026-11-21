"""Thin typed adapters from concrete pipelines to experiment stages."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .clustering_pipeline import ClusteringPipelineRun, PartitionManifest
from .experiment_orchestrator import (
    CallableExperimentStage,
    DataProduct,
    DataProductSpec,
    StageExecution,
)
from .pipeline_core import ScaleTask
from .metric_pipeline import MetricResultTable, PartitionMetricPipelineRun
from .plotting_pipeline import FigureSpec, PlotPipelineRun
from .statistics_pipeline import (
    ComposedAlignedPair,
    ComposedStatisticsPipelineRun,
    CurveQuery,
    CurveTable,
    StatisticResultTable,
    StatisticsPipelineRun,
)


@runtime_checkable
class ClusteringPipelineRunner(Protocol):
    """Run a bound collection of clustering scale tasks."""

    def run(self, tasks: tuple[ScaleTask, ...]) -> ClusteringPipelineRun:
        """Return a manifest and native clustering report."""


@runtime_checkable
class PartitionMetricPipelineRunner(Protocol):
    """Evaluate one configured metric over a partition manifest."""

    def run(self, manifest: PartitionManifest) -> PartitionMetricPipelineRun:
        """Return a metric result table and native task report."""


@runtime_checkable
class StatisticsPipelineRunner(Protocol):
    """Compute pairwise statistics over explicitly selected curves."""

    def run(
        self,
        metric_results: MetricResultTable,
        queries: tuple[CurveQuery, ...],
    ) -> StatisticsPipelineRun:
        """Return selected curves, alignments, and statistic results."""


@runtime_checkable
class ComposedStatisticsPipelineRunner(Protocol):
    """Compute statistics over explicitly composed feature curves."""

    def run(
        self,
        metric_results: MetricResultTable,
        *,
        reference_queries: tuple[CurveQuery, ...],
        target_queries: tuple[CurveQuery, ...],
    ) -> ComposedStatisticsPipelineRun:
        """Return source curves, composed pair, and statistic results."""


@runtime_checkable
class PlotPipelineRunner(Protocol):
    """Render one bound figure from a metric result table."""

    def run(
        self,
        results: MetricResultTable,
        figure_spec: FigureSpec,
    ) -> PlotPipelineRun:
        """Return one complete committed plotting result."""


def make_clustering_stage(
    *,
    stage_id: str,
    pipeline: ClusteringPipelineRunner,
    tasks: tuple[ScaleTask, ...],
    product_key: str = "partitions.main",
) -> CallableExperimentStage:
    """Bind one clustering pipeline run to a manifest-producing stage."""

    if not isinstance(pipeline, ClusteringPipelineRunner):
        raise TypeError("pipeline must satisfy ClusteringPipelineRunner")
    if not isinstance(tasks, tuple) or not tasks:
        raise ValueError("tasks must be a non-empty tuple")
    if any(not isinstance(task, ScaleTask) for task in tasks):
        raise TypeError("tasks must contain ScaleTask objects")
    task_keys = tuple(task.task_key for task in tasks)
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("tasks must have unique identities")

    manifest_spec = DataProductSpec(product_key, PartitionManifest)

    def execute_clustering(store):
        run = pipeline.run(tasks)
        if not isinstance(run, ClusteringPipelineRun):
            raise TypeError("clustering pipeline must return ClusteringPipelineRun")
        return StageExecution(
            products=(DataProduct(
                manifest_spec,
                run.manifest,
                producer_stage=stage_id,
            ),),
            native_report=run.report,
        )

    return CallableExperimentStage(
        stage_id=stage_id,
        dependencies=(),
        required_products=(),
        provided_products=(manifest_spec,),
        executor=execute_clustering,
    )


def make_partition_metric_stage(
    *,
    stage_id: str,
    pipeline: PartitionMetricPipelineRunner,
    dependencies: tuple[str, ...],
    manifest_key: str = "partitions.main",
    result_key: str = "metrics.main",
) -> CallableExperimentStage:
    """Bind one partition metric pipeline to typed manifest and result keys."""

    if not isinstance(pipeline, PartitionMetricPipelineRunner):
        raise TypeError("pipeline must satisfy PartitionMetricPipelineRunner")
    manifest_spec = DataProductSpec(manifest_key, PartitionManifest)
    result_spec = DataProductSpec(result_key, MetricResultTable)

    def execute_metric(store):
        manifest = store.require(manifest_spec)
        run = pipeline.run(manifest)
        if not isinstance(run, PartitionMetricPipelineRun):
            raise TypeError(
                "partition metric pipeline must return PartitionMetricPipelineRun"
            )
        return StageExecution(
            products=(DataProduct(
                result_spec,
                run.results,
                producer_stage=stage_id,
            ),),
            native_report=run.report,
        )

    return CallableExperimentStage(
        stage_id=stage_id,
        dependencies=dependencies,
        required_products=(manifest_spec,),
        provided_products=(result_spec,),
        executor=execute_metric,
    )


def make_metric_table_combine_stage(
    *,
    stage_id: str,
    dependencies: tuple[str, ...],
    input_keys: tuple[str, ...],
    output_key: str = "metrics.combined",
) -> CallableExperimentStage:
    """Combine named metric tables into one new typed product."""

    if not isinstance(input_keys, tuple) or len(input_keys) < 2:
        raise ValueError("input_keys must contain at least two keys")
    if len(input_keys) != len(set(input_keys)):
        raise ValueError("input_keys must contain unique keys")
    input_specs = tuple(
        DataProductSpec(key, MetricResultTable) for key in input_keys
    )
    output_spec = DataProductSpec(output_key, MetricResultTable)

    def combine_tables(store):
        tables = tuple(store.require(spec) for spec in input_specs)
        combined = MetricResultTable.combine(*tables)
        return StageExecution((DataProduct(
            output_spec,
            combined,
            producer_stage=stage_id,
        ),))

    return CallableExperimentStage(
        stage_id=stage_id,
        dependencies=dependencies,
        required_products=input_specs,
        provided_products=(output_spec,),
        executor=combine_tables,
    )


def make_statistics_stage(
    *,
    stage_id: str,
    pipeline: StatisticsPipelineRunner,
    dependencies: tuple[str, ...],
    queries: tuple[CurveQuery, ...],
    metric_key: str = "metrics.combined",
    curves_key: str = "curves.selected",
    results_key: str = "statistics.main",
) -> CallableExperimentStage:
    """Bind a standard statistics pipeline to typed input and output keys."""

    if not isinstance(pipeline, StatisticsPipelineRunner):
        raise TypeError("pipeline must satisfy StatisticsPipelineRunner")
    if not isinstance(queries, tuple) or len(queries) < 2:
        raise ValueError("queries must contain at least two CurveQuery objects")
    if any(not isinstance(query, CurveQuery) for query in queries):
        raise TypeError("queries must contain CurveQuery objects")

    metric_spec = DataProductSpec(metric_key, MetricResultTable)
    curves_spec = DataProductSpec(curves_key, CurveTable)
    results_spec = DataProductSpec(results_key, StatisticResultTable)

    def execute_statistics(store):
        metric_results = store.require(metric_spec)
        run = pipeline.run(metric_results, queries)
        if not isinstance(run, StatisticsPipelineRun):
            raise TypeError("statistics pipeline must return StatisticsPipelineRun")
        return StageExecution(products=(
            DataProduct(
                curves_spec,
                run.curves,
                producer_stage=stage_id,
            ),
            DataProduct(
                results_spec,
                run.results,
                producer_stage=stage_id,
            ),
        ))

    return CallableExperimentStage(
        stage_id=stage_id,
        dependencies=dependencies,
        required_products=(metric_spec,),
        provided_products=(curves_spec, results_spec),
        executor=execute_statistics,
    )


def make_composed_statistics_stage(
    *,
    stage_id: str,
    pipeline: ComposedStatisticsPipelineRunner,
    dependencies: tuple[str, ...],
    reference_queries: tuple[CurveQuery, ...],
    target_queries: tuple[CurveQuery, ...],
    metric_key: str = "metrics.combined",
    reference_curves_key: str = "curves.reference",
    target_curves_key: str = "curves.target",
    composed_pair_key: str = "curves.composed-pair",
    results_key: str = "statistics.composed",
) -> CallableExperimentStage:
    """Bind a composed statistics pipeline to four typed output products."""

    if not isinstance(pipeline, ComposedStatisticsPipelineRunner):
        raise TypeError("pipeline must satisfy ComposedStatisticsPipelineRunner")
    for field_name, queries in (
        ("reference_queries", reference_queries),
        ("target_queries", target_queries),
    ):
        if not isinstance(queries, tuple) or not queries:
            raise ValueError(f"{field_name} must be a non-empty tuple")
        if any(not isinstance(query, CurveQuery) for query in queries):
            raise TypeError(f"{field_name} must contain CurveQuery objects")
    if len(reference_queries) != len(target_queries):
        raise ValueError("reference_queries and target_queries must have equal length")

    metric_spec = DataProductSpec(metric_key, MetricResultTable)
    reference_spec = DataProductSpec(reference_curves_key, CurveTable)
    target_spec = DataProductSpec(target_curves_key, CurveTable)
    pair_spec = DataProductSpec(composed_pair_key, ComposedAlignedPair)
    results_spec = DataProductSpec(results_key, StatisticResultTable)

    def execute_composed_statistics(store):
        metric_results = store.require(metric_spec)
        run = pipeline.run(
            metric_results,
            reference_queries=reference_queries,
            target_queries=target_queries,
        )
        if not isinstance(run, ComposedStatisticsPipelineRun):
            raise TypeError(
                "composed statistics pipeline must return "
                "ComposedStatisticsPipelineRun"
            )
        return StageExecution(products=(
            DataProduct(
                reference_spec,
                run.reference_curves,
                producer_stage=stage_id,
            ),
            DataProduct(
                target_spec,
                run.target_curves,
                producer_stage=stage_id,
            ),
            DataProduct(
                pair_spec,
                run.composed_pair,
                producer_stage=stage_id,
            ),
            DataProduct(
                results_spec,
                run.results,
                producer_stage=stage_id,
            ),
        ))

    return CallableExperimentStage(
        stage_id=stage_id,
        dependencies=dependencies,
        required_products=(metric_spec,),
        provided_products=(
            reference_spec,
            target_spec,
            pair_spec,
            results_spec,
        ),
        executor=execute_composed_statistics,
    )


def make_plot_stage(
    *,
    stage_id: str,
    pipeline: PlotPipelineRunner,
    dependencies: tuple[str, ...],
    figure_spec: FigureSpec,
    metric_key: str = "metrics.combined",
    plot_run_key: str = "plots.main",
) -> CallableExperimentStage:
    """Bind a plotting pipeline to one metric input and terminal run product."""

    if not isinstance(pipeline, PlotPipelineRunner):
        raise TypeError("pipeline must satisfy PlotPipelineRunner")
    if not isinstance(figure_spec, FigureSpec):
        raise TypeError("figure_spec must be a FigureSpec")
    metric_spec = DataProductSpec(metric_key, MetricResultTable)
    plot_run_spec = DataProductSpec(plot_run_key, PlotPipelineRun)

    def execute_plot(store):
        metric_results = store.require(metric_spec)
        run = pipeline.run(metric_results, figure_spec)
        if not isinstance(run, PlotPipelineRun):
            raise TypeError("plot pipeline must return PlotPipelineRun")
        if run.plot_data.figure_spec != figure_spec:
            raise ValueError("plot pipeline run does not match the bound FigureSpec")
        return StageExecution((DataProduct(
            plot_run_spec,
            run,
            producer_stage=stage_id,
        ),))

    return CallableExperimentStage(
        stage_id=stage_id,
        dependencies=dependencies,
        required_products=(metric_spec,),
        provided_products=(plot_run_spec,),
        executor=execute_plot,
    )
