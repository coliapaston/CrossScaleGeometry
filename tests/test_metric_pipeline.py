from pathlib import Path
import tempfile
import unittest

import pandas as pd

from crossscale.metric_pipeline import (
    METRIC_RESULT_COLUMNS,
    MetricEvaluation,
    MetricObservation,
    MultiScriptEntropyMetric,
    MetricResultRow,
    MetricResultTable,
    MetricOutputLayout,
    ModelMetricContext,
    OrthographicSimilarityMetric,
    PartitionMetric,
    PartitionMetricPipeline,
    aggregate_seed_metric_results,
    multi_script_entropy_summary_frame,
    orthographic_similarity_summary_frame,
)
from crossscale.clustering_pipeline import PartitionManifest, PartitionManifestRow
from crossscale.pipeline_core import RunPolicy
from crossscale.pipeline_core import ScaleTask


def make_task(**overrides):
    values = {
        "analysis_id": "low_energy_lt050_v1",
        "model_name": "mistralai/Mistral-7B-v0.1",
        "space_name": "output_proj",
        "scale": 44,
        "source": "full_pca_prefix",
        "seed_type": None,
        "seed": None,
    }
    values.update(overrides)
    return ScaleTask(**values)


def make_row(task=None, **overrides):
    values = {
        "task": task or make_task(),
        "partition_method": "hdbscan",
        "metric": "orthographic_similarity",
        "component": "score",
        "aggregation": "mean",
        "value": 0.42,
        "scope": "partition",
        "config_hash": "a" * 64,
        "reused": False,
    }
    values.update(overrides)
    return MetricResultRow(**values)


class DummyMetric:
    name = "dummy_cluster_fraction"

    def __init__(self):
        self.calls = []

    def config(self):
        return {"cluster_column": "cluster_id"}

    def prepare(self, model_context):
        self.calls.append(("prepare", model_context.model_name))
        return {"model_name": model_context.model_name}

    def evaluate(self, partition, prepared_state):
        self.calls.append(("evaluate", prepared_state["model_name"], len(partition)))
        clustered_fraction = float((partition["cluster_id"] >= 0).mean())
        return MetricEvaluation(
            observations=(
                MetricObservation(
                    component="clustered_fraction",
                    aggregation="mean",
                    value=clustered_fraction,
                ),
            ),
            detail_tables={"counts": partition.groupby("cluster_id").size().reset_index()},
            metadata={"row_count": len(partition)},
        )

    def finalize(self, prepared_state):
        self.calls.append(("finalize", prepared_state["model_name"]))


class MetricResultRowTests(unittest.TestCase):
    def test_flattened_row_uses_canonical_columns(self):
        row = make_row()
        serialized = row.to_dict()

        self.assertEqual(tuple(serialized), METRIC_RESULT_COLUMNS)
        self.assertEqual(serialized["analysis_id"], row.task.analysis_id)
        self.assertEqual(serialized["metric"], row.metric)

    def test_seeded_round_trip_preserves_identity(self):
        row = make_row(
            task=make_task(seed_type="projection", seed=42),
            reused=True,
        )
        restored = MetricResultRow.from_dict(row.to_dict())
        self.assertEqual(restored, row)
        self.assertEqual(restored.identity, row.identity)

    def test_rejects_nonfinite_value(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite scalar"):
                    make_row(value=value)

    def test_rejects_empty_semantic_fields(self):
        fields = (
            "partition_method",
            "metric",
            "component",
            "aggregation",
            "scope",
            "config_hash",
        )
        for field_name in fields:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    make_row(**{field_name: ""})


class MetricResultTableTests(unittest.TestCase):
    def test_long_form_supports_multiple_components_and_metrics(self):
        task = make_task()
        table = MetricResultTable(
            rows=(
                make_row(task=task),
                make_row(
                    task=task,
                    metric="multi_script_entropy",
                    component="normalized",
                    value=0.3,
                ),
                make_row(
                    task=task,
                    metric="multi_script_entropy",
                    component="raw",
                    value=0.7,
                ),
            )
        )
        frame = table.to_frame()

        self.assertEqual(tuple(frame.columns), METRIC_RESULT_COLUMNS)
        self.assertEqual(len(frame), 3)
        self.assertEqual(set(frame["component"]), {"score", "normalized", "raw"})

    def test_combine_preserves_table_and_row_order(self):
        os_row = make_row(metric="orthographic_similarity", value=0.4)
        mse_row = make_row(
            metric="multi_script_entropy",
            component="normalized",
            value=0.2,
        )
        combined = MetricResultTable.combine(
            MetricResultTable((os_row,)),
            MetricResultTable((mse_row,)),
        )
        self.assertEqual(combined.rows, (os_row, mse_row))

    def test_combine_accepts_empty_input_and_empty_tables(self):
        empty = MetricResultTable.combine()
        combined = MetricResultTable.combine(empty, MetricResultTable())
        self.assertEqual(empty.rows, ())
        self.assertEqual(combined.rows, ())

    def test_combine_rejects_wrong_type_and_duplicate_identity(self):
        table = MetricResultTable((make_row(),))
        with self.assertRaises(TypeError):
            MetricResultTable.combine(table, "not-a-table")
        with self.assertRaises(ValueError):
            MetricResultTable.combine(table, table)


class OrthographicSimilaritySummaryFrameTests(unittest.TestCase):
    @staticmethod
    def _table(task=None):
        selected_task = task or make_task()
        return MetricResultTable(
            rows=(
                make_row(task=selected_task, value=0.42),
                make_row(
                    task=selected_task,
                    aggregation="std",
                    value=0.12,
                ),
                make_row(
                    task=selected_task,
                    component="cluster_count",
                    aggregation="count",
                    value=37.0,
                ),
            )
        )

    def test_returns_legacy_unseeded_summary_columns(self):
        frame = orthographic_similarity_summary_frame(self._table())

        self.assertEqual(
            list(frame.columns),
            [
                "model_name",
                "space_name",
                "pca_dim",
                "l2_norm",
                "min_cluster_size",
                "min_samples",
                "metric",
                "cluster_selection_method",
                "cluster_selection_epsilon",
                "global_mean",
                "global_std",
                "n_clusters",
            ],
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "pca_dim"], 44)
        self.assertEqual(frame.loc[0, "global_mean"], 0.42)
        self.assertEqual(frame.loc[0, "global_std"], 0.12)
        self.assertEqual(frame.loc[0, "n_clusters"], 37)

    def test_preserves_seed_under_its_semantic_column(self):
        task = make_task(seed_type="pca_seed", seed=1813382118)
        frame = orthographic_similarity_summary_frame(self._table(task))

        self.assertIn("pca_seed", frame.columns)
        self.assertEqual(frame.loc[0, "pca_seed"], 1813382118)

    def test_rejects_incomplete_observation_set(self):
        table = MetricResultTable(rows=(make_row(),))

        with self.assertRaisesRegex(ValueError, "Invalid O-S observations"):
            orthographic_similarity_summary_frame(table)


class MultiScriptEntropySummaryFrameTests(unittest.TestCase):
    @staticmethod
    def _table(task=None):
        selected_task = task or make_task()
        base = {
            "task": selected_task,
            "metric": "multi_script_entropy",
        }
        return MetricResultTable(
            rows=(
                make_row(**base, component="raw", value=0.32),
                make_row(
                    **base,
                    component="raw",
                    aggregation="std",
                    value=0.48,
                ),
                make_row(**base, component="normalized", value=0.09),
                make_row(
                    **base,
                    component="normalized",
                    aggregation="std",
                    value=0.14,
                ),
                make_row(
                    **base,
                    component="cluster_count",
                    aggregation="count",
                    value=308.0,
                ),
            )
        )

    def test_returns_legacy_unseeded_summary_columns(self):
        frame = multi_script_entropy_summary_frame(self._table())

        self.assertEqual(
            list(frame.columns),
            [
                "model_name",
                "space_name",
                "pca_dim",
                "l2_norm",
                "min_cluster_size",
                "min_samples",
                "metric",
                "cluster_selection_method",
                "cluster_selection_epsilon",
                "num_clusters",
                "mean_H",
                "std_H",
                "mean_H_norm",
                "std_H_norm",
            ],
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "num_clusters"], 308)
        self.assertEqual(frame.loc[0, "mean_H"], 0.32)
        self.assertEqual(frame.loc[0, "std_H"], 0.48)
        self.assertEqual(frame.loc[0, "mean_H_norm"], 0.09)
        self.assertEqual(frame.loc[0, "std_H_norm"], 0.14)

    def test_preserves_seed_under_its_semantic_column(self):
        task = make_task(seed_type="permutation_seed", seed=42)
        frame = multi_script_entropy_summary_frame(self._table(task))

        self.assertIn("permutation_seed", frame.columns)
        self.assertEqual(frame.loc[0, "permutation_seed"], 42)

    def test_rejects_incomplete_observation_set(self):
        table = MetricResultTable(
            rows=(
                make_row(metric="multi_script_entropy", component="raw"),
            )
        )

        with self.assertRaisesRegex(ValueError, "Invalid MS-E observations"):
            multi_script_entropy_summary_frame(table)


class SeedMetricAggregationTests(unittest.TestCase):
    def make_seed_table(self, seeds=(0, 42), scales=(2, 3)):
        rows = []
        for scale in scales:
            for seed in seeds:
                rows.append(
                    make_row(
                        task=make_task(
                            analysis_id="random_control",
                            scale=scale,
                            source="randomized_pca",
                            seed_type="pca_seed",
                            seed=seed,
                        ),
                        value=float(scale + seed / 42 if seed else scale),
                        config_hash=f"{scale:02d}{seed:062d}",
                    )
                )
        return MetricResultTable(rows=tuple(rows))

    def test_builds_canonical_seed_mean_and_sample_sd_rows(self):
        aggregated = aggregate_seed_metric_results(
            self.make_seed_table(),
            input_analysis_id="random_control",
            input_source="randomized_pca",
            seed_type="pca_seed",
            expected_seeds=(0, 42),
            output_analysis_id="random_control_seed_aggregate",
            output_source="randomized_pca_seed_aggregate",
        )

        self.assertEqual(len(aggregated.rows), 4)
        self.assertEqual(
            {row.aggregation for row in aggregated.rows},
            {"seed_mean", "seed_std"},
        )
        self.assertTrue(all(row.task.seed is None for row in aggregated.rows))
        values = {
            (row.task.scale, row.aggregation): row.value
            for row in aggregated.rows
        }
        self.assertAlmostEqual(values[(2, "seed_mean")], 2.5)
        self.assertAlmostEqual(values[(2, "seed_std")], 2 ** -0.5)

    def test_rejects_missing_or_unexpected_seeds(self):
        with self.assertRaisesRegex(ValueError, "does not match expected seeds"):
            aggregate_seed_metric_results(
                self.make_seed_table(seeds=(0,)),
                input_analysis_id="random_control",
                input_source="randomized_pca",
                seed_type="pca_seed",
                expected_seeds=(0, 42),
                output_analysis_id="aggregate",
                output_source="seed_aggregate",
            )

    def test_rejects_invalid_seed_and_ddof_contracts(self):
        table = self.make_seed_table()
        common = {
            "input_analysis_id": "random_control",
            "input_source": "randomized_pca",
            "seed_type": "pca_seed",
            "output_analysis_id": "aggregate",
            "output_source": "seed_aggregate",
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_seed_metric_results(
                table,
                expected_seeds=(0, 0),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "smaller"):
            aggregate_seed_metric_results(
                table,
                expected_seeds=(0, 42),
                ddof=2,
                **common,
            )


class PartitionMetricContractTests(unittest.TestCase):
    def test_model_context_validates_model_and_metadata(self):
        context = ModelMetricContext(
            model_name="mistralai/Mistral-7B-v0.1",
            metadata={"tokenizer_path": "mistralai/Mistral-7B-v0.1/tokenizer"},
        )

        self.assertEqual(context.model_name, "mistralai/Mistral-7B-v0.1")
        self.assertEqual(
            context.metadata["tokenizer_path"],
            "mistralai/Mistral-7B-v0.1/tokenizer",
        )
        with self.assertRaisesRegex(ValueError, "model_name"):
            ModelMetricContext(model_name="")
        with self.assertRaisesRegex(ValueError, "metadata keys"):
            ModelMetricContext(model_name="model", metadata={"": "value"})

    def test_metric_observation_validates_identity_and_value(self):
        observation = MetricObservation(
            component="score",
            aggregation="median",
            value=0.25,
            scope="cluster",
        )

        self.assertEqual(observation.identity, ("score", "median", "cluster"))
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            MetricObservation(component="score", aggregation="mean", value=float("nan"))

    def test_metric_evaluation_supports_multiple_values_and_details(self):
        details = pd.DataFrame({"cluster_id": [0, 1], "score": [0.2, 0.8]})
        evaluation = MetricEvaluation(
            observations=(
                MetricObservation("score", "mean", 0.5),
                MetricObservation("score", "median", 0.5),
            ),
            detail_tables={"clusters": details},
            metadata={"n_clusters": 2},
        )

        self.assertEqual(len(evaluation.observations), 2)
        pd.testing.assert_frame_equal(evaluation.detail_tables["clusters"], details)
        self.assertEqual(evaluation.metadata["n_clusters"], 2)

    def test_metric_evaluation_rejects_duplicate_values(self):
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            MetricEvaluation(
                observations=(
                    MetricObservation("score", "mean", 0.1),
                    MetricObservation("score", "mean", 0.2),
                )
            )

    def test_dummy_metric_conforms_and_runs_full_lifecycle(self):
        metric = DummyMetric()
        self.assertIsInstance(metric, PartitionMetric)
        context = ModelMetricContext(model_name="example/model")
        partition = pd.DataFrame(
            {
                "token_id": [0, 1, 2, 3],
                "cluster_id": [0, -1, 1, -1],
                "probability": [1.0, 0.0, 0.8, 0.0],
            }
        )

        state = metric.prepare(context)
        evaluation = metric.evaluate(partition, state)
        metric.finalize(state)

        self.assertEqual(evaluation.observations[0].value, 0.5)
        self.assertEqual(
            metric.calls,
            [
                ("prepare", "example/model"),
                ("evaluate", "example/model", 4),
                ("finalize", "example/model"),
            ],
        )


class OrthographicSimilarityMetricTests(unittest.TestCase):
    def test_matches_existing_numeric_functions_exactly(self):
        from crossscale.metrics.orthographic import (
            compute_cluster_level_morphology_df,
            compute_global_cluster_stats,
        )
        from crossscale.partition.hdbscan import filter_valid_clusters

        partition = pd.DataFrame(
            {
                "token_id": [0, 1, 2, 3, 4, 5],
                "cluster_id": [0, 0, 1, 1, 1, -1],
                "token_str": ["cat", "bat", "dog", "fog", "log", "noise"],
            }
        )
        legacy_details = compute_cluster_level_morphology_df(
            filter_valid_clusters(partition),
        )
        legacy_stats = compute_global_cluster_stats(legacy_details, ddof=0)
        metric = OrthographicSimilarityMetric(
            tokenizer_loader=lambda path: self.fail("tokenizer should be supplied")
        )
        state = metric.prepare(
            ModelMetricContext(model_name="example/model", metadata={"tokenizer": object()})
        )

        evaluation = metric.evaluate(partition, state)

        pd.testing.assert_frame_equal(
            evaluation.detail_tables["clusters"],
            legacy_details,
            check_exact=True,
        )
        values = {
            (item.component, item.aggregation): item.value
            for item in evaluation.observations
        }
        self.assertEqual(values[("score", "mean")], legacy_stats["global_mean"])
        self.assertEqual(values[("score", "std")], legacy_stats["global_std"])
        self.assertEqual(values[("cluster_count", "count")], 2.0)

    def test_prepare_loads_tokenizer_once_and_recovers_token_strings(self):
        class FakeTokenizer:
            def convert_ids_to_tokens(self, token_id):
                return {0: "cat", 1: "bat", 2: "dog", 3: "fog"}[int(token_id)]

        load_calls = []

        def load_tokenizer(path):
            load_calls.append(path)
            return FakeTokenizer()

        metric = OrthographicSimilarityMetric(tokenizer_loader=load_tokenizer)
        context = ModelMetricContext(
            model_name="example/model",
            metadata={"tokenizer_path": "example/tokenizer"},
        )
        partition = pd.DataFrame(
            {
                "token_id": [0, 1, 2, 3],
                "cluster_id": [0, 0, 1, 1],
            }
        )

        state = metric.prepare(context)
        first = metric.evaluate(partition, state)
        second = metric.evaluate(partition, state)

        self.assertEqual(load_calls, ["example/tokenizer"])
        pd.testing.assert_frame_equal(
            first.detail_tables["clusters"],
            second.detail_tables["clusters"],
            check_exact=True,
        )
        self.assertIsInstance(metric, PartitionMetric)

    def test_config_declares_fixed_large_cluster_approximation(self):
        approximation = OrthographicSimilarityMetric().config()[
            "large_cluster_approximation"
        ]

        self.assertEqual(
            approximation,
            {
                "exact_token_limit": 512,
                "sample_size": 128,
                "n_repeats": 10,
                "random_state": 1813382118,
            },
        )


class MultiScriptEntropyMetricTests(unittest.TestCase):
    class FakeTokenizer:
        tokens = ["cat", "bat", "кот", "пес", "gata", "skylos", "!"]
        vocab_size = len(tokens)

        def convert_ids_to_tokens(self, token_ids):
            if isinstance(token_ids, list):
                return [self.tokens[int(token_id)] for token_id in token_ids]
            return self.tokens[int(token_ids)]

    def test_matches_existing_numeric_chain_exactly(self):
        from crossscale.metrics.script_entropy import (
            annotate_token_scripts,
            build_cluster_script_count_df,
            compute_cluster_entropy_df,
            filter_noise_clusters,
            partition_entropy_stats,
            standardize_cluster_input_df,
        )

        tokenizer = self.FakeTokenizer()
        partition = pd.DataFrame(
            {
                "token_id": [0, 2, 4, 1, 3, 5, 6],
                "cluster_id": [0, 0, 0, 1, 1, 1, -1],
            }
        )
        metric = MultiScriptEntropyMetric(
            tokenizer_loader=lambda path: self.fail("tokenizer should be supplied")
        )
        state = metric.prepare(
            ModelMetricContext(
                model_name="example/model",
                metadata={"tokenizer": tokenizer},
            )
        )
        global_scripts = state["script_cache"]["global_script_set"]
        legacy_tokens = annotate_token_scripts(
            standardize_cluster_input_df(
                partition,
                model_name="example/model",
                tokenizer=tokenizer,
            )
        )
        legacy_counts = build_cluster_script_count_df(
            filter_noise_clusters(legacy_tokens),
            global_script_set=global_scripts,
        )
        legacy_clusters = compute_cluster_entropy_df(
            legacy_counts,
            global_script_set=global_scripts,
        )
        legacy_stats = partition_entropy_stats(legacy_clusters)

        evaluation = metric.evaluate(partition, state)

        self.assertEqual(set(evaluation.detail_tables), {"clusters"})
        pd.testing.assert_frame_equal(
            evaluation.detail_tables["clusters"], legacy_clusters, check_exact=True
        )
        values = {
            (item.component, item.aggregation): item.value
            for item in evaluation.observations
        }
        self.assertEqual(values[("raw", "mean")], legacy_stats["mean_H"])
        self.assertEqual(values[("raw", "std")], legacy_stats["std_H"])
        self.assertEqual(
            values[("normalized", "mean")], legacy_stats["mean_H_norm"]
        )
        self.assertEqual(
            values[("normalized", "std")], legacy_stats["std_H_norm"]
        )
        self.assertEqual(values[("cluster_count", "count")], 2.0)

    def test_prepare_loads_tokenizer_and_builds_cache_once(self):
        load_calls = []

        def load_tokenizer(path):
            load_calls.append(path)
            return self.FakeTokenizer()

        metric = MultiScriptEntropyMetric(tokenizer_loader=load_tokenizer)
        state = metric.prepare(
            ModelMetricContext(
                model_name="example/model",
                metadata={"tokenizer_path": "example/tokenizer"},
            )
        )
        partition = pd.DataFrame(
            {"token_id": [0, 2, 1, 3], "cluster_id": [0, 0, 1, 1]}
        )

        first = metric.evaluate(partition, state)
        second = metric.evaluate(partition, state)

        self.assertEqual(load_calls, ["example/tokenizer"])
        self.assertGreater(state["script_cache"]["global_script_set_size"], 1)
        self.assertNotIn("vocab_df", state["script_cache"])
        pd.testing.assert_frame_equal(
            first.detail_tables["clusters"],
            second.detail_tables["clusters"],
            check_exact=True,
        )
        self.assertIsInstance(metric, PartitionMetric)

    def test_ignores_supplied_token_text_and_decodes_from_token_ids(self):
        tokenizer = self.FakeTokenizer()
        metric = MultiScriptEntropyMetric(
            tokenizer_loader=lambda path: tokenizer
        )
        state = metric.prepare(
            ModelMetricContext(
                model_name="example/model",
                metadata={"tokenizer": tokenizer},
            )
        )
        partition = pd.DataFrame(
            {
                "token_id": [0, 2, 1, 3],
                "cluster_id": [0, 0, 1, 1],
                "token_str": ["肩", "肩", "肩", "肩"],
            }
        )
        canonical = partition.drop(columns="token_str")

        observed = metric.evaluate(partition, state)
        expected = metric.evaluate(canonical, state)

        self.assertEqual(observed.observations, expected.observations)
        pd.testing.assert_frame_equal(
            observed.detail_tables["clusters"],
            expected.detail_tables["clusters"],
            check_exact=True,
        )


class PartitionMetricPipelineTests(unittest.TestCase):
    def _manifest(self, root, task_specs):
        rows = []
        for index, task in enumerate(task_specs):
            partition_path = root / f"partition_{index}.csv"
            partition = pd.DataFrame(
                {
                    "token_id": [0, 1, 2, 3],
                    "cluster_id": [0, -1, index, -1],
                }
            )
            partition.to_csv(partition_path, index=False)
            rows.append(
                PartitionManifestRow(
                    task=task,
                    method="fake_partition",
                    partition_csv_path=str(partition_path),
                    n_tokens=4,
                    n_clusters=len(set(partition["cluster_id"]) - {-1}),
                    n_noise=2,
                    noise_fraction=0.5,
                    config_hash=f"partition-{index}",
                    reused=False,
                )
            )
        return PartitionManifest(rows=tuple(rows))

    def test_runs_manifest_and_prepares_once_per_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tasks = (
                make_task(scale=44),
                make_task(scale=95),
                make_task(model_name="example/second", scale=44),
                make_task(model_name="example/second", scale=95),
            )
            manifest = self._manifest(root, tasks)
            context_calls = []

            def context_factory(model_name, rows):
                context_calls.append((model_name, len(rows)))
                return ModelMetricContext(model_name=model_name)

            metric = DummyMetric()
            pipeline = PartitionMetricPipeline(
                metric=metric,
                output_layout=MetricOutputLayout(root / "metric_output"),
                context_factory=context_factory,
            )

            run = pipeline.run(manifest)

            self.assertEqual(len(run.results.rows), 4)
            self.assertEqual(run.report.computed_count, 4)
            self.assertEqual(
                context_calls,
                [
                    ("mistralai/Mistral-7B-v0.1", 2),
                    ("example/second", 2),
                ],
            )
            self.assertEqual(
                [call[0] for call in metric.calls].count("prepare"),
                2,
            )
            self.assertEqual(
                [call[0] for call in metric.calls].count("evaluate"),
                4,
            )
            self.assertEqual(
                [call[0] for call in metric.calls].count("finalize"),
                2,
            )
            self.assertTrue(run.result_table_path.is_file())
            for record in run.report.records:
                self.assertTrue(all(Path(path).is_file() for path in record.output_paths))

    def test_accepts_manifest_csv_and_does_not_require_probability(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._manifest(root, (make_task(),))
            manifest_path = root / "partition_manifest.csv"
            manifest.write_csv(manifest_path)
            metric = DummyMetric()
            pipeline = PartitionMetricPipeline(
                metric=metric,
                output_layout=MetricOutputLayout(root / "metric_output"),
            )

            run = pipeline.run(manifest_path)

            self.assertEqual(run.results.rows[0].value, 0.5)
            self.assertEqual(run.results.rows[0].partition_method, "fake_partition")

    def test_resume_validates_outputs_without_preparing_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._manifest(root, (make_task(), make_task(scale=95)))
            layout = MetricOutputLayout(root / "metric_output")
            PartitionMetricPipeline(metric=DummyMetric(), output_layout=layout).run(manifest)
            resumed_metric = DummyMetric()
            resumed = PartitionMetricPipeline(
                metric=resumed_metric,
                output_layout=layout,
                run_policy=RunPolicy(existing_output="resume"),
            ).run(manifest)

            self.assertEqual(resumed.report.reused_count, 2)
            self.assertEqual(resumed.report.computed_count, 0)
            self.assertTrue(all(row.reused for row in resumed.results.rows))
            self.assertEqual(resumed_metric.calls, [])

    def test_resume_rejects_incomplete_detail_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task = make_task()
            manifest = self._manifest(root, (task,))
            layout = MetricOutputLayout(root / "metric_output")
            PartitionMetricPipeline(metric=DummyMetric(), output_layout=layout).run(manifest)
            paths = layout.paths_for(task, "dummy_cluster_fraction", "fake_partition")
            (paths.details_dir / "counts.csv").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "missing detail"):
                PartitionMetricPipeline(
                    metric=DummyMetric(),
                    output_layout=layout,
                    run_policy=RunPolicy(existing_output="resume"),
                ).run(manifest)

    def test_error_mode_preflights_before_metric_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._manifest(root, (make_task(),))
            layout = MetricOutputLayout(root / "metric_output")
            PartitionMetricPipeline(metric=DummyMetric(), output_layout=layout).run(manifest)
            metric = DummyMetric()

            with self.assertRaisesRegex(FileExistsError, "outputs already exist"):
                PartitionMetricPipeline(metric=metric, output_layout=layout).run(manifest)
            self.assertEqual(metric.calls, [])

    def test_replace_recomputes_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._manifest(root, (make_task(),))
            layout = MetricOutputLayout(root / "metric_output")
            PartitionMetricPipeline(metric=DummyMetric(), output_layout=layout).run(manifest)
            metric = DummyMetric()
            replaced = PartitionMetricPipeline(
                metric=metric,
                output_layout=layout,
                run_policy=RunPolicy(existing_output="replace"),
            ).run(manifest)

            self.assertEqual(replaced.report.computed_count, 1)
            self.assertEqual([call[0] for call in metric.calls], ["prepare", "evaluate", "finalize"])

    def test_orthographic_adapter_pipeline_writes_details_and_resumes(self):
        class FakeTokenizer:
            def convert_ids_to_tokens(self, token_id):
                return ["cat", "noise", "bat", "unused"][int(token_id)]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task = make_task()
            manifest = self._manifest(root, (task,))
            layout = MetricOutputLayout(root / "os_output")
            load_calls = []

            def load_tokenizer(path):
                load_calls.append(path)
                return FakeTokenizer()

            metric = OrthographicSimilarityMetric(tokenizer_loader=load_tokenizer)
            run = PartitionMetricPipeline(
                metric=metric,
                output_layout=layout,
            ).run(manifest)

            self.assertEqual(len(run.results.rows), 3)
            self.assertEqual(load_calls, ["mistralai/Mistral-7B-v0.1/tokenizer"])
            paths = layout.paths_for(task, metric.name, "fake_partition")
            self.assertTrue((paths.details_dir / "clusters.csv").is_file())

            resumed_load_calls = []
            resumed = PartitionMetricPipeline(
                metric=OrthographicSimilarityMetric(
                    tokenizer_loader=lambda path: resumed_load_calls.append(path)
                ),
                output_layout=layout,
                run_policy=RunPolicy(existing_output="resume"),
            ).run(manifest)
            self.assertEqual(resumed.report.reused_count, 1)
            self.assertEqual(resumed_load_calls, [])

    def test_multi_script_adapter_pipeline_writes_details_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task = make_task()
            manifest = self._manifest(root, (task,))
            layout = MetricOutputLayout(root / "mse_output")
            load_calls = []

            def load_tokenizer(path):
                load_calls.append(path)
                return MultiScriptEntropyMetricTests.FakeTokenizer()

            metric = MultiScriptEntropyMetric(tokenizer_loader=load_tokenizer)
            run = PartitionMetricPipeline(
                metric=metric,
                output_layout=layout,
            ).run(manifest)

            self.assertEqual(len(run.results.rows), 5)
            self.assertEqual(load_calls, ["mistralai/Mistral-7B-v0.1/tokenizer"])
            paths = layout.paths_for(task, metric.name, "fake_partition")
            self.assertEqual(
                {path.name for path in paths.details_dir.glob("*.csv")},
                {"clusters.csv"},
            )

            resumed_load_calls = []
            resumed = PartitionMetricPipeline(
                metric=MultiScriptEntropyMetric(
                    tokenizer_loader=lambda path: resumed_load_calls.append(path)
                ),
                output_layout=layout,
                run_policy=RunPolicy(existing_output="resume"),
            ).run(manifest)
            self.assertEqual(resumed.report.reused_count, 1)
            self.assertEqual(resumed_load_calls, [])

    def test_multi_script_pipeline_never_persists_decoded_token_text(self):
        class ControlCharacterTokenizer:
            tokens = ("\x00", "\x00\x00", "alpha", "beta")
            vocab_size = len(tokens)

            def decode(self, token_ids, clean_up_tokenization_spaces=False):
                return "".join(self.tokens[int(token_id)] for token_id in token_ids)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task = make_task(model_name="gpt-oss")
            manifest = self._manifest(root, (task,))
            layout = MetricOutputLayout(root / "mse_output")
            tokenizer = ControlCharacterTokenizer()

            run = PartitionMetricPipeline(
                metric=MultiScriptEntropyMetric(
                    tokenizer_loader=lambda path: tokenizer
                ),
                output_layout=layout,
            ).run(manifest)

            self.assertEqual(len(run.results.rows), 5)
            paths = layout.paths_for(task, "multi_script_entropy", "fake_partition")
            detail_paths = tuple(paths.details_dir.glob("*.csv"))
            self.assertEqual({path.name for path in detail_paths}, {"clusters.csv"})
            persisted = b"".join(
                path.read_bytes() for path in root.rglob("*") if path.is_file()
            )
            self.assertNotIn(b"\x00", persisted)
            self.assertNotIn(b"alpha", persisted)
            self.assertNotIn(b"beta", persisted)


class MetricResultTablePersistenceTests(unittest.TestCase):

    def test_rejects_duplicate_observation_identity(self):
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            MetricResultTable(rows=(make_row(value=0.1), make_row(value=0.2)))

    def test_csv_round_trip_preserves_seed_and_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            table = MetricResultTable(
                rows=(
                    make_row(),
                    make_row(
                        task=make_task(
                            scale=95,
                            seed_type="projection",
                            seed=1000,
                        ),
                        value=0.9,
                        reused=True,
                    ),
                )
            )
            path = Path(temp_dir) / "nested" / "metric_results.csv"

            table.write_csv(path)
            restored = MetricResultTable.read_csv(path)

            self.assertEqual(restored, table)
            self.assertEqual(str(table.to_frame()["seed"].dtype), "Int64")

    def test_write_refuses_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metric_results.csv"
            table = MetricResultTable(rows=(make_row(),))
            table.write_csv(path)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                table.write_csv(path)
            table.write_csv(path, overwrite=True)

    def test_rejects_noncanonical_frame(self):
        frame = MetricResultTable(rows=(make_row(),)).to_frame()
        reordered = frame[list(reversed(frame.columns))]
        with self.assertRaisesRegex(ValueError, "canonical order"):
            MetricResultTable.from_frame(reordered)

    def test_empty_table_retains_schema(self):
        frame = MetricResultTable().to_frame()
        self.assertTrue(frame.empty)
        self.assertEqual(tuple(frame.columns), METRIC_RESULT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
