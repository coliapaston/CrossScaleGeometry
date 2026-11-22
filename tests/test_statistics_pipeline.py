from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from crossscale.metric_pipeline import MetricResultRow, MetricResultTable
from crossscale.pipeline_core import ScaleTask
from crossscale.statistics_pipeline import (
    ALIGNMENT_STRATEGIES,
    AlignedCurvePair,
    CompositeObservationKey,
    ComposedAlignedPair,
    ComposedCurve,
    ComposedStatisticsPipeline,
    CorrelationMatrix,
    CorrelationMatrixBuilder,
    CURVE_TABLE_COLUMNS,
    CurveAligner,
    CurveComposer,
    CurveQuery,
    CurveRow,
    CurveTable,
    MAEStatistic,
    PAIR_MODES,
    PairwiseStatistic,
    PearsonStatistic,
    R2Statistic,
    SpearmanStatistic,
    STATISTIC_RESULT_COLUMNS,
    StatisticObservation,
    StatisticResult,
    StatisticResultRow,
    StatisticResultTable,
    StatisticsPipeline,
    WeightedMeanComposer,
    ZScoreConcatenationComposer,
)


def make_task(scale=10, **overrides):
    values = {
        "analysis_id": "analysis_a",
        "model_name": "model_a",
        "space_name": "output_proj",
        "scale": scale,
        "source": "main_pca",
        "seed_type": None,
        "seed": None,
    }
    values.update(overrides)
    return ScaleTask(**values)


def make_metric_row(scale=10, **overrides):
    values = {
        "task": make_task(scale=scale),
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


def make_query(**overrides):
    values = {
        "analysis_id": "analysis_a",
        "model_name": "model_a",
        "source": "main_pca",
        "metric": "orthographic_similarity",
        "component": "score",
        "aggregation": "mean",
        "scope": "partition",
        "space_name": "output_proj",
        "partition_method": "hdbscan",
    }
    values.update(overrides)
    return CurveQuery(**values)


def make_curve_row(scale=10, **overrides):
    query = make_query()
    values = {
        "curve_id": query.curve_id,
        "analysis_id": query.analysis_id,
        "model_name": query.model_name,
        "space_name": "output_proj",
        "source": query.source,
        "seed_type": query.seed_type,
        "seed": query.seed,
        "partition_method": "hdbscan",
        "metric": query.metric,
        "component": query.component,
        "aggregation": query.aggregation,
        "scope": query.scope,
        "scale": scale,
        "value": scale / 100.0,
        "config_hash": "a" * 64,
        "reused": False,
    }
    values.update(overrides)
    return CurveRow(**values)


class DummyMeanDifferenceStatistic:
    name = "mean_difference"

    def config(self):
        return {"direction": "target_minus_reference"}

    def compute(self, reference_values, target_values):
        reference = np.asarray(reference_values, dtype=float)
        target = np.asarray(target_values, dtype=float)
        if reference.shape != target.shape:
            raise ValueError("input shapes must match")
        return StatisticResult(
            statistic=self.name,
            observations=(
                StatisticObservation(
                    component="difference",
                    value=float(np.mean(target - reference)),
                ),
            ),
            n_points=len(reference),
            metadata=self.config(),
        )


class DummyConcatenationComposer:
    name = "dummy_concatenation"

    def config(self):
        return {"alignment_strategy": "exact"}

    def compose(self, reference_curves, target_curves):
        if len(reference_curves) != len(target_curves) or not reference_curves:
            raise ValueError("curve collections must have the same non-zero length")
        reference_values = []
        target_values = []
        keys = []
        reference_ids = []
        target_ids = []
        aligner = CurveAligner("exact")
        for feature_index, (reference, target) in enumerate(
            zip(reference_curves, target_curves)
        ):
            aligned = aligner.align(reference, target)
            feature_slot = f"feature_{feature_index}"
            keys.extend(
                CompositeObservationKey(feature_slot, scale)
                for scale in aligned.scales
            )
            reference_values.extend(aligned.reference_values)
            target_values.extend(aligned.target_values)
            reference_ids.append(aligned.reference_curve_id)
            target_ids.append(aligned.target_curve_id)
        config_hash = "d" * 64
        return ComposedAlignedPair(
            reference=ComposedCurve(
                curve_id="dummy_reference",
                composer_name=self.name,
                observation_keys=tuple(keys),
                values=tuple(reference_values),
                source_curve_ids=tuple(reference_ids),
                config_hash=config_hash,
            ),
            target=ComposedCurve(
                curve_id="dummy_target",
                composer_name=self.name,
                observation_keys=tuple(keys),
                values=tuple(target_values),
                source_curve_ids=tuple(target_ids),
                config_hash=config_hash,
            ),
        )


class CurveComposerContractTests(unittest.TestCase):
    @staticmethod
    def make_curve(seed, values):
        query = make_query(seed_type="curve", seed=seed)
        return CurveTable(rows=tuple(
            make_curve_row(
                scale=scale,
                value=value,
                curve_id=query.curve_id,
                seed_type="curve",
                seed=seed,
            )
            for scale, value in zip((10, 20, 30), values)
        ))

    def test_dummy_composer_combines_two_features_without_pipeline_changes(self):
        composer = DummyConcatenationComposer()
        references = (
            self.make_curve(1, (1.0, 2.0, 3.0)),
            self.make_curve(2, (2.0, 4.0, 8.0)),
        )
        targets = (
            self.make_curve(3, (1.2, 1.8, 3.4)),
            self.make_curve(4, (2.2, 3.7, 8.5)),
        )

        pair = composer.compose(references, targets)

        self.assertIsInstance(composer, CurveComposer)
        self.assertEqual(len(pair.observation_keys), 6)
        self.assertEqual(
            [key.feature_slot for key in pair.observation_keys],
            ["feature_0"] * 3 + ["feature_1"] * 3,
        )
        self.assertEqual(pair.observation_keys[0].scale, 10)
        self.assertEqual(pair.observation_keys[3].scale, 10)
        result = MAEStatistic().compute(pair.reference_array, pair.target_array)
        self.assertEqual(result.n_points, 6)

    def test_composed_curve_preserves_keys_and_returns_read_only_array(self):
        keys = (
            CompositeObservationKey("metric_a", 10),
            CompositeObservationKey("metric_b", 10),
        )
        curve = ComposedCurve(
            curve_id="composed_a",
            composer_name="dummy",
            observation_keys=keys,
            values=(0.1, 0.2),
            source_curve_ids=("curve_a", "curve_b"),
            config_hash="a" * 64,
            metadata={"normalization": "none"},
        )

        self.assertEqual(
            tuple(curve.to_frame().columns),
            ("feature_slot", "scale", "value"),
        )
        self.assertFalse(curve.array.flags.writeable)
        with self.assertRaises(ValueError):
            curve.array[0] = 2.0

    def test_composed_curve_rejects_duplicate_keys_and_invalid_values(self):
        key = CompositeObservationKey("metric_a", 10)
        with self.assertRaisesRegex(ValueError, "unique"):
            ComposedCurve(
                curve_id="composed_a",
                composer_name="dummy",
                observation_keys=(key, key),
                values=(0.1, 0.2),
                source_curve_ids=("curve_a",),
                config_hash="a" * 64,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            ComposedCurve(
                curve_id="composed_a",
                composer_name="dummy",
                observation_keys=(key,),
                values=(float("nan"),),
                source_curve_ids=("curve_a",),
                config_hash="a" * 64,
            )

    def test_composed_pair_requires_identical_keys_and_configuration(self):
        def make_composed(curve_id, feature_slot="metric_a", config_hash="a" * 64):
            return ComposedCurve(
                curve_id=curve_id,
                composer_name="dummy",
                observation_keys=(CompositeObservationKey(feature_slot, 10),),
                values=(0.1,),
                source_curve_ids=(f"source_{curve_id}",),
                config_hash=config_hash,
            )

        with self.assertRaisesRegex(ValueError, "identical observation keys"):
            ComposedAlignedPair(
                reference=make_composed("reference"),
                target=make_composed("target", feature_slot="metric_b"),
            )
        with self.assertRaisesRegex(ValueError, "one composer config"):
            ComposedAlignedPair(
                reference=make_composed("reference"),
                target=make_composed("target", config_hash="b" * 64),
            )

    def test_observation_key_validates_feature_and_scale(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            CompositeObservationKey("", 10)
        with self.assertRaisesRegex(ValueError, "positive"):
            CompositeObservationKey("metric_a", 0)


class ZScoreConcatenationComposerTests(unittest.TestCase):
    @staticmethod
    def make_curve(seed, scales, values):
        query = make_query(seed_type="curve", seed=seed)
        return CurveTable(rows=tuple(
            make_curve_row(
                scale=scale,
                value=value,
                curve_id=query.curve_id,
                seed_type="curve",
                seed=seed,
            )
            for scale, value in zip(scales, values)
        ))

    def test_two_feature_result_matches_baseline_zscore_helper(self):
        from crossscale.compare import _safe_standardize_against_baseline

        scales = (10, 20, 30, 40)
        morph_reference = (0.2, 0.5, 0.9, 1.4)
        morph_target = (0.3, 0.45, 1.1, 1.2)
        script_reference = (0.05, 0.2, 0.4, 0.8)
        script_target = (0.08, 0.18, 0.5, 0.7)
        z_morph_reference, z_morph_target = _safe_standardize_against_baseline(
            np.asarray(morph_reference), np.asarray(morph_target),
        )
        z_script_reference, z_script_target = _safe_standardize_against_baseline(
            np.asarray(script_reference), np.asarray(script_target),
        )
        expected_reference = np.concatenate([z_morph_reference, z_script_reference])
        expected_target = np.concatenate([z_morph_target, z_script_target])
        composer = ZScoreConcatenationComposer(
            feature_slots=("orthographic_similarity", "multi_script_entropy"),
        )

        pair = composer.compose(
            (
                self.make_curve(1, scales, morph_reference),
                self.make_curve(2, scales, script_reference),
            ),
            (
                self.make_curve(3, scales, morph_target),
                self.make_curve(4, scales, script_target),
            ),
        )

        self.assertIsInstance(composer, CurveComposer)
        np.testing.assert_allclose(pair.reference_array, expected_reference)
        np.testing.assert_allclose(pair.target_array, expected_target)
        self.assertEqual(len(pair.observation_keys), 2 * len(scales))
        self.assertEqual(pair.reference.metadata, pair.target.metadata)

    def test_three_features_extend_in_feature_block_order(self):
        scales = (10, 20, 30)
        composer = ZScoreConcatenationComposer(
            feature_slots=("metric_a", "metric_b", "metric_c"),
        )
        references = tuple(
            self.make_curve(seed, scales, values)
            for seed, values in (
                (1, (1.0, 2.0, 4.0)),
                (2, (2.0, 5.0, 9.0)),
                (3, (3.0, 8.0, 12.0)),
            )
        )
        targets = tuple(
            self.make_curve(seed, scales, values)
            for seed, values in (
                (4, (1.5, 2.5, 3.5)),
                (5, (2.5, 4.5, 10.0)),
                (6, (2.0, 9.0, 11.0)),
            )
        )

        pair = composer.compose(references, targets)

        self.assertEqual(len(pair.observation_keys), 9)
        self.assertEqual(
            [key.feature_slot for key in pair.observation_keys],
            ["metric_a"] * 3 + ["metric_b"] * 3 + ["metric_c"] * 3,
        )
        for block_start in (0, 3, 6):
            block = pair.reference_array[block_start:block_start + 3]
            self.assertAlmostEqual(float(np.mean(block)), 0.0)
            self.assertAlmostEqual(float(np.std(block, ddof=0)), 1.0)

    def test_constant_reference_uses_existing_mean_centering_fallback(self):
        scales = (10, 20, 30)
        composer = ZScoreConcatenationComposer(
            feature_slots=("constant", "varying"),
        )
        pair = composer.compose(
            (
                self.make_curve(1, scales, (2.0, 2.0, 2.0)),
                self.make_curve(2, scales, (1.0, 2.0, 3.0)),
            ),
            (
                self.make_curve(3, scales, (1.0, 2.5, 4.0)),
                self.make_curve(4, scales, (1.5, 2.5, 2.0)),
            ),
        )

        np.testing.assert_array_equal(pair.reference_array[:3], np.zeros(3))
        np.testing.assert_array_equal(
            pair.target_array[:3],
            np.array([-1.0, 0.5, 2.0]),
        )

    def test_alignment_policy_is_explicit_across_curves_and_features(self):
        exact = ZScoreConcatenationComposer(
            feature_slots=("metric_a", "metric_b"),
        )
        references = (
            self.make_curve(1, (10, 20, 30), (1.0, 2.0, 3.0)),
            self.make_curve(2, (10, 20), (2.0, 3.0)),
        )
        targets = (
            self.make_curve(3, (10, 20, 30), (1.5, 2.5, 3.5)),
            self.make_curve(4, (10, 20), (2.5, 3.5)),
        )
        with self.assertRaisesRegex(ValueError, "across features"):
            exact.compose(references, targets)

        intersection = ZScoreConcatenationComposer(
            feature_slots=("metric_a", "metric_b"),
            alignment_strategy="intersection",
        )
        pair = intersection.compose(references, targets)
        self.assertEqual(
            pair.observation_keys,
            (
                CompositeObservationKey("metric_a", 10),
                CompositeObservationKey("metric_a", 20),
                CompositeObservationKey("metric_b", 10),
                CompositeObservationKey("metric_b", 20),
            ),
        )

    def test_configuration_and_collection_lengths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            ZScoreConcatenationComposer(feature_slots=("metric_a",))
        with self.assertRaisesRegex(ValueError, "unique"):
            ZScoreConcatenationComposer(
                feature_slots=("metric_a", "metric_a"),
            )
        composer = ZScoreConcatenationComposer(
            feature_slots=("metric_a", "metric_b"),
        )
        curve = self.make_curve(1, (10, 20), (1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "feature_slots length"):
            composer.compose((curve,), (curve,))


class WeightedMeanComposerTests(unittest.TestCase):
    @staticmethod
    def make_curve(seed, scales, values):
        return ZScoreConcatenationComposerTests.make_curve(seed, scales, values)

    def test_two_feature_weighted_mean_uses_normalized_weights(self):
        scales = (10, 20, 30)
        composer = WeightedMeanComposer(
            feature_slots=("metric_a", "metric_b"),
            weights=(1.0, 3.0),
            output_feature_slot="joint_score",
        )
        pair = composer.compose(
            (
                self.make_curve(1, scales, (1.0, 2.0, 3.0)),
                self.make_curve(2, scales, (5.0, 6.0, 7.0)),
            ),
            (
                self.make_curve(3, scales, (2.0, 3.0, 4.0)),
                self.make_curve(4, scales, (6.0, 7.0, 8.0)),
            ),
        )

        self.assertIsInstance(composer, CurveComposer)
        np.testing.assert_allclose(pair.reference_array, (4.0, 5.0, 6.0))
        np.testing.assert_allclose(pair.target_array, (5.0, 6.0, 7.0))
        self.assertEqual(
            pair.observation_keys,
            tuple(
                CompositeObservationKey("joint_score", scale)
                for scale in scales
            ),
        )
        self.assertEqual(composer.normalized_weights, (0.25, 0.75))
        self.assertEqual(pair.reference.metadata["normalization"], "none")

    def test_three_features_support_zero_weight_and_arbitrary_scale_order(self):
        scales = (30, 10, 20)
        composer = WeightedMeanComposer(
            feature_slots=("metric_a", "metric_b", "metric_c"),
            weights=(2.0, 0.0, 2.0),
        )
        pair = composer.compose(
            (
                self.make_curve(1, scales, (3.0, 1.0, 2.0)),
                self.make_curve(2, scales, (300.0, 100.0, 200.0)),
                self.make_curve(3, scales, (9.0, 3.0, 6.0)),
            ),
            (
                self.make_curve(4, scales, (6.0, 2.0, 4.0)),
                self.make_curve(5, scales, (600.0, 200.0, 400.0)),
                self.make_curve(6, scales, (12.0, 4.0, 8.0)),
            ),
        )

        self.assertEqual(
            tuple(key.scale for key in pair.observation_keys),
            (10, 20, 30),
        )
        np.testing.assert_allclose(pair.reference_array, (2.0, 4.0, 6.0))
        np.testing.assert_allclose(pair.target_array, (3.0, 6.0, 9.0))

    def test_exact_and_intersection_scale_policies_are_explicit(self):
        references = (
            self.make_curve(1, (10, 20, 30), (1.0, 2.0, 3.0)),
            self.make_curve(2, (10, 20), (3.0, 4.0)),
        )
        targets = (
            self.make_curve(3, (10, 20, 30), (2.0, 3.0, 4.0)),
            self.make_curve(4, (10, 20), (4.0, 5.0)),
        )
        exact = WeightedMeanComposer(
            feature_slots=("metric_a", "metric_b"),
            weights=(1.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "across features"):
            exact.compose(references, targets)

        intersection = WeightedMeanComposer(
            feature_slots=("metric_a", "metric_b"),
            weights=(1.0, 1.0),
            alignment_strategy="intersection",
        )
        pair = intersection.compose(references, targets)
        self.assertEqual(
            tuple(key.scale for key in pair.observation_keys),
            (10, 20),
        )

    def test_invalid_weights_fail_before_composition(self):
        invalid_cases = (
            ((1.0,), "match"),
            ((1.0, -1.0), "non-negative"),
            ((0.0, 0.0), "positive"),
            ((1.0, float("nan")), "finite"),
        )
        for weights, message in invalid_cases:
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, message):
                    WeightedMeanComposer(
                        feature_slots=("metric_a", "metric_b"),
                        weights=weights,
                    )


class ComposedStatisticsPipelineTests(unittest.TestCase):
    @staticmethod
    def make_inputs():
        reference_queries = (
            make_query(
                metric="orthographic_similarity",
                component="score",
                aggregation="mean",
            ),
            make_query(
                metric="multi_script_entropy",
                component="normalized",
                aggregation="mean",
            ),
        )
        target_queries = tuple(
            make_query(
                source="random_projection",
                seed_type="projection",
                seed=42,
                metric=query.metric,
                component=query.component,
                aggregation=query.aggregation,
            )
            for query in reference_queries
        )
        rows = []
        for scale, values in (
            (10, (0.2, 0.05, 0.3, 0.08)),
            (20, (0.5, 0.2, 0.45, 0.18)),
            (30, (0.9, 0.4, 1.1, 0.5)),
            (40, (1.4, 0.8, 1.2, 0.7)),
        ):
            morph_reference, script_reference, morph_target, script_target = values
            rows.extend((
                make_metric_row(
                    scale=scale,
                    metric="orthographic_similarity",
                    component="score",
                    aggregation="mean",
                    value=morph_reference,
                ),
                make_metric_row(
                    scale=scale,
                    metric="multi_script_entropy",
                    component="normalized",
                    aggregation="mean",
                    value=script_reference,
                ),
                make_metric_row(
                    scale=scale,
                    task=make_task(
                        scale=scale,
                        source="random_projection",
                        seed_type="projection",
                        seed=42,
                    ),
                    metric="orthographic_similarity",
                    component="score",
                    aggregation="mean",
                    value=morph_target,
                ),
                make_metric_row(
                    scale=scale,
                    task=make_task(
                        scale=scale,
                        source="random_projection",
                        seed_type="projection",
                        seed=42,
                    ),
                    metric="multi_script_entropy",
                    component="normalized",
                    aggregation="mean",
                    value=script_target,
                ),
            ))
        return MetricResultTable(rows=tuple(rows)), reference_queries, target_queries

    def test_zscore_composer_runs_spearman_and_mae_in_one_dispatch(self):
        metric_results, reference_queries, target_queries = self.make_inputs()
        pipeline = ComposedStatisticsPipeline(
            composer=ZScoreConcatenationComposer(
                feature_slots=("orthographic_similarity", "multi_script_entropy"),
            ),
            statistics=(SpearmanStatistic(), MAEStatistic()),
        )

        run = pipeline.run(
            metric_results,
            reference_queries=reference_queries,
            target_queries=target_queries,
        )

        self.assertEqual(len(run.reference_curves.curve_ids), 2)
        self.assertEqual(len(run.target_curves.curve_ids), 2)
        self.assertEqual(len(run.composed_pair.observation_keys), 8)
        self.assertEqual(len(run.results.rows), 3)
        self.assertEqual(
            {(row.statistic, row.component) for row in run.results.rows},
            {("spearman", "rho"), ("spearman", "p_value"), ("mae", "mae")},
        )
        expected_spearman = SpearmanStatistic().compute(
            run.composed_pair.reference_array,
            run.composed_pair.target_array,
        )
        self.assertEqual(
            next(row.value for row in run.results.rows if row.component == "rho"),
            expected_spearman.value_for("rho"),
        )
        self.assertTrue(all(
            row.reference_curve_id == run.composed_pair.reference.curve_id
            and row.target_curve_id == run.composed_pair.target.curve_id
            and row.n_points == 8
            for row in run.results.rows
        ))

    def test_composer_is_swappable_without_dispatch_changes(self):
        metric_results, reference_queries, target_queries = self.make_inputs()
        pipeline = ComposedStatisticsPipeline(
            composer=WeightedMeanComposer(
                feature_slots=("orthographic_similarity", "multi_script_entropy"),
                weights=(1.0, 1.0),
            ),
            statistics=(MAEStatistic(),),
        )

        run = pipeline.run(
            metric_results,
            reference_queries=reference_queries,
            target_queries=target_queries,
        )

        self.assertEqual(len(run.composed_pair.observation_keys), 4)
        self.assertEqual(run.results.rows[0].n_points, 4)
        self.assertEqual(run.results.rows[0].statistic, "mae")

    def test_query_collections_and_adapter_names_are_validated(self):
        metric_results, reference_queries, target_queries = self.make_inputs()
        pipeline = ComposedStatisticsPipeline(
            composer=ZScoreConcatenationComposer(
                feature_slots=("orthographic_similarity", "multi_script_entropy"),
            ),
            statistics=(MAEStatistic(),),
        )
        with self.assertRaisesRegex(ValueError, "equal length"):
            pipeline.run(
                metric_results,
                reference_queries=reference_queries,
                target_queries=target_queries[:1],
            )
        with self.assertRaisesRegex(ValueError, "unique names"):
            ComposedStatisticsPipeline(
                composer=WeightedMeanComposer(
                    feature_slots=("metric_a", "metric_b"),
                    weights=(1.0, 1.0),
                ),
                statistics=(MAEStatistic(), MAEStatistic()),
            )


class PairwiseStatisticContractTests(unittest.TestCase):
    def test_dummy_adapter_satisfies_protocol_and_computes(self):
        statistic = DummyMeanDifferenceStatistic()
        self.assertIsInstance(statistic, PairwiseStatistic)

        result = statistic.compute(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
        )

        self.assertEqual(result.statistic, "mean_difference")
        self.assertEqual(result.n_points, 3)
        self.assertEqual(result.value_for("difference"), 2.0)
        self.assertEqual(result.metadata["direction"], "target_minus_reference")

    def test_observation_rejects_invalid_identity_or_value(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            StatisticObservation(component="", value=1.0)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    StatisticObservation(component="coefficient", value=value)

    def test_result_rejects_duplicate_components(self):
        observation = StatisticObservation("coefficient", 0.5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            StatisticResult(
                statistic="dummy",
                observations=(observation, observation),
                n_points=3,
            )

    def test_result_validates_point_count_and_component_lookup(self):
        result = StatisticResult(
            statistic="dummy",
            observations=(StatisticObservation("coefficient", 0.5),),
            n_points=3,
        )
        with self.assertRaises(KeyError):
            result.value_for("missing")
        with self.assertRaisesRegex(ValueError, "positive"):
            StatisticResult(
                statistic="dummy",
                observations=(StatisticObservation("coefficient", 0.5),),
                n_points=0,
            )


class RealStatisticAdapterTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([0.2, 0.8, 0.4, 1.1, 0.6], dtype=float)
        self.target = np.array([0.1, 0.7, 0.5, 1.3, 0.55], dtype=float)

    def test_spearman_matches_existing_helper_and_scipy(self):
        from crossscale.compare import compute_spearman_stats

        expected_legacy = compute_spearman_stats(self.reference, self.target)
        expected_scipy = spearmanr(self.reference, self.target)
        result = SpearmanStatistic().compute(self.reference, self.target)

        self.assertAlmostEqual(result.value_for("rho"), expected_legacy.rho)
        self.assertAlmostEqual(result.value_for("p_value"), expected_legacy.pvalue)
        self.assertAlmostEqual(result.value_for("rho"), float(expected_scipy.statistic))
        self.assertAlmostEqual(result.value_for("p_value"), float(expected_scipy.pvalue))
        self.assertEqual(result.n_points, len(self.reference))

    def test_pearson_matches_existing_helper_and_scipy(self):
        from crossscale.pearson import compute_pearson_stats

        expected_legacy = compute_pearson_stats(self.reference, self.target)
        expected_scipy = pearsonr(self.reference, self.target)
        result = PearsonStatistic().compute(self.reference, self.target)

        self.assertAlmostEqual(result.value_for("r"), expected_legacy.r)
        self.assertAlmostEqual(result.value_for("p_value"), expected_legacy.pvalue)
        self.assertAlmostEqual(result.value_for("r"), float(expected_scipy.statistic))
        self.assertAlmostEqual(result.value_for("p_value"), float(expected_scipy.pvalue))
        self.assertEqual(result.n_points, len(self.reference))

    def test_mae_matches_direct_notebook_definition(self):
        expected = float(np.mean(np.abs(self.target - self.reference)))
        result = MAEStatistic().compute(self.reference, self.target)
        self.assertEqual(result.value_for("mae"), expected)
        self.assertEqual(result.n_points, len(self.reference))

    def test_r2_matches_notebook_formula_and_sklearn_direction(self):
        residual_sum = float(np.sum((self.target - self.reference) ** 2))
        total_sum = float(np.sum((self.reference - self.reference.mean()) ** 2))
        expected_notebook = 1.0 - residual_sum / total_sum
        result = R2Statistic().compute(self.reference, self.target)

        self.assertEqual(result.value_for("r2"), expected_notebook)
        self.assertEqual(
            result.value_for("r2"),
            float(r2_score(self.reference, self.target)),
        )
        reverse = R2Statistic().compute(self.target, self.reference)
        self.assertNotEqual(result.value_for("r2"), reverse.value_for("r2"))

    def test_all_adapters_satisfy_pairwise_protocol(self):
        for statistic in (
            SpearmanStatistic(),
            PearsonStatistic(),
            MAEStatistic(),
            R2Statistic(),
        ):
            with self.subTest(statistic=statistic.name):
                self.assertIsInstance(statistic, PairwiseStatistic)
                self.assertTrue(statistic.config())

    def test_invalid_shapes_sizes_and_values_fail_explicitly(self):
        for statistic in (
            SpearmanStatistic(),
            PearsonStatistic(),
            MAEStatistic(),
            R2Statistic(),
        ):
            with self.subTest(statistic=statistic.name, case="shape"):
                with self.assertRaisesRegex(ValueError, "share a shape"):
                    statistic.compute(np.array([1.0, 2.0]), np.array([1.0]))
            with self.subTest(statistic=statistic.name, case="finite"):
                with self.assertRaisesRegex(ValueError, "finite"):
                    statistic.compute(
                        np.array([1.0, float("nan")]),
                        np.array([1.0, 2.0]),
                    )

    def test_undefined_correlations_and_r2_fail_explicitly(self):
        constant = np.array([1.0, 1.0, 1.0])
        varying = np.array([1.0, 2.0, 3.0])
        for statistic in (SpearmanStatistic(), PearsonStatistic()):
            with self.subTest(statistic=statistic.name):
                with self.assertRaisesRegex(ValueError, "undefined"):
                    statistic.compute(constant, varying)
        with self.assertRaisesRegex(ValueError, "constant reference"):
            R2Statistic().compute(constant, varying)


class StatisticResultTableTests(unittest.TestCase):
    def make_aligned(self):
        return AlignedCurvePair(
            reference_curve_id="reference_curve",
            target_curve_id="target_curve",
            scales=(10, 20, 30),
            reference_values=(1.0, 2.0, 3.0),
            target_values=(1.1, 1.9, 3.2),
            strategy="exact",
        )

    def test_maps_multi_component_result_to_long_form(self):
        aligned = self.make_aligned()
        result = SpearmanStatistic().compute(
            aligned.reference_array,
            aligned.target_array,
        )

        table = StatisticResultTable.from_result(
            aligned,
            result,
            config_hash="a" * 64,
        )

        self.assertEqual([row.component for row in table.rows], ["rho", "p_value"])
        self.assertTrue(all(row.n_points == 3 for row in table.rows))
        self.assertTrue(all(row.reference_curve_id == "reference_curve" for row in table.rows))
        self.assertEqual(tuple(table.to_frame().columns), STATISTIC_RESULT_COLUMNS)

    def test_csv_round_trip_preserves_direction_and_values(self):
        table = StatisticResultTable.from_result(
            self.make_aligned(),
            MAEStatistic().compute(
                self.make_aligned().reference_array,
                self.make_aligned().target_array,
            ),
            config_hash="b" * 64,
            reused=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "statistics.csv"
            table.write_csv(path)
            restored = StatisticResultTable.read_csv(path)
        self.assertEqual(restored, table)
        self.assertEqual(restored.rows[0].reference_curve_id, "reference_curve")
        self.assertEqual(restored.rows[0].target_curve_id, "target_curve")

    def test_direction_is_part_of_row_identity(self):
        first = StatisticResultRow(
            reference_curve_id="a",
            target_curve_id="b",
            alignment_strategy="exact",
            statistic="r2",
            component="r2",
            value=0.8,
            n_points=3,
            config_hash="a" * 64,
        )
        reverse = StatisticResultRow(
            reference_curve_id="b",
            target_curve_id="a",
            alignment_strategy="exact",
            statistic="r2",
            component="r2",
            value=0.7,
            n_points=3,
            config_hash="a" * 64,
        )
        table = StatisticResultTable(rows=(first, reverse))
        self.assertEqual(len(table.rows), 2)

    def test_rejects_duplicate_components_and_point_mismatch(self):
        row = StatisticResultRow(
            reference_curve_id="a",
            target_curve_id="b",
            alignment_strategy="exact",
            statistic="mae",
            component="mae",
            value=0.1,
            n_points=3,
            config_hash="a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            StatisticResultTable(rows=(row, row))

        aligned = self.make_aligned()
        mismatched = StatisticResult(
            statistic="dummy",
            observations=(StatisticObservation("value", 1.0),),
            n_points=2,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            StatisticResultTable.from_result(
                aligned,
                mismatched,
                config_hash="a" * 64,
            )

    def test_from_frame_requires_canonical_columns(self):
        frame = StatisticResultTable.from_result(
            self.make_aligned(),
            MAEStatistic().compute(
                self.make_aligned().reference_array,
                self.make_aligned().target_array,
            ),
            config_hash="a" * 64,
        ).to_frame().drop(columns=["reused"])
        with self.assertRaisesRegex(ValueError, "canonical"):
            StatisticResultTable.from_frame(frame)

    def test_write_refuses_silent_overwrite(self):
        table = StatisticResultTable.from_result(
            self.make_aligned(),
            MAEStatistic().compute(
                self.make_aligned().reference_array,
                self.make_aligned().target_array,
            ),
            config_hash="a" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "statistics.csv"
            table.write_csv(path)
            with self.assertRaises(FileExistsError):
                table.write_csv(path)


class StatisticsPipelineTests(unittest.TestCase):
    @staticmethod
    def make_three_curve_inputs(scales=(10, 20, 30)):
        queries = (
            make_query(
                metric="orthographic_similarity",
                component="score",
                aggregation="mean",
            ),
            make_query(
                metric="multi_script_entropy",
                component="normalized",
                aggregation="mean",
            ),
            make_query(
                metric="orthographic_similarity",
                component="score",
                aggregation="std",
            ),
        )
        rows = []
        for scale in scales:
            rows.extend((
                make_metric_row(
                    scale=scale,
                    metric="orthographic_similarity",
                    component="score",
                    aggregation="mean",
                    value=scale / 100.0,
                ),
                make_metric_row(
                    scale=scale,
                    metric="multi_script_entropy",
                    component="normalized",
                    aggregation="mean",
                    value=scale / 120.0 + 0.01,
                ),
                make_metric_row(
                    scale=scale,
                    metric="orthographic_similarity",
                    component="score",
                    aggregation="std",
                    value=scale / 90.0 + 0.03,
                ),
            ))
        return MetricResultTable(rows=tuple(rows)), queries

    def test_all_pairs_generates_three_pairs_for_three_curves(self):
        metric_results, queries = self.make_three_curve_inputs()
        pipeline = StatisticsPipeline(
            statistics=(SpearmanStatistic(), MAEStatistic()),
        )

        run = pipeline.run(metric_results, queries)

        self.assertEqual(PAIR_MODES, ("all_pairs", "reference_only"))
        self.assertEqual(len(run.curves.curve_ids), 3)
        self.assertEqual(len(run.alignments), 3)
        self.assertEqual(len(run.results.rows), 9)
        pair_ids = {
            (row.reference_curve_id, row.target_curve_id)
            for row in run.results.rows
        }
        self.assertEqual(len(pair_ids), 3)

    def test_reference_only_uses_requested_curve_as_first_argument(self):
        metric_results, queries = self.make_three_curve_inputs()
        run = StatisticsPipeline(
            statistics=(R2Statistic(),),
            pair_mode="reference_only",
            reference_index=1,
        ).run(metric_results, queries)

        self.assertEqual(len(run.alignments), 2)
        expected_reference = queries[1].curve_id
        self.assertTrue(all(
            row.reference_curve_id == expected_reference
            for row in run.results.rows
        ))

    def test_arbitrary_dummy_statistic_runs_without_pipeline_changes(self):
        metric_results, queries = self.make_three_curve_inputs()
        run = StatisticsPipeline(
            statistics=(DummyMeanDifferenceStatistic(),),
            pair_mode="reference_only",
        ).run(metric_results, queries)

        self.assertEqual(len(run.results.rows), 2)
        self.assertTrue(all(
            row.statistic == "mean_difference"
            for row in run.results.rows
        ))

    def test_exact_alignment_failure_is_not_silently_dropped(self):
        metric_results, queries = self.make_three_curve_inputs()
        filtered_rows = tuple(
            row
            for row in metric_results.rows
            if not (
                row.metric == "multi_script_entropy"
                and row.task.scale == 20
            )
        )
        with self.assertRaisesRegex(ValueError, "identical scales"):
            StatisticsPipeline(statistics=(MAEStatistic(),)).run(
                MetricResultTable(rows=filtered_rows),
                queries,
            )

    def test_pipeline_validates_modes_queries_and_unique_statistics(self):
        self.assertTrue(StatisticsPipeline(statistics=(MAEStatistic(),)).config())
        with self.assertRaisesRegex(ValueError, "pair_mode"):
            StatisticsPipeline(statistics=(MAEStatistic(),), pair_mode="implicit")
        with self.assertRaisesRegex(ValueError, "unique names"):
            StatisticsPipeline(statistics=(MAEStatistic(), MAEStatistic()))

        metric_results, queries = self.make_three_curve_inputs()
        with self.assertRaisesRegex(ValueError, "at least two"):
            StatisticsPipeline(statistics=(MAEStatistic(),)).run(
                metric_results,
                queries[:1],
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            StatisticsPipeline(
                statistics=(MAEStatistic(),),
                pair_mode="reference_only",
                reference_index=5,
            ).run(metric_results, queries)


class CorrelationMatrixBuilderTests(unittest.TestCase):
    def make_spearman_run(self):
        metric_results, queries = StatisticsPipelineTests.make_three_curve_inputs()
        run = StatisticsPipeline(
            statistics=(SpearmanStatistic(),),
            pair_mode="all_pairs",
        ).run(metric_results, queries)
        return run, queries

    def test_builds_symmetric_three_curve_matrix_from_pipeline_results(self):
        run, queries = self.make_spearman_run()

        matrix = CorrelationMatrixBuilder(
            statistic="spearman",
            component="rho",
        ).build(run.results)
        frame = matrix.to_frame()

        self.assertIsInstance(matrix, CorrelationMatrix)
        self.assertEqual(matrix.curve_ids, tuple(query.curve_id for query in queries))
        np.testing.assert_array_equal(np.diag(frame.to_numpy()), np.ones(3))
        np.testing.assert_array_equal(frame.to_numpy(), frame.to_numpy().T)
        for row in run.results.rows:
            if row.component == "rho":
                self.assertEqual(
                    frame.loc[row.reference_curve_id, row.target_curve_id],
                    row.value,
                )

    def test_explicit_curve_order_controls_rows_and_columns(self):
        run, queries = self.make_spearman_run()
        requested_order = tuple(query.curve_id for query in reversed(queries))

        matrix = CorrelationMatrixBuilder(
            statistic="spearman",
            component="rho",
        ).build(run.results, curve_ids=requested_order)

        self.assertEqual(matrix.curve_ids, requested_order)
        self.assertEqual(tuple(matrix.to_frame().index), requested_order)
        self.assertEqual(tuple(matrix.to_frame().columns), requested_order)

    def test_missing_pair_is_rejected(self):
        run, _ = self.make_spearman_run()
        selected = tuple(
            row for row in run.results.rows
            if row.component == "rho"
        )
        incomplete = StatisticResultTable(rows=selected[:-1])

        with self.assertRaisesRegex(ValueError, "complete all-pairs"):
            CorrelationMatrixBuilder(
                statistic="spearman",
                component="rho",
            ).build(incomplete)

    def test_reverse_duplicate_pair_is_rejected(self):
        run, _ = self.make_spearman_run()
        selected = tuple(
            row for row in run.results.rows
            if row.component == "rho"
        )
        first = selected[0]
        reverse = StatisticResultRow(
            reference_curve_id=first.target_curve_id,
            target_curve_id=first.reference_curve_id,
            alignment_strategy=first.alignment_strategy,
            statistic=first.statistic,
            component=first.component,
            value=first.value,
            n_points=first.n_points,
            config_hash=first.config_hash,
        )

        with self.assertRaisesRegex(ValueError, "duplicate unordered pair"):
            CorrelationMatrixBuilder(
                statistic="spearman",
                component="rho",
            ).build(StatisticResultTable(rows=selected + (reverse,)))

    def test_selection_and_provenance_must_be_unambiguous(self):
        run, queries = self.make_spearman_run()
        builder = CorrelationMatrixBuilder(statistic="spearman", component="rho")
        with self.assertRaisesRegex(ValueError, "matched"):
            CorrelationMatrixBuilder(
                statistic="pearson",
                component="r",
            ).build(run.results)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            builder.build(run.results, curve_ids=tuple(
                query.curve_id for query in queries[:2]
            ))

        rho_rows = tuple(row for row in run.results.rows if row.component == "rho")
        conflicting = StatisticResultRow(
            **{
                **rho_rows[-1].to_dict(),
                "config_hash": "f" * 64,
            }
        )
        with self.assertRaisesRegex(ValueError, "one alignment strategy"):
            builder.build(StatisticResultTable(rows=rho_rows[:-1] + (conflicting,)))


class CurveQueryTests(unittest.TestCase):
    def test_identity_is_stable_and_serializes_all_query_fields(self):
        first = make_query()
        second = make_query()

        self.assertEqual(first.curve_id, second.curve_id)
        self.assertEqual(
            tuple(first.to_dict()),
            (
                "analysis_id",
                "model_name",
                "source",
                "seed_type",
                "seed",
                "metric",
                "component",
                "aggregation",
                "scope",
                "space_name",
                "partition_method",
            ),
        )

    def test_seed_fields_must_be_complete(self):
        with self.assertRaisesRegex(ValueError, "both be set"):
            make_query(seed_type="projection")
        with self.assertRaisesRegex(ValueError, "both be set"):
            make_query(seed=42)
        self.assertEqual(
            make_query(seed_type="projection", seed=42).seed,
            42,
        )

    def test_selects_and_sorts_one_curve(self):
        rows = (
            make_metric_row(scale=30),
            make_metric_row(scale=10),
            make_metric_row(
                scale=20,
                metric="multi_script_entropy",
                component="normalized",
            ),
            make_metric_row(scale=20),
        )

        curve = make_query().select(MetricResultTable(rows=rows))

        self.assertEqual([row.scale for row in curve.rows], [10, 20, 30])
        self.assertEqual([row.value for row in curve.rows], [0.1, 0.2, 0.3])
        self.assertEqual(curve.curve_ids, (make_query().curve_id,))

    def test_seed_query_does_not_match_unseeded_rows(self):
        seeded_query = make_query(seed_type="projection", seed=42)
        with self.assertRaisesRegex(ValueError, "matched no"):
            seeded_query.select(MetricResultTable(rows=(make_metric_row(),)))

    def test_empty_and_ambiguous_queries_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "matched no"):
            make_query(metric="missing").select(
                MetricResultTable(rows=(make_metric_row(),))
            )

        ambiguous = MetricResultTable(rows=(
            make_metric_row(scale=10),
            make_metric_row(
                scale=20,
                task=make_task(scale=20, space_name="input_embd"),
            ),
        ))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            make_query(space_name=None).select(ambiguous)

    def test_duplicate_scales_across_distinct_tasks_fail_explicitly(self):
        rows = MetricResultTable(rows=(
            make_metric_row(scale=10),
            make_metric_row(
                scale=10,
                task=make_task(scale=10, space_name="input_embd"),
            ),
        ))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            make_query(space_name=None).select(rows)


class CurveTableTests(unittest.TestCase):
    def test_canonical_frame_and_csv_round_trip(self):
        table = CurveTable(rows=(
            make_curve_row(scale=10),
            make_curve_row(
                scale=20,
                seed_type="projection",
                seed=42,
                curve_id=make_query(seed_type="projection", seed=42).curve_id,
            ),
        ))
        frame = table.to_frame()
        self.assertEqual(tuple(frame.columns), CURVE_TABLE_COLUMNS)
        self.assertEqual(str(frame["seed"].dtype), "Int64")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.csv"
            table.write_csv(path)
            restored = CurveTable.read_csv(path)
        self.assertEqual(restored, table)

    def test_rejects_duplicate_curve_scale_identity(self):
        row = make_curve_row()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CurveTable(rows=(row, row))

    def test_rejects_conflicting_semantics_for_one_curve_id(self):
        first = make_curve_row(scale=10)
        second = make_curve_row(
            scale=20,
            model_name="model_b",
            curve_id=first.curve_id,
        )
        with self.assertRaisesRegex(ValueError, "multiple curve semantics"):
            CurveTable(rows=(first, second))

    def test_combine_supports_multiple_curves(self):
        first = CurveTable(rows=(make_curve_row(scale=10),))
        seeded_query = make_query(seed_type="projection", seed=42)
        second = CurveTable(rows=(make_curve_row(
            scale=10,
            curve_id=seeded_query.curve_id,
            seed_type="projection",
            seed=42,
        ),))

        combined = CurveTable.combine(first, second)

        self.assertEqual(combined.curve_ids, (first.rows[0].curve_id, seeded_query.curve_id))
        self.assertEqual(len(combined.rows), 2)

    def test_from_frame_requires_canonical_columns(self):
        frame = CurveTable(rows=(make_curve_row(),)).to_frame().drop(columns=["scope"])
        with self.assertRaisesRegex(ValueError, "canonical"):
            CurveTable.from_frame(frame)

    def test_write_refuses_silent_overwrite(self):
        table = CurveTable(rows=(make_curve_row(),))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.csv"
            table.write_csv(path)
            with self.assertRaises(FileExistsError):
                table.write_csv(path)


class CurveAlignerTests(unittest.TestCase):
    @staticmethod
    def make_curve(curve_seed, scales):
        query = make_query(seed_type="curve", seed=curve_seed)
        return CurveTable(rows=tuple(
            make_curve_row(
                scale=scale,
                value=scale + curve_seed / 10.0,
                curve_id=query.curve_id,
                seed_type="curve",
                seed=curve_seed,
            )
            for scale in scales
        ))

    def test_exact_is_default_and_sorts_scales(self):
        reference = self.make_curve(1, (30, 10, 20))
        target = self.make_curve(2, (20, 30, 10))

        aligned = CurveAligner().align(reference, target)

        self.assertEqual(aligned.strategy, "exact")
        self.assertEqual(aligned.scales, (10, 20, 30))
        self.assertEqual(aligned.reference_values, (10.1, 20.1, 30.1))
        self.assertEqual(aligned.target_values, (10.2, 20.2, 30.2))
        self.assertEqual(
            tuple(aligned.to_frame().columns),
            ("scale", "reference_value", "target_value"),
        )

    def test_exact_rejects_any_scale_mismatch(self):
        reference = self.make_curve(1, (10, 20, 30))
        target = self.make_curve(2, (10, 30, 40))
        with self.assertRaisesRegex(
            ValueError,
            r"missing_from_target=\[20\].*extra_in_target=\[40\]",
        ):
            CurveAligner("exact").align(reference, target)

    def test_intersection_only_drops_scales_when_explicit(self):
        reference = self.make_curve(1, (10, 20, 30))
        target = self.make_curve(2, (20, 30, 40))

        aligned = CurveAligner("intersection").align(reference, target)

        self.assertEqual(aligned.scales, (20, 30))
        self.assertEqual(aligned.reference_values, (20.1, 30.1))
        self.assertEqual(aligned.target_values, (20.2, 30.2))

    def test_intersection_rejects_disjoint_curves(self):
        with self.assertRaisesRegex(ValueError, "no common scales"):
            CurveAligner("intersection").align(
                self.make_curve(1, (10, 20)),
                self.make_curve(2, (30, 40)),
            )

    def test_reference_allows_only_target_extras(self):
        aligned = CurveAligner("reference").align(
            self.make_curve(1, (10, 30)),
            self.make_curve(2, (10, 20, 30)),
        )
        self.assertEqual(aligned.scales, (10, 30))

        with self.assertRaisesRegex(ValueError, "target to cover"):
            CurveAligner("reference").align(
                self.make_curve(1, (10, 20, 30)),
                self.make_curve(2, (10, 30)),
            )

    def test_rejects_empty_or_multi_curve_tables(self):
        one = self.make_curve(1, (10, 20))
        two = self.make_curve(2, (10, 20))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            CurveAligner().align(CurveTable(), one)
        with self.assertRaisesRegex(ValueError, "exactly one curve"):
            CurveAligner().align(CurveTable.combine(one, two), one)

    def test_contract_validates_strategy_and_lengths(self):
        self.assertEqual(ALIGNMENT_STRATEGIES, ("exact", "intersection", "reference"))
        with self.assertRaisesRegex(ValueError, "strategy"):
            CurveAligner("implicit")
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            AlignedCurvePair(
                reference_curve_id="a",
                target_curve_id="b",
                scales=(10, 20),
                reference_values=(0.1,),
                target_values=(0.2, 0.3),
                strategy="exact",
            )
