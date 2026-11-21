import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from crossscale.clustering_pipeline import (
    FullPCAPrefixProvider,
    HDBSCANAdapter,
    IdentityTransform,
    InMemoryRepresentationProvider,
    ClusteringOutputLayout,
    ClusteringPipeline,
    MatrixTransform,
    PARTITION_MANIFEST_COLUMNS,
    PartitionManifest,
    PartitionManifestRow,
    PartitionMethod,
    PartitionResult,
    RepresentationProvider,
    RepresentationResult,
    RowL2Normalizer,
    SavedPerScaleProvider,
    default_saved_per_scale_path,
    load_partition_result,
    validate_partition_output,
    write_partition_output,
)
from crossscale.pipeline_core import RunPolicy, ScaleTask


def make_task(**overrides):
    values = {
        "analysis_id": "validation",
        "model_name": "organization/model",
        "space_name": "output_proj",
        "scale": 2,
        "source": "full_pca_prefix",
        "seed_type": None,
        "seed": None,
    }
    values.update(overrides)
    return ScaleTask(**values)


class RepresentationResultTests(unittest.TestCase):
    def test_accepts_matching_two_dimensional_matrix(self):
        task = make_task(scale=3)
        result = RepresentationResult(task=task, matrix=np.ones((4, 3)))
        self.assertEqual(result.matrix.shape, (4, 3))

    def test_rejects_non_array_or_wrong_shape(self):
        with self.assertRaisesRegex(TypeError, "NumPy array"):
            RepresentationResult(task=make_task(), matrix=[[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            RepresentationResult(task=make_task(), matrix=np.ones(2))
        with self.assertRaisesRegex(ValueError, "does not match task scale"):
            RepresentationResult(task=make_task(scale=3), matrix=np.ones((4, 2)))


class FullPCAPrefixProviderTests(unittest.TestCase):
    def test_returns_exact_prefix_and_caches_one_load_per_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task_small = make_task(scale=2)
            task_large = make_task(scale=4)
            matrix_path = (
                Path(temp_dir)
                / task_small.model_name
                / "full_pca"
                / "X_pca.npy"
            )
            matrix_path.parent.mkdir(parents=True)
            full_matrix = np.arange(30, dtype=np.float32).reshape(5, 6)
            np.save(matrix_path, full_matrix)

            load_count = 0

            def counting_loader(path, **kwargs):
                nonlocal load_count
                load_count += 1
                return np.load(path, **kwargs)

            provider = FullPCAPrefixProvider(
                temp_dir,
                mmap_mode=None,
                array_loader=counting_loader,
            )
            small = provider.load(task_small)
            large = provider.load(task_large)

            np.testing.assert_array_equal(small.matrix, full_matrix[:, :2])
            np.testing.assert_array_equal(large.matrix, full_matrix[:, :4])
            self.assertEqual(load_count, 1)
            self.assertTrue(small.metadata["is_prefix_view"])
            self.assertIsInstance(provider, RepresentationProvider)

    def test_rejects_scale_wider_than_full_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task(scale=4)
            matrix_path = (
                Path(temp_dir) / task.model_name / "full_pca" / "X_pca.npy"
            )
            matrix_path.parent.mkdir(parents=True)
            np.save(matrix_path, np.ones((5, 3), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "exceeds full PCA width"):
                FullPCAPrefixProvider(temp_dir).load(task)

    def test_missing_full_matrix_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Full PCA matrix not found"):
                FullPCAPrefixProvider(temp_dir).load(make_task())

    def test_clear_cache_forces_next_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task()
            matrix_path = (
                Path(temp_dir) / task.model_name / "full_pca" / "X_pca.npy"
            )
            matrix_path.parent.mkdir(parents=True)
            np.save(matrix_path, np.ones((3, 2), dtype=np.float32))
            provider = FullPCAPrefixProvider(temp_dir, mmap_mode=None)

            first = provider.load(task).matrix
            provider.clear_cache()
            second = provider.load(task).matrix

            self.assertIsNot(first, second)


class SavedPerScaleProviderTests(unittest.TestCase):
    def test_default_layout_matches_existing_randomized_pca_layout(self):
        task = make_task(
            scale=16,
            source="randomized_pca",
            seed_type="pca",
            seed=42,
        )
        path = default_saved_per_scale_path(Path("comp"), task)
        self.assertEqual(
            path,
            Path("comp/organization/model/random_pca/seed_00042/pca_dim_16/X_pca.npy"),
        )

    def test_custom_resolver_loads_scale_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task(source="custom_saved")
            matrix = np.arange(8, dtype=np.float32).reshape(4, 2)
            path = Path(temp_dir) / "custom" / f"scale_{task.scale}.npy"
            path.parent.mkdir(parents=True)
            np.save(path, matrix)

            def resolver(root, current_task):
                return root / "custom" / f"scale_{current_task.scale}.npy"

            result = SavedPerScaleProvider(
                temp_dir,
                path_resolver=resolver,
                mmap_mode=None,
            ).load(task)

            np.testing.assert_array_equal(result.matrix, matrix)
            self.assertEqual(result.metadata["source_path"], str(path))

    def test_default_layout_requires_seed(self):
        with self.assertRaisesRegex(ValueError, "requires a seed"):
            default_saved_per_scale_path(Path("comp"), make_task())


class InMemoryRepresentationProviderTests(unittest.TestCase):
    def test_loads_by_task_or_explicit_task_key(self):
        first_task = make_task(scale=2)
        second_task = make_task(scale=3)
        first_matrix = np.ones((4, 2))
        second_matrix = np.ones((4, 3)) * 2
        provider = InMemoryRepresentationProvider(
            {
                first_task: first_matrix,
                second_task.task_key: second_matrix,
            }
        )

        self.assertIs(provider.load(first_task).matrix, first_matrix)
        self.assertIs(provider.load(second_task).matrix, second_matrix)

    def test_missing_task_is_explicit(self):
        with self.assertRaisesRegex(KeyError, "No in-memory representation"):
            InMemoryRepresentationProvider({}).load(make_task())


class MatrixTransformTests(unittest.TestCase):
    def test_identity_returns_exact_input_object(self):
        task = make_task(scale=2)
        matrix = np.arange(8, dtype=np.float64).reshape(4, 2)
        transform = IdentityTransform()

        self.assertIs(transform.transform(matrix, task), matrix)
        self.assertEqual(transform.config(), {"name": "identity"})
        self.assertIsInstance(transform, MatrixTransform)

    def test_row_l2_matches_existing_function_elementwise(self):
        from crossscale.pca_hdbscan import apply_l2_normalization

        task = make_task(scale=3)
        matrix = np.array(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 2.0]],
            dtype=np.float64,
        )
        transform = RowL2Normalizer(eps=1e-12)

        actual = transform.transform(matrix, task)
        expected = apply_l2_normalization(matrix, l2_norm=True, eps=1e-12)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(
            transform.config(),
            {"name": "row_l2_normalization", "eps": 1e-12},
        )

    def test_row_l2_has_expected_nonzero_row_norms(self):
        task = make_task(scale=2)
        matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        actual = RowL2Normalizer().transform(matrix, task)

        self.assertAlmostEqual(float(np.linalg.norm(actual[0])), 1.0, places=6)
        self.assertEqual(float(np.linalg.norm(actual[1])), 0.0)

    def test_transforms_reject_invalid_input_shape(self):
        for transform in (IdentityTransform(), RowL2Normalizer()):
            with self.subTest(transform=transform.name):
                with self.assertRaisesRegex(ValueError, "must be 2D"):
                    transform.transform(np.ones(2), make_task())
                with self.assertRaisesRegex(ValueError, "does not match task scale"):
                    transform.transform(np.ones((3, 3)), make_task(scale=2))

    def test_row_l2_rejects_invalid_epsilon(self):
        for eps in (-1.0, float("nan"), float("inf"), "invalid"):
            with self.subTest(eps=eps):
                with self.assertRaises((TypeError, ValueError)):
                    RowL2Normalizer(eps=eps)


class FakePartitionMethod:
    name = "sign_partition"

    def fit_partition(self, matrix, task):
        labels = (matrix[:, 0] >= 0).astype(np.int64)
        return PartitionResult(task=task, method=self.name, labels=labels)

    def config(self):
        return {"name": self.name, "column": 0}


class PartitionResultTests(unittest.TestCase):
    def test_summarizes_clusters_noise_and_tokens(self):
        result = PartitionResult(
            task=make_task(),
            method="fake",
            labels=np.array([-1, 0, 0, 2, -1], dtype=np.int32),
            probabilities=np.array([0.1, 0.8, 0.9, 0.7, 0.2], dtype=np.float64),
        )

        self.assertEqual(result.n_tokens, 5)
        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(result.n_noise, 2)
        self.assertAlmostEqual(result.noise_fraction, 0.4)

    def test_to_frame_matches_existing_token_table_contract(self):
        result = PartitionResult(
            task=make_task(),
            method="fake",
            labels=np.array([-1, 3], dtype=np.int64),
            probabilities=np.array([0.25, 0.75], dtype=np.float32),
        )
        frame = result.to_frame()

        self.assertEqual(frame.columns.tolist(), ["token_id", "cluster_id", "probability"])
        np.testing.assert_array_equal(frame["token_id"].to_numpy(), [0, 1])
        np.testing.assert_array_equal(frame["cluster_id"].to_numpy(), [-1, 3])
        np.testing.assert_array_equal(frame["probability"].to_numpy(), [0.25, 0.75])

    def test_to_frame_uses_nan_when_probabilities_are_unavailable(self):
        result = PartitionResult(
            task=make_task(),
            method="fake",
            labels=np.array([0, 0], dtype=np.int64),
        )
        self.assertTrue(result.to_frame()["probability"].isna().all())

    def test_empty_partition_has_zero_noise_fraction(self):
        result = PartitionResult(
            task=make_task(),
            method="fake",
            labels=np.array([], dtype=np.int64),
        )
        self.assertEqual(result.noise_fraction, 0.0)

    def test_rejects_invalid_labels(self):
        with self.assertRaisesRegex(TypeError, "NumPy array"):
            PartitionResult(task=make_task(), method="fake", labels=[0, 1])
        with self.assertRaisesRegex(ValueError, "labels must be 1D"):
            PartitionResult(
                task=make_task(),
                method="fake",
                labels=np.ones((2, 1), dtype=np.int64),
            )
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            PartitionResult(
                task=make_task(),
                method="fake",
                labels=np.array([0.0, 1.0]),
            )

    def test_rejects_invalid_probabilities(self):
        invalid_values = (
            np.array([0.5]),
            np.array([0.5, np.nan]),
            np.array([0.5, 1.1]),
        )
        expected_messages = ("equal lengths", "finite", "within")
        for probabilities, message in zip(invalid_values, expected_messages):
            with self.subTest(probabilities=probabilities):
                with self.assertRaisesRegex(ValueError, message):
                    PartitionResult(
                        task=make_task(),
                        method="fake",
                        labels=np.array([0, 1], dtype=np.int64),
                        probabilities=probabilities,
                    )


class PartitionMethodProtocolTests(unittest.TestCase):
    def test_fake_method_satisfies_protocol_without_pipeline_changes(self):
        method = FakePartitionMethod()
        task = make_task()
        matrix = np.array([[-1.0, 0.0], [1.0, 0.0]])

        self.assertIsInstance(method, PartitionMethod)
        np.testing.assert_array_equal(
            method.fit_partition(matrix, task).labels,
            [0, 1],
        )
        self.assertEqual(method.config()["name"], "sign_partition")


class HDBSCANAdapterTests(unittest.TestCase):
    def test_passes_exact_parameters_and_standardizes_old_table(self):
        from crossscale.pca_hdbscan import build_token_cluster_df

        task = make_task(scale=2)
        matrix = np.arange(8, dtype=np.float32).reshape(4, 2)
        labels = np.array([-1, 0, 0, 1], dtype=np.int64)
        probabilities = np.array([0.0, 0.8, 0.9, 0.7], dtype=np.float32)
        calls = []

        def fake_runner(received_matrix, **parameters):
            calls.append((received_matrix, parameters))
            return labels, probabilities

        adapter = HDBSCANAdapter(
            min_cluster_size=7,
            min_samples=3,
            cluster_selection_method="leaf",
            metric="euclidean",
            cluster_selection_epsilon=0.25,
            runner=fake_runner,
        )
        result = adapter.fit_partition(matrix, task)

        self.assertIsInstance(adapter, PartitionMethod)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], matrix)
        self.assertEqual(
            calls[0][1],
            {
                "min_cluster_size": 7,
                "min_samples": 3,
                "cluster_selection_method": "leaf",
                "metric": "euclidean",
                "cluster_selection_epsilon": 0.25,
            },
        )
        self.assertTrue(
            result.to_frame().equals(build_token_cluster_df(labels, probabilities))
        )
        self.assertEqual(result.metadata["parameters"], adapter.config())

    def test_injected_table_builder_is_called(self):
        import pandas as pd

        task = make_task()
        labels = np.array([0, 1], dtype=np.int64)
        probabilities = np.array([0.5, 0.6], dtype=np.float32)
        builder_calls = []

        def fake_runner(matrix, **parameters):
            return labels, probabilities

        def fake_builder(received_labels, received_probabilities):
            builder_calls.append((received_labels, received_probabilities))
            return pd.DataFrame(
                {
                    "token_id": np.arange(2),
                    "cluster_id": received_labels,
                    "probability": received_probabilities,
                }
            )

        adapter = HDBSCANAdapter(runner=fake_runner, token_table_builder=fake_builder)
        result = adapter.fit_partition(np.ones((2, 2)), task)
        self.assertEqual(len(builder_calls), 1)
        np.testing.assert_array_equal(result.labels, labels)

    def test_rejects_malformed_token_table(self):
        task = make_task()

        def fake_runner(matrix, **parameters):
            return np.array([0, 1]), np.array([0.5, 0.6])

        def missing_column_builder(labels, probabilities):
            import pandas as pd

            return pd.DataFrame({"token_id": [0, 1], "cluster_id": labels})

        adapter = HDBSCANAdapter(
            runner=fake_runner,
            token_table_builder=missing_column_builder,
        )
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            adapter.fit_partition(np.ones((2, 2)), task)

    def test_rejects_noncontiguous_token_ids(self):
        import pandas as pd

        task = make_task()

        def fake_runner(matrix, **parameters):
            return np.array([0, 1]), np.array([0.5, 0.6])

        def bad_ids_builder(labels, probabilities):
            return pd.DataFrame(
                {
                    "token_id": [1, 2],
                    "cluster_id": labels,
                    "probability": probabilities,
                }
            )

        adapter = HDBSCANAdapter(runner=fake_runner, token_table_builder=bad_ids_builder)
        with self.assertRaisesRegex(ValueError, "contiguous and zero-based"):
            adapter.fit_partition(np.ones((2, 2)), task)

    def test_validates_parameters(self):
        for field_name in ("min_cluster_size", "min_samples"):
            with self.subTest(field_name=field_name):
                with self.assertRaises((TypeError, ValueError)):
                    HDBSCANAdapter(**{field_name: 0})
        for field_name in ("cluster_selection_method", "metric"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    HDBSCANAdapter(**{field_name: ""})
        with self.assertRaisesRegex(ValueError, "cluster_selection_epsilon"):
            HDBSCANAdapter(cluster_selection_epsilon=-0.1)


def make_manifest_row(task=None, **overrides):
    values = {
        "task": task or make_task(),
        "method": "hdbscan",
        "partition_csv_path": "partitions/model/scale_2/partition.csv",
        "n_tokens": 5,
        "n_clusters": 2,
        "n_noise": 1,
        "noise_fraction": 0.2,
        "config_hash": "b" * 64,
        "reused": False,
    }
    values.update(overrides)
    return PartitionManifestRow(**values)


class PartitionManifestRowTests(unittest.TestCase):
    def test_flattens_task_and_partition_fields(self):
        row = make_manifest_row()
        serialized = row.to_dict()

        self.assertEqual(tuple(serialized), PARTITION_MANIFEST_COLUMNS)
        self.assertEqual(serialized["analysis_id"], row.task.analysis_id)
        self.assertEqual(serialized["method"], "hdbscan")

    def test_round_trip_with_optional_seed(self):
        row = make_manifest_row(
            task=make_task(seed_type="projection", seed=42),
            reused=True,
        )
        restored = PartitionManifestRow.from_dict(row.to_dict())
        self.assertEqual(restored, row)

    def test_validates_counts_and_noise_fraction(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            make_manifest_row(n_tokens=2, n_noise=3, noise_fraction=1.5)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            make_manifest_row(noise_fraction=0.3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_manifest_row(n_clusters=-1)


class PartitionManifestTests(unittest.TestCase):
    def test_dataframe_has_canonical_columns_and_nullable_seed(self):
        manifest = PartitionManifest(
            rows=(
                make_manifest_row(task=make_task(scale=2)),
                make_manifest_row(
                    task=make_task(
                        scale=3,
                        seed_type="projection",
                        seed=42,
                    ),
                    partition_csv_path="partitions/model/scale_3/partition.csv",
                ),
            )
        )
        frame = manifest.to_frame()

        self.assertEqual(tuple(frame.columns), PARTITION_MANIFEST_COLUMNS)
        self.assertEqual(str(frame["seed"].dtype), "Int64")
        self.assertTrue(frame.loc[0, "seed"] is pd.NA)
        self.assertEqual(frame.loc[1, "seed"], 42)

    def test_csv_round_trip_preserves_all_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = PartitionManifest(
                rows=(
                    make_manifest_row(task=make_task(scale=2)),
                    make_manifest_row(
                        task=make_task(
                            scale=4,
                            seed_type="pca",
                            seed=1000,
                        ),
                        partition_csv_path="partitions/model/scale_4/partition.csv",
                        reused=True,
                    ),
                )
            )
            path = Path(temp_dir) / "nested" / "partition_manifest.csv"

            manifest.write_csv(path)
            restored = PartitionManifest.read_csv(path)

            self.assertEqual(restored, manifest)

    def test_write_refuses_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "partition_manifest.csv"
            manifest = PartitionManifest(rows=(make_manifest_row(),))
            manifest.write_csv(path)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                manifest.write_csv(path)
            manifest.write_csv(path, overwrite=True)

    def test_rejects_duplicate_task_method_identity(self):
        with self.assertRaisesRegex(ValueError, "duplicate task-method"):
            PartitionManifest(rows=(make_manifest_row(), make_manifest_row()))

    def test_rejects_noncanonical_frame_columns(self):
        frame = PartitionManifest(rows=(make_manifest_row(),)).to_frame()
        with self.assertRaisesRegex(ValueError, "canonical order"):
            PartitionManifest.from_frame(frame[list(reversed(frame.columns))])


class ClusteringOutputTests(unittest.TestCase):
    def make_result(self, task=None):
        return PartitionResult(
            task=task or make_task(),
            method="fake",
            labels=np.array([-1, 0, 0, 1], dtype=np.int64),
            probabilities=np.array([0.1, 0.8, 0.9, 0.7], dtype=np.float32),
            metadata={"parameters": {"threshold": 0.0}},
        )

    def test_layout_keeps_output_under_explicit_root(self):
        task = make_task(seed_type="projection", seed=42)
        paths = ClusteringOutputLayout("isolated_output").paths_for(task, "fake")

        self.assertEqual(
            paths.partition_csv,
            Path(
                "isolated_output/validation/partitions/organization/model/"
                "output_proj/full_pca_prefix/projection/seed_00042/fake/"
                "scale_2/partition.csv"
            ),
        )
        self.assertEqual(paths.metadata_json.parent, paths.partition_csv.parent)

    def test_write_load_and_validation_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.make_result()
            paths = ClusteringOutputLayout(temp_dir).paths_for(result.task, result.method)
            config_hash = "c" * 64
            row = write_partition_output(
                result,
                paths,
                config_hash=config_hash,
                config={"method": {"name": "fake"}},
            )
            loaded = load_partition_result(
                paths.partition_csv,
                task=result.task,
                method=result.method,
            )
            reused_row = validate_partition_output(
                paths,
                expected_task=result.task,
                expected_method=result.method,
                expected_config_hash=config_hash,
            )

            np.testing.assert_array_equal(loaded.labels, result.labels)
            np.testing.assert_array_equal(loaded.probabilities, result.probabilities)
            self.assertFalse(row.reused)
            self.assertTrue(reused_row.reused)
            self.assertEqual(row.n_clusters, reused_row.n_clusters)
            self.assertEqual(list(paths.task_dir.glob("*.tmp")), [])

    def test_metadata_contains_config_and_method_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.make_result()
            paths = ClusteringOutputLayout(temp_dir).paths_for(result.task, result.method)
            write_partition_output(
                result,
                paths,
                config_hash="d" * 64,
                config={"transforms": [{"name": "identity"}]},
            )
            with paths.metadata_json.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            self.assertEqual(metadata["task"], result.task.to_dict())
            self.assertEqual(metadata["method_metadata"], dict(result.metadata))
            self.assertEqual(metadata["n_noise"], result.n_noise)

    def test_existing_or_incomplete_output_requires_explicit_action(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.make_result()
            paths = ClusteringOutputLayout(temp_dir).paths_for(result.task, result.method)
            paths.task_dir.mkdir(parents=True)
            paths.partition_csv.write_text("incomplete", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_partition_output(
                    result,
                    paths,
                    config_hash="e" * 64,
                    config={},
                )
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                validate_partition_output(
                    paths,
                    expected_task=result.task,
                    expected_method=result.method,
                    expected_config_hash="e" * 64,
                )

    def test_replace_is_explicit_and_produces_valid_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.make_result()
            paths = ClusteringOutputLayout(temp_dir).paths_for(result.task, result.method)
            write_partition_output(
                result,
                paths,
                config_hash="f" * 64,
                config={},
            )
            replacement = PartitionResult(
                task=result.task,
                method=result.method,
                labels=np.array([0, 0], dtype=np.int64),
                probabilities=np.array([1.0, 1.0], dtype=np.float32),
            )
            row = write_partition_output(
                replacement,
                paths,
                config_hash="1" * 64,
                config={},
                replace=True,
            )

            self.assertEqual(row.n_tokens, 2)
            validate_partition_output(
                paths,
                expected_task=result.task,
                expected_method=result.method,
                expected_config_hash="1" * 64,
            )

    def test_validation_rejects_config_or_summary_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.make_result()
            paths = ClusteringOutputLayout(temp_dir).paths_for(result.task, result.method)
            write_partition_output(
                result,
                paths,
                config_hash="2" * 64,
                config={},
            )
            with self.assertRaisesRegex(ValueError, "config hash"):
                validate_partition_output(
                    paths,
                    expected_task=result.task,
                    expected_method=result.method,
                    expected_config_hash="3" * 64,
                )

            with paths.metadata_json.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata["n_tokens"] = 999
            with paths.metadata_json.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle)
            with self.assertRaisesRegex(ValueError, "n_tokens"):
                validate_partition_output(
                    paths,
                    expected_task=result.task,
                    expected_method=result.method,
                    expected_config_hash="2" * 64,
                )

    def test_load_rejects_partial_probabilities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "partition.csv"
            pd.DataFrame(
                {
                    "token_id": [0, 1],
                    "cluster_id": [0, 0],
                    "probability": [0.5, np.nan],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "all missing or all present"):
                load_partition_result(path, task=make_task(), method="fake")


class CountingProvider:
    def __init__(self, matrices):
        self.matrices = matrices
        self.calls = []

    def load(self, task):
        self.calls.append(task.task_key)
        return RepresentationResult(task=task, matrix=self.matrices[task.task_key])


class CountingTransform:
    name = "add_one"

    def __init__(self):
        self.calls = []

    def transform(self, matrix, task):
        self.calls.append(task.task_key)
        return matrix + 1.0

    def config(self):
        return {"name": self.name}


class CountingPartitionMethod:
    name = "fake"

    def __init__(self, fail_scale=None):
        self.calls = []
        self.fail_scale = fail_scale

    def fit_partition(self, matrix, task):
        self.calls.append((task.task_key, matrix.copy()))
        if task.scale == self.fail_scale:
            raise RuntimeError("intentional failure")
        labels = (matrix[:, 0] >= 1.0).astype(np.int64)
        return PartitionResult(
            task=task,
            method=self.name,
            labels=labels,
            probabilities=np.ones(matrix.shape[0], dtype=np.float32),
            metadata={"observed_shape": list(matrix.shape)},
        )

    def config(self):
        return {"name": self.name, "threshold": 1.0}


class ClusteringPipelineTests(unittest.TestCase):
    def make_components(self, output_root, tasks, *, policy=None, fail_scale=None):
        matrices = {
            task.task_key: np.arange(4 * task.scale, dtype=np.float32).reshape(
                4,
                task.scale,
            )
            for task in tasks
        }
        provider = CountingProvider(matrices)
        transform = CountingTransform()
        method = CountingPartitionMethod(fail_scale=fail_scale)
        pipeline = ClusteringPipeline(
            provider=provider,
            transforms=(transform,),
            method=method,
            output_layout=ClusteringOutputLayout(output_root),
            run_policy=policy,
            clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
        )
        return pipeline, provider, transform, method

    def test_runs_each_task_once_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = (make_task(scale=2), make_task(scale=3))
            pipeline, provider, transform, method = self.make_components(temp_dir, tasks)

            run = pipeline.run(tasks)

            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(len(transform.calls), 2)
            self.assertEqual(len(method.calls), 2)
            self.assertEqual(len(run.manifest.rows), 2)
            self.assertEqual(run.report.computed_count, 2)
            self.assertEqual(run.report.reused_count, 0)
            self.assertTrue(run.report.succeeded)
            self.assertTrue(run.manifest_path.is_file())
            self.assertEqual(PartitionManifest.read_csv(run.manifest_path), run.manifest)
            self.assertEqual(method.calls[0][1][0, 0], 1.0)

    def test_resume_validates_and_skips_all_adapters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = (make_task(scale=2), make_task(scale=3))
            first, _, _, _ = self.make_components(temp_dir, tasks)
            first.run(tasks)
            resumed, provider, transform, method = self.make_components(
                temp_dir,
                tasks,
                policy=RunPolicy(existing_output="resume"),
            )

            run = resumed.run(tasks)

            self.assertEqual(provider.calls, [])
            self.assertEqual(transform.calls, [])
            self.assertEqual(method.calls, [])
            self.assertEqual(run.report.reused_count, 2)
            self.assertTrue(all(row.reused for row in run.manifest.rows))

    def test_error_mode_preflights_before_any_adapter_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = (make_task(scale=2),)
            first, _, _, _ = self.make_components(temp_dir, tasks)
            first.run(tasks)
            second, provider, transform, method = self.make_components(temp_dir, tasks)

            with self.assertRaisesRegex(FileExistsError, "already exist"):
                second.run(tasks)
            self.assertEqual(provider.calls, [])
            self.assertEqual(transform.calls, [])
            self.assertEqual(method.calls, [])

    def test_replace_recomputes_existing_tasks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = (make_task(scale=2),)
            first, _, _, _ = self.make_components(temp_dir, tasks)
            first.run(tasks)
            replacement, provider, _, method = self.make_components(
                temp_dir,
                tasks,
                policy=RunPolicy(existing_output="replace"),
            )

            run = replacement.run(tasks)

            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(len(method.calls), 1)
            self.assertEqual(run.report.computed_count, 1)

    def test_resume_rejects_incomplete_output_without_computing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = (make_task(scale=2),)
            pipeline, provider, _, method = self.make_components(
                temp_dir,
                tasks,
                policy=RunPolicy(existing_output="resume"),
            )
            paths = pipeline.output_layout.paths_for(tasks[0], method.name)
            paths.task_dir.mkdir(parents=True)
            paths.partition_csv.write_text("incomplete", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                pipeline.run(tasks)
            self.assertEqual(provider.calls, [])
            self.assertEqual(method.calls, [])

    def test_continue_on_error_returns_failed_record_and_successes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks = (make_task(scale=2), make_task(scale=3))
            pipeline, _, _, _ = self.make_components(
                temp_dir,
                tasks,
                policy=RunPolicy(stop_on_error=False),
                fail_scale=2,
            )

            run = pipeline.run(tasks)

            self.assertEqual(run.report.failed_count, 1)
            self.assertEqual(run.report.computed_count, 1)
            self.assertEqual(len(run.manifest.rows), 1)
            self.assertFalse(run.report.succeeded)

    def test_full_pca_source_and_output_roots_remain_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "source"
            output_root = Path(temp_dir) / "output"
            task = make_task(scale=2)
            pca_path = source_root / task.model_name / "full_pca" / "X_pca.npy"
            pca_path.parent.mkdir(parents=True)
            np.save(pca_path, np.ones((4, 3), dtype=np.float32))
            method = CountingPartitionMethod()
            pipeline = ClusteringPipeline(
                provider=FullPCAPrefixProvider(source_root, mmap_mode=None),
                transforms=(IdentityTransform(),),
                method=method,
                output_layout=ClusteringOutputLayout(output_root),
            )

            run = pipeline.run((task,))

            self.assertTrue(str(run.manifest_path).startswith(str(output_root)))
            self.assertFalse(any(source_root.rglob("partition.csv")))
            self.assertTrue(any(output_root.rglob("partition.csv")))


class LegacySavedPCARootTests(unittest.TestCase):
    def setUp(self):
        from crossscale import pca_hdbscan

        self.module = pca_hdbscan
        self.parameters = [
            {
                "min_cluster_size": 5,
                "min_samples": 5,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "cluster_selection_epsilon": 0.0,
            }
        ]

    def fake_summary(self, **kwargs):
        return {
            "pca_dim": kwargs["pca_dim"],
            "cluster_csv_path": str(Path(kwargs["out_root"]) / "partition.csv"),
        }

    def test_default_keeps_backward_compatible_shared_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                self.module,
                "load_saved_pca_result",
                return_value={
                    "X_pca": np.ones((4, 2), dtype=np.float32),
                    "save_dir": "mock",
                },
            ) as loader, patch.object(
                self.module,
                "run_gpu_hdbscan",
                return_value=(np.array([0, 0, 1, -1]), np.ones(4, dtype=np.float32)),
            ), patch.object(
                self.module,
                "save_run_outputs",
                side_effect=self.fake_summary,
            ) as saver:
                self.module.run_saved_pca_hdbscan_grid(
                    candidate_dims=[2],
                    hdbscan_param_grid=self.parameters,
                    model_name="model",
                    space_name="space",
                    pca_mode="full",
                    out_root=temp_dir,
                )

            self.assertEqual(loader.call_args.kwargs["out_root"], temp_dir)
            self.assertEqual(saver.call_args.kwargs["out_root"], temp_dir)

    def test_explicit_source_root_does_not_change_output_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = str(Path(temp_dir) / "source")
            output_root = str(Path(temp_dir) / "output")
            with patch.object(
                self.module,
                "load_saved_pca_result",
                return_value={
                    "X_pca": np.ones((4, 2), dtype=np.float32),
                    "save_dir": "mock",
                },
            ) as loader, patch.object(
                self.module,
                "run_gpu_hdbscan",
                return_value=(np.array([0, 0, 1, -1]), np.ones(4, dtype=np.float32)),
            ), patch.object(
                self.module,
                "save_run_outputs",
                side_effect=self.fake_summary,
            ) as saver:
                self.module.run_saved_pca_hdbscan_grid(
                    candidate_dims=[2],
                    hdbscan_param_grid=self.parameters,
                    model_name="model",
                    space_name="space",
                    pca_mode="full",
                    out_root=output_root,
                    pca_source_root=source_root,
                )

            self.assertEqual(loader.call_args.kwargs["out_root"], source_root)
            self.assertEqual(saver.call_args.kwargs["out_root"], output_root)
            self.assertTrue(
                (Path(output_root) / "model" / "space" / "summary_full_pca.csv").is_file()
            )

    def test_explicit_source_root_applies_to_randomized_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = str(Path(temp_dir) / "source")
            output_root = str(Path(temp_dir) / "output")
            with patch.object(
                self.module,
                "load_saved_pca_result",
                return_value={
                    "X_pca": np.ones((4, 2), dtype=np.float32),
                    "save_dir": "mock",
                },
            ) as loader, patch.object(
                self.module,
                "run_gpu_hdbscan",
                return_value=(np.array([0, 0, 1, -1]), np.ones(4, dtype=np.float32)),
            ), patch.object(
                self.module,
                "save_run_outputs",
                side_effect=self.fake_summary,
            ) as saver:
                self.module.run_saved_pca_hdbscan_grid(
                    candidate_dims=[2],
                    hdbscan_param_grid=self.parameters,
                    model_name="model",
                    space_name="space",
                    pca_mode="randomized",
                    pca_seed=42,
                    out_root=output_root,
                    pca_source_root=source_root,
                )

            self.assertEqual(loader.call_args.kwargs["out_root"], source_root)
            self.assertEqual(saver.call_args.kwargs["out_root"], output_root)


if __name__ == "__main__":
    unittest.main()
