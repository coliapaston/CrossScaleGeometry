import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cluster_anlys.pipeline_core import RunPolicy, ScaleTask
from cluster_anlys.representation_pipeline import (
    REPRESENTATION_MANIFEST_COLUMNS,
    RandomizedPCABuilder,
    RandomizedPCAOutputLayout,
    RandomizedPCARepresentationPipeline,
    RepresentationBuilder,
    RepresentationManifest,
    randomized_pca_matrix_path,
)


def make_task(**overrides):
    values = {
        "analysis_id": "low12_cross_seed",
        "model_name": "organization/model",
        "space_name": "output_proj",
        "scale": 2,
        "source": "randomized_pca",
        "seed_type": "pca_seed",
        "seed": 42,
    }
    values.update(overrides)
    return ScaleTask(**values)


class FakeRandomizedPCARunner:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        matrix,
        *,
        pca_dim,
        pca_seed,
        model_name,
        out_root,
        n_oversamples,
        power_iteration_normalizer,
        iterated_power,
    ):
        self.calls.append(
            {
                "model_name": model_name,
                "pca_dim": pca_dim,
                "pca_seed": pca_seed,
                "out_root": out_root,
                "n_oversamples": n_oversamples,
                "power_iteration_normalizer": power_iteration_normalizer,
                "iterated_power": iterated_power,
            }
        )
        task = make_task(
            model_name=model_name,
            scale=pca_dim,
            seed=pca_seed,
        )
        paths = RandomizedPCAOutputLayout(out_root).paths_for(task)
        paths.task_dir.mkdir(parents=True, exist_ok=True)
        n_samples, n_features = matrix.shape
        np.save(paths.matrix, matrix[:, :pca_dim].astype(np.float32))
        np.save(
            paths.components,
            np.eye(pca_dim, n_features, dtype=np.float32),
        )
        np.save(
            paths.explained_ratio,
            np.full(pca_dim, 1.0 / pca_dim, dtype=np.float32),
        )
        np.save(paths.mean, np.mean(matrix, axis=0).astype(np.float32))
        paths.metadata.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "pca_dim": pca_dim,
                    "pca_seed": pca_seed,
                }
            ),
            encoding="utf-8",
        )


class RandomizedPCAOutputLayoutTests(unittest.TestCase):
    def test_matches_the_existing_unpadded_repository_layout(self):
        task = make_task(scale=16, seed=42)
        layout = RandomizedPCAOutputLayout("comp")

        paths = layout.paths_for(task)

        expected = Path(
            "comp/organization/model/random_pca/seed_42/pca_dim_16/X_pca.npy"
        )
        self.assertEqual(paths.matrix, expected)
        self.assertEqual(randomized_pca_matrix_path(Path("comp"), task), expected)

    def test_requires_a_seed(self):
        task = ScaleTask(
            analysis_id="validation",
            model_name="organization/model",
            space_name="output_proj",
            scale=2,
            source="full_pca",
        )
        with self.assertRaisesRegex(ValueError, "require a seed"):
            RandomizedPCAOutputLayout("comp").paths_for(task)


class RandomizedPCABuilderTests(unittest.TestCase):
    def test_is_a_narrow_representation_builder(self):
        runner = FakeRandomizedPCARunner()
        builder = RandomizedPCABuilder(runner=runner)

        self.assertIsInstance(builder, RepresentationBuilder)
        self.assertEqual(builder.config()["fit_strategy"], "independent_per_scale")
        self.assertNotIn("hdbscan", builder.config())

    def test_rejects_invalid_solver_configuration(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            RandomizedPCABuilder(n_oversamples=0)
        with self.assertRaisesRegex(ValueError, "iterated_power"):
            RandomizedPCABuilder(iterated_power=-1)


class RandomizedPCARepresentationPipelineTests(unittest.TestCase):
    def make_pipeline(
        self,
        output_root,
        runner,
        loader,
        *,
        policy=None,
        n_oversamples=10,
    ):
        return RandomizedPCARepresentationPipeline(
            builder=RandomizedPCABuilder(
                runner=runner,
                n_oversamples=n_oversamples,
            ),
            output_layout=RandomizedPCAOutputLayout(output_root),
            matrix_loader=loader,
            run_policy=policy,
            clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
        )

    def test_fits_each_scale_independently_and_loads_each_model_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRandomizedPCARunner()
            loader_calls = []

            def loader(model_name):
                loader_calls.append(model_name)
                return np.arange(24, dtype=np.float32).reshape(6, 4)

            tasks = (
                make_task(scale=2, seed=0),
                make_task(scale=3, seed=0),
                make_task(scale=2, seed=42),
            )
            pipeline = self.make_pipeline(temp_dir, runner, loader)

            run = pipeline.run(tasks)

            self.assertEqual(loader_calls, ["organization/model"])
            self.assertEqual(
                [(call["pca_dim"], call["pca_seed"]) for call in runner.calls],
                [(2, 0), (3, 0), (2, 42)],
            )
            self.assertEqual(run.report.computed_count, 3)
            self.assertEqual(len(run.manifest.rows), 3)
            self.assertTrue(run.manifest_path.is_file())
            self.assertEqual(
                RepresentationManifest.read_csv(run.manifest_path),
                run.manifest,
            )
            self.assertEqual(
                tuple(run.manifest.to_frame().columns),
                REPRESENTATION_MANIFEST_COLUMNS,
            )

    def test_resume_validates_complete_outputs_without_loading_the_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task()
            first_runner = FakeRandomizedPCARunner()
            first = self.make_pipeline(
                temp_dir,
                first_runner,
                lambda model_name: np.ones((5, 4), dtype=np.float32),
            )
            first.run((task,))

            resumed_runner = FakeRandomizedPCARunner()
            loader_calls = []
            resumed = self.make_pipeline(
                temp_dir,
                resumed_runner,
                lambda model_name: loader_calls.append(model_name),
                policy=RunPolicy(existing_output="resume"),
            )

            run = resumed.run((task,))

            self.assertEqual(loader_calls, [])
            self.assertEqual(resumed_runner.calls, [])
            self.assertEqual(run.report.reused_count, 1)
            self.assertTrue(run.manifest.rows[0].reused)

    def test_resume_rejects_partial_output_before_loading_the_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task()
            layout = RandomizedPCAOutputLayout(temp_dir)
            paths = layout.paths_for(task)
            paths.task_dir.mkdir(parents=True)
            np.save(paths.matrix, np.ones((5, 2), dtype=np.float32))
            loader_calls = []
            pipeline = self.make_pipeline(
                temp_dir,
                FakeRandomizedPCARunner(),
                lambda model_name: loader_calls.append(model_name),
                policy=RunPolicy(existing_output="resume"),
            )

            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                pipeline.run((task,))

            self.assertEqual(loader_calls, [])

    def test_resume_rejects_a_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task()
            first = self.make_pipeline(
                temp_dir,
                FakeRandomizedPCARunner(),
                lambda model_name: np.ones((5, 4), dtype=np.float32),
                n_oversamples=10,
            )
            first.run((task,))
            loader_calls = []
            resumed = self.make_pipeline(
                temp_dir,
                FakeRandomizedPCARunner(),
                lambda model_name: loader_calls.append(model_name),
                policy=RunPolicy(existing_output="resume"),
                n_oversamples=20,
            )

            with self.assertRaisesRegex(ValueError, "config hash"):
                resumed.run((task,))

            self.assertEqual(loader_calls, [])

    def test_error_mode_preflights_before_loading_the_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task()
            first = self.make_pipeline(
                temp_dir,
                FakeRandomizedPCARunner(),
                lambda model_name: np.ones((5, 4), dtype=np.float32),
            )
            first.run((task,))
            loader_calls = []
            second = self.make_pipeline(
                temp_dir,
                FakeRandomizedPCARunner(),
                lambda model_name: loader_calls.append(model_name),
            )

            with self.assertRaisesRegex(FileExistsError, "already exist"):
                second.run((task,))

            self.assertEqual(loader_calls, [])

    def test_replace_recomputes_and_preserves_a_complete_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            task = make_task()
            matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
            first = self.make_pipeline(
                temp_dir,
                FakeRandomizedPCARunner(),
                lambda model_name: matrix,
            )
            first.run((task,))
            replacement_runner = FakeRandomizedPCARunner()
            replacement = self.make_pipeline(
                temp_dir,
                replacement_runner,
                lambda model_name: matrix + 1,
                policy=RunPolicy(existing_output="replace"),
            )

            run = replacement.run((task,))

            self.assertEqual(len(replacement_runner.calls), 1)
            self.assertEqual(run.report.computed_count, 1)
            paths = RandomizedPCAOutputLayout(temp_dir).paths_for(task)
            self.assertTrue(all(path.is_file() for path in paths.all_files))
            np.testing.assert_array_equal(
                np.load(paths.matrix),
                (matrix + 1)[:, : task.scale],
            )

    def test_task_contract_supports_three_models_twelve_scales_and_eight_seeds(self):
        models = ("model-a", "model-b", "model-c")
        scales = tuple(range(2, 14))
        seeds = (0, 42, 1000, 9999, 1813382118, 827307999, 1627694678, 1911784257)
        tasks = tuple(
            make_task(model_name=model, scale=scale, seed=seed)
            for model in models
            for seed in seeds
            for scale in scales
        )

        self.assertEqual(len(tasks), 288)
        self.assertEqual(len({task.task_key for task in tasks}), 288)


if __name__ == "__main__":
    unittest.main()
