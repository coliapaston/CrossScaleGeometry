from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cluster_anlys.clustering_pipeline import (
    ClusteringOutputLayout,
    PartitionManifest,
    PartitionManifestRow,
)
from cluster_anlys.permutation_pipeline import (
    GlobalTokenPermutation,
    LengthBucketTokenPermutation,
    PartitionPermutationPipeline,
    PartitionPermutationStrategy,
)
from cluster_anlys.pipeline_core import RunPolicy, ScaleTask


class FakeTokenizer:
    def __init__(self):
        self.tokens = {
            0: "noise",
            1: "a",
            2: "bb",
            3: "ccc",
            4: "dddd",
            5: "eeeee",
            6: "noise-token",
            7: "verylongtoken",
        }

    def convert_ids_to_tokens(self, token_id):
        return self.tokens[int(token_id)]


def make_baseline_manifest(root, *, scales=(2,)):
    rows = []
    for scale in scales:
        task = ScaleTask(
            analysis_id="low12_baseline",
            model_name="organization/model",
            space_name="output_proj",
            scale=scale,
            source="full_pca_prefix",
        )
        path = Path(root) / f"baseline_scale_{scale}.csv"
        frame = pd.DataFrame(
            {
                "token_id": np.arange(8, dtype=np.int64),
                "cluster_id": [-1, 0, 0, 1, 1, 1, -1, 2],
                "probability": [0.0, 0.8, 0.9, 0.7, 0.6, 0.5, 0.0, 1.0],
            }
        )
        frame.to_csv(path, index=False)
        rows.append(
            PartitionManifestRow(
                task=task,
                method="hdbscan",
                partition_csv_path=str(path),
                n_tokens=8,
                n_clusters=3,
                n_noise=2,
                noise_fraction=0.25,
                config_hash=f"{scale:064d}",
                reused=False,
            )
        )
    return PartitionManifest(rows=tuple(rows))


class PermutationStrategyTests(unittest.TestCase):
    def test_both_strategies_satisfy_the_narrow_protocol(self):
        self.assertIsInstance(GlobalTokenPermutation(), PartitionPermutationStrategy)
        self.assertIsInstance(
            LengthBucketTokenPermutation(tokenizer_loader=lambda path: FakeTokenizer()),
            PartitionPermutationStrategy,
        )
        self.assertNotIn("metric", GlobalTokenPermutation().config())
        self.assertNotIn("hdbscan", LengthBucketTokenPermutation().config())


class PartitionPermutationPipelineTests(unittest.TestCase):
    def make_pipeline(self, output_root, strategy, *, seeds=(0, 42), policy=None):
        return PartitionPermutationPipeline(
            strategy=strategy,
            analysis_id=f"low12_{strategy.name}",
            seeds=seeds,
            output_layout=ClusteringOutputLayout(output_root),
            run_policy=policy,
            clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
        )

    def test_global_permutation_removes_noise_and_preserves_partition_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = make_baseline_manifest(temp_dir)
            pipeline = self.make_pipeline(
                Path(temp_dir) / "outputs",
                GlobalTokenPermutation(),
            )

            run = pipeline.run(baseline)

            self.assertEqual(len(run.manifest.rows), 2)
            self.assertEqual(run.report.computed_count, 2)
            self.assertTrue(run.manifest_path.is_file())
            self.assertEqual(PartitionManifest.read_csv(run.manifest_path), run.manifest)
            baseline_valid = pd.read_csv(
                baseline.rows[0].partition_csv_path
            ).query("cluster_id != -1").reset_index(drop=True)
            for row in run.manifest.rows:
                control = pd.read_csv(row.partition_csv_path)
                self.assertEqual(row.n_tokens, 6)
                self.assertEqual(row.n_clusters, 3)
                self.assertEqual(row.n_noise, 0)
                self.assertEqual(
                    sorted(control["token_id"].tolist()),
                    sorted(baseline_valid["token_id"].tolist()),
                )
                np.testing.assert_array_equal(
                    control["cluster_id"],
                    baseline_valid["cluster_id"],
                )
                np.testing.assert_array_equal(
                    control["probability"],
                    baseline_valid["probability"],
                )

    def test_length_bucket_strategy_never_crosses_original_bucket_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = make_baseline_manifest(temp_dir)
            loader_calls = []
            strategy = LengthBucketTokenPermutation(
                tokenizer_loader=lambda path: loader_calls.append(path) or FakeTokenizer()
            )
            pipeline = self.make_pipeline(
                Path(temp_dir) / "outputs",
                strategy,
                seeds=(0,),
            )

            run = pipeline.run(baseline)

            self.assertEqual(loader_calls, ["organization/model/tokenizer"])
            control = pd.read_csv(run.manifest.rows[0].partition_csv_path)
            baseline_valid = pd.read_csv(
                baseline.rows[0].partition_csv_path
            ).query("cluster_id != -1").reset_index(drop=True)
            tokenizer = FakeTokenizer()

            def bucket(token_id):
                length = len(tokenizer.convert_ids_to_tokens(token_id))
                if length <= 2:
                    return "1-2"
                if length <= 5:
                    return "3-5"
                if length <= 10:
                    return "6-10"
                return "11+"

            self.assertEqual(
                [bucket(value) for value in control["token_id"]],
                [bucket(value) for value in baseline_valid["token_id"]],
            )

    def test_resume_validates_and_reuses_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = make_baseline_manifest(temp_dir)
            output_root = Path(temp_dir) / "outputs"
            first = self.make_pipeline(output_root, GlobalTokenPermutation())
            first_run = first.run(baseline)
            modification_times = {
                row.partition_csv_path: Path(row.partition_csv_path).stat().st_mtime_ns
                for row in first_run.manifest.rows
            }
            resumed = self.make_pipeline(
                output_root,
                GlobalTokenPermutation(),
                policy=RunPolicy(existing_output="resume"),
            )

            resumed_run = resumed.run(baseline)

            self.assertEqual(resumed_run.report.reused_count, 2)
            self.assertTrue(all(row.reused for row in resumed_run.manifest.rows))
            self.assertEqual(
                modification_times,
                {
                    row.partition_csv_path: Path(row.partition_csv_path).stat().st_mtime_ns
                    for row in resumed_run.manifest.rows
                },
            )

    def test_resume_rejects_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = make_baseline_manifest(temp_dir)
            output_root = Path(temp_dir) / "outputs"
            pipeline = self.make_pipeline(
                output_root,
                GlobalTokenPermutation(),
                seeds=(0,),
                policy=RunPolicy(existing_output="resume"),
            )
            task = ScaleTask(
                analysis_id="low12_token_permutation",
                model_name="organization/model",
                space_name="output_proj",
                scale=2,
                source="token_permutation",
                seed_type="permutation_seed",
                seed=0,
            )
            paths = ClusteringOutputLayout(output_root).paths_for(
                task,
                "token_permutation",
            )
            paths.task_dir.mkdir(parents=True)
            paths.partition_csv.write_text("partial", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                pipeline.run(baseline)

    def test_error_mode_preflights_before_loading_baseline_partitions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = make_baseline_manifest(temp_dir)
            output_root = Path(temp_dir) / "outputs"
            first = self.make_pipeline(
                output_root,
                GlobalTokenPermutation(),
                seeds=(0,),
            )
            first.run(baseline)
            load_calls = []
            second = PartitionPermutationPipeline(
                strategy=GlobalTokenPermutation(),
                analysis_id="low12_token_permutation",
                seeds=(0,),
                output_layout=ClusteringOutputLayout(output_root),
                partition_loader=lambda path: load_calls.append(path),
            )

            with self.assertRaisesRegex(FileExistsError, "already exist"):
                second.run(baseline)

            self.assertEqual(load_calls, [])

    def test_contract_scales_to_thirty_six_partitions_and_ten_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = make_baseline_manifest(temp_dir, scales=tuple(range(1, 13)))
            replicated_rows = []
            for model_index in range(3):
                for row in baseline.rows:
                    task = ScaleTask(
                        analysis_id=row.task.analysis_id,
                        model_name=f"model-{model_index}",
                        space_name=row.task.space_name,
                        scale=row.task.scale,
                        source=row.task.source,
                    )
                    replicated_rows.append(
                        PartitionManifestRow(
                            task=task,
                            method=row.method,
                            partition_csv_path=row.partition_csv_path,
                            n_tokens=row.n_tokens,
                            n_clusters=row.n_clusters,
                            n_noise=row.n_noise,
                            noise_fraction=row.noise_fraction,
                            config_hash=row.config_hash,
                            reused=False,
                        )
                    )
            seeds = tuple(range(10))
            tasks = tuple(
                ScaleTask(
                    analysis_id="control",
                    model_name=row.task.model_name,
                    space_name=row.task.space_name,
                    scale=row.task.scale,
                    source="token_permutation",
                    seed_type="permutation_seed",
                    seed=seed,
                )
                for row in replicated_rows
                for seed in seeds
            )

            self.assertEqual(len(replicated_rows), 36)
            self.assertEqual(len(tasks), 360)
            self.assertEqual(len({task.task_key for task in tasks}), 360)


if __name__ == "__main__":
    unittest.main()
