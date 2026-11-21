import unittest

from crossscale.pipeline_core import (
    RunPolicy,
    ScaleTask,
    StageRunRecord,
    StageRunReport,
    stable_config_hash,
)


def make_task(**overrides):
    values = {
        "analysis_id": "low_energy_lt050_v1",
        "model_name": "mistral",
        "space_name": "pca",
        "scale": 64,
        "source": "full_pca_prefix",
        "seed_type": None,
        "seed": None,
    }
    values.update(overrides)
    return ScaleTask(**values)


class ScaleTaskTests(unittest.TestCase):
    def test_deterministic_key(self):
        self.assertEqual(make_task().task_key, make_task().task_key)

    def test_key_changes_with_identity_field(self):
        variants = {
            "analysis_id": "another_analysis",
            "model_name": "gemma",
            "space_name": "random_projection",
            "scale": 65,
            "source": "saved_per_scale",
        }
        for field_name, value in variants.items():
            with self.subTest(field_name=field_name):
                self.assertNotEqual(
                    make_task().task_key,
                    make_task(**{field_name: value}).task_key,
                )

    def test_key_changes_with_seed_fields(self):
        self.assertNotEqual(
            make_task().task_key,
            make_task(seed_type="projection", seed=0).task_key,
        )

    def test_round_trip_preserves_identity(self):
        task = make_task(seed_type="projection", seed=42)
        restored = ScaleTask.from_dict(task.to_dict())

        self.assertEqual(restored, task)
        self.assertEqual(restored.task_key, task.task_key)
        self.assertEqual(restored.canonical_json(), task.canonical_json())

    def test_rejects_non_positive_scale(self):
        for scale in (0, -1):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "greater than zero"):
                    make_task(scale=scale)

    def test_rejects_non_integer_scale(self):
        for scale in (1.5, "64", True):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(TypeError, "scale must be an integer"):
                    make_task(scale=scale)

    def test_rejects_empty_required_string(self):
        fields = ("analysis_id", "model_name", "space_name", "source")
        for field_name in fields:
            for value in ("", "   ", None):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, field_name):
                        make_task(**{field_name: value})

    def test_rejects_incomplete_seed_identity(self):
        for seed_type, seed in (("projection", None), (None, 42)):
            with self.subTest(seed_type=seed_type, seed=seed):
                with self.assertRaisesRegex(ValueError, "must either both be set"):
                    make_task(seed_type=seed_type, seed=seed)

    def test_rejects_invalid_seed(self):
        for seed in (-1, 1.5, "42", True):
            with self.subTest(seed=seed):
                expected_error = ValueError if seed == -1 else TypeError
                with self.assertRaises(expected_error):
                    make_task(seed_type="projection", seed=seed)

    def test_from_dict_rejects_missing_or_extra_fields(self):
        serialized = make_task().to_dict()
        serialized.pop("source")
        serialized["unexpected"] = "value"

        with self.assertRaisesRegex(
            ValueError,
            "missing=.*source.*extra=.*unexpected",
        ):
            ScaleTask.from_dict(serialized)


class RunPolicyTests(unittest.TestCase):
    def test_safe_defaults(self):
        policy = RunPolicy()

        self.assertEqual(policy.existing_output, "error")
        self.assertTrue(policy.validate_existing)
        self.assertTrue(policy.stop_on_error)
        self.assertTrue(policy.atomic_write)

    def test_accepts_supported_existing_output_modes(self):
        for mode in ("error", "resume", "replace"):
            with self.subTest(mode=mode):
                policy = RunPolicy(existing_output=mode)
                self.assertEqual(policy.existing_output, mode)

    def test_rejects_unknown_existing_output_mode(self):
        with self.assertRaisesRegex(ValueError, "existing_output must be one of"):
            RunPolicy(existing_output="skip")

    def test_resume_requires_validation(self):
        with self.assertRaisesRegex(ValueError, "resume requires"):
            RunPolicy(existing_output="resume", validate_existing=False)

    def test_non_resume_modes_can_disable_validation(self):
        for mode in ("error", "replace"):
            with self.subTest(mode=mode):
                policy = RunPolicy(
                    existing_output=mode,
                    validate_existing=False,
                )
                self.assertFalse(policy.validate_existing)

    def test_rejects_non_boolean_flags(self):
        fields = ("validate_existing", "stop_on_error", "atomic_write")
        for field_name in fields:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(TypeError, field_name):
                    RunPolicy(**{field_name: 1})

    def test_serializes_all_fields(self):
        policy = RunPolicy(
            existing_output="replace",
            validate_existing=False,
            stop_on_error=False,
            atomic_write=False,
        )

        self.assertEqual(
            policy.to_dict(),
            {
                "existing_output": "replace",
                "validate_existing": False,
                "stop_on_error": False,
                "atomic_write": False,
            },
        )


def make_record(task=None, **overrides):
    values = {
        "task": task or make_task(),
        "stage": "clustering",
        "status": "computed",
        "output_paths": ("partition.csv", "metadata.json"),
        "config_hash": "a" * 64,
        "started_at": "2026-08-09T10:00:00+00:00",
        "ended_at": "2026-08-09T10:01:00+00:00",
        "error_message": None,
        "reused": False,
        "computed": True,
    }
    values.update(overrides)
    return StageRunRecord(**values)


class StableConfigHashTests(unittest.TestCase):
    def test_mapping_order_does_not_change_hash(self):
        left = {"method": "hdbscan", "params": {"min_samples": 5, "alpha": 1.0}}
        right = {"params": {"alpha": 1.0, "min_samples": 5}, "method": "hdbscan"}

        self.assertEqual(stable_config_hash(left), stable_config_hash(right))

    def test_configuration_value_changes_hash(self):
        self.assertNotEqual(
            stable_config_hash({"min_samples": 5}),
            stable_config_hash({"min_samples": 6}),
        )

    def test_rejects_non_json_configuration(self):
        with self.assertRaises(TypeError):
            stable_config_hash({"invalid": object()})

    def test_rejects_non_finite_float(self):
        with self.assertRaises(ValueError):
            stable_config_hash({"invalid": float("nan")})


class StageRunRecordTests(unittest.TestCase):
    def test_computed_record_serializes_task_identity(self):
        record = make_record()
        serialized = record.to_dict()

        self.assertEqual(serialized["task"], record.task.to_dict())
        self.assertEqual(serialized["task_key"], record.task.task_key)
        self.assertEqual(serialized["output_paths"], ["partition.csv", "metadata.json"])

    def test_accepts_reused_record(self):
        record = make_record(
            status="reused",
            reused=True,
            computed=False,
        )
        self.assertTrue(record.reused)

    def test_failed_record_requires_error_message(self):
        with self.assertRaisesRegex(ValueError, "require.*error_message"):
            make_record(status="failed", computed=False)

    def test_successful_record_rejects_error_message(self):
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            make_record(error_message="unexpected")

    def test_rejects_inconsistent_status_flags(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            make_record(status="reused", reused=False, computed=False)

    def test_rejects_both_outcome_flags(self):
        with self.assertRaisesRegex(ValueError, "both reused and computed"):
            make_record(reused=True, computed=True)

    def test_rejects_invalid_output_paths(self):
        with self.assertRaisesRegex(TypeError, "must be a tuple"):
            make_record(output_paths=["partition.csv"])
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            make_record(output_paths=("",))


class StageRunReportTests(unittest.TestCase):
    def test_summarizes_outcomes(self):
        records = (
            make_record(task=make_task(scale=16)),
            make_record(
                task=make_task(scale=32),
                status="reused",
                reused=True,
                computed=False,
            ),
            make_record(
                task=make_task(scale=64),
                status="failed",
                computed=False,
                error_message="failure",
                output_paths=(),
            ),
        )
        report = StageRunReport(stage="clustering", records=records)

        self.assertEqual(report.computed_count, 1)
        self.assertEqual(report.reused_count, 1)
        self.assertEqual(report.failed_count, 1)
        self.assertFalse(report.succeeded)
        self.assertEqual(len(report.to_dict()["records"]), 3)

    def test_rejects_mixed_stages(self):
        with self.assertRaisesRegex(ValueError, "belong to the report stage"):
            StageRunReport(
                stage="clustering",
                records=(make_record(stage="metrics"),),
            )

    def test_rejects_duplicate_task_identity(self):
        with self.assertRaisesRegex(ValueError, "duplicate task identities"):
            StageRunReport(
                stage="clustering",
                records=(make_record(), make_record(status="computed")),
            )


if __name__ == "__main__":
    unittest.main()
