from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import unittest

from cluster_anlys.experiment_orchestrator import (
    CallableExperimentStage,
    DataProduct,
    DataProductSpec,
    DataProductStore,
    ExperimentStage,
    ExperimentOrchestrator,
    OrchestratorRunReport,
    OrchestratorStageRecord,
    StageExecution,
    StageRegistry,
)
from cluster_anlys.pipeline_core import (
    ScaleTask,
    StageRunRecord,
    StageRunReport,
)


class BaseResult:
    pass


class SpecificResult(BaseResult):
    pass


class OtherResult:
    pass


class DataProductSpecTests(unittest.TestCase):
    def test_declares_stable_key_and_runtime_type(self):
        spec = DataProductSpec("metrics.os", BaseResult)
        self.assertEqual(spec.key, "metrics.os")
        self.assertTrue(spec.type_name.endswith(".BaseResult"))
        spec.validate_value(SpecificResult())

    def test_rejects_invalid_key_and_unbounded_type(self):
        with self.assertRaises(ValueError):
            DataProductSpec("metrics/os", BaseResult)
        with self.assertRaises(ValueError):
            DataProductSpec("1metrics", BaseResult)
        with self.assertRaises(TypeError):
            DataProductSpec("metrics", "BaseResult")
        with self.assertRaises(ValueError):
            DataProductSpec("metrics", object)


class DataProductTests(unittest.TestCase):
    def test_validates_value_and_exposes_provenance_summary(self):
        value = SpecificResult()
        product = DataProduct(
            DataProductSpec("metrics.os", BaseResult),
            value,
            producer_stage="os-stage",
        )
        summary = product.to_summary()
        self.assertIs(product.value, value)
        self.assertEqual(summary["producer_stage"], "os-stage")
        self.assertTrue(summary["declared_type"].endswith(".BaseResult"))
        self.assertTrue(summary["actual_type"].endswith(".SpecificResult"))

    def test_is_frozen_and_rejects_wrong_value_or_producer(self):
        spec = DataProductSpec("metrics.os", BaseResult)
        with self.assertRaises(TypeError):
            DataProduct(spec, OtherResult())
        with self.assertRaises(ValueError):
            DataProduct(spec, BaseResult(), producer_stage="bad stage")
        product = DataProduct(spec, BaseResult())
        with self.assertRaises(FrozenInstanceError):
            product.producer_stage = "changed"


class DataProductStoreTests(unittest.TestCase):
    def test_publish_returns_new_store_and_preserves_original(self):
        empty = DataProductStore()
        product = DataProduct(
            DataProductSpec("partitions.main", BaseResult),
            SpecificResult(),
            producer_stage="clustering",
        )
        populated = empty.publish(product)
        self.assertEqual(empty.keys, ())
        self.assertEqual(populated.keys, ("partitions.main",))
        self.assertIs(populated.product("partitions.main"), product)

    def test_require_allows_declared_subtype_for_base_requirement(self):
        value = SpecificResult()
        store = DataProductStore((DataProduct(
            DataProductSpec("metrics.os", SpecificResult),
            value,
            producer_stage="os-stage",
        ),))
        required = DataProductSpec("metrics.os", BaseResult)
        self.assertIs(store.require(required), value)

    def test_require_rejects_missing_or_incompatible_declaration(self):
        store = DataProductStore((DataProduct(
            DataProductSpec("metrics.os", BaseResult),
            SpecificResult(),
        ),))
        with self.assertRaises(KeyError):
            store.require(DataProductSpec("metrics.mse", BaseResult))
        with self.assertRaises(TypeError):
            store.require(DataProductSpec("metrics.os", SpecificResult))

    def test_rejects_duplicate_keys_in_constructor_and_publish(self):
        first = DataProduct(DataProductSpec("metrics.os", BaseResult), BaseResult())
        second = DataProduct(DataProductSpec("metrics.os", BaseResult), BaseResult())
        with self.assertRaises(ValueError):
            DataProductStore((first, second))
        store = DataProductStore((first,))
        with self.assertRaises(ValueError):
            store.publish(second)

    def test_publish_many_is_atomic_and_preserves_order(self):
        first = DataProduct(DataProductSpec("metrics.os", BaseResult), BaseResult())
        second = DataProduct(DataProductSpec("metrics.mse", BaseResult), BaseResult())
        third = DataProduct(DataProductSpec("statistics.main", BaseResult), BaseResult())
        store = DataProductStore((first,)).publish_many((second, third))
        self.assertEqual(store.keys, (
            "metrics.os",
            "metrics.mse",
            "statistics.main",
        ))
        with self.assertRaises(ValueError):
            store.publish_many((second,))
        self.assertEqual(store.keys, (
            "metrics.os",
            "metrics.mse",
            "statistics.main",
        ))

    def test_summary_omits_values(self):
        store = DataProductStore((DataProduct(
            DataProductSpec("metrics.os", BaseResult),
            BaseResult(),
        ),))
        summary = store.to_summary()
        self.assertEqual(len(summary), 1)
        self.assertNotIn("value", summary[0])


class StageExecutionTests(unittest.TestCase):
    def test_exposes_unique_product_keys(self):
        product = DataProduct(
            DataProductSpec("metrics.os", BaseResult),
            BaseResult(),
            producer_stage="os-stage",
        )
        execution = StageExecution((product,))
        self.assertEqual(execution.product_keys, ("metrics.os",))
        with self.assertRaises(ValueError):
            StageExecution(())
        with self.assertRaises(TypeError):
            StageExecution((product,), native_report="not-a-report")


class CallableExperimentStageTests(unittest.TestCase):
    def make_stage(self, executor):
        return CallableExperimentStage(
            stage_id="statistics-stage",
            dependencies=("metric-stage",),
            required_products=(DataProductSpec("metrics.main", BaseResult),),
            provided_products=(
                DataProductSpec("statistics.main", BaseResult),
            ),
            executor=executor,
        )

    def test_satisfies_protocol_and_executes_declared_contract(self):
        input_value = SpecificResult()
        output_value = SpecificResult()
        input_store = DataProductStore((DataProduct(
            DataProductSpec("metrics.main", SpecificResult),
            input_value,
            producer_stage="metric-stage",
        ),))
        calls = []

        def executor(store):
            calls.append(store.require(DataProductSpec("metrics.main", BaseResult)))
            return StageExecution((DataProduct(
                DataProductSpec("statistics.main", SpecificResult),
                output_value,
                producer_stage="statistics-stage",
            ),))

        stage = self.make_stage(executor)
        execution = stage.execute(input_store)
        self.assertIsInstance(stage, ExperimentStage)
        self.assertEqual(calls, [input_value])
        self.assertIs(execution.products[0].value, output_value)

    def test_validates_required_products_before_calling_executor(self):
        calls = []

        def executor(store):
            calls.append(store)
            raise AssertionError("executor must not be called")

        stage = self.make_stage(executor)
        with self.assertRaises(KeyError):
            stage.execute(DataProductStore())
        self.assertEqual(calls, [])

    def test_rejects_wrong_output_key_producer_and_type(self):
        input_store = DataProductStore((DataProduct(
            DataProductSpec("metrics.main", BaseResult),
            BaseResult(),
            producer_stage="metric-stage",
        ),))

        def wrong_key(store):
            return StageExecution((DataProduct(
                DataProductSpec("statistics.other", BaseResult),
                BaseResult(),
                producer_stage="statistics-stage",
            ),))

        with self.assertRaises(ValueError):
            self.make_stage(wrong_key).execute(input_store)

        def wrong_producer(store):
            return StageExecution((DataProduct(
                DataProductSpec("statistics.main", BaseResult),
                BaseResult(),
                producer_stage="other-stage",
            ),))

        with self.assertRaises(ValueError):
            self.make_stage(wrong_producer).execute(input_store)

        def wrong_type(store):
            return StageExecution((DataProduct(
                DataProductSpec("statistics.main", OtherResult),
                OtherResult(),
                producer_stage="statistics-stage",
            ),))

        with self.assertRaises(TypeError):
            self.make_stage(wrong_type).execute(input_store)

    def test_rejects_invalid_stage_declarations_and_result(self):
        def executor(store):
            return "not-an-execution"

        stage = self.make_stage(executor)
        input_store = DataProductStore((DataProduct(
            DataProductSpec("metrics.main", BaseResult),
            BaseResult(),
            producer_stage="metric-stage",
        ),))
        with self.assertRaises(TypeError):
            stage.execute(input_store)
        with self.assertRaises(ValueError):
            CallableExperimentStage(
                stage_id="metric-stage",
                dependencies=("metric-stage",),
                required_products=(),
                provided_products=(DataProductSpec("metrics.main", BaseResult),),
                executor=lambda store: None,
            )


def make_registry_stage(
    stage_id,
    *,
    dependencies=(),
    required_products=(),
    provided_products=None,
):
    provided = provided_products or (
        DataProductSpec(f"outputs.{stage_id}", BaseResult),
    )
    return CallableExperimentStage(
        stage_id=stage_id,
        dependencies=dependencies,
        required_products=required_products,
        provided_products=provided,
        executor=lambda store: StageExecution(tuple(
            DataProduct(spec, spec.value_type(), producer_stage=stage_id)
            for spec in provided
        )),
    )


class StageRegistryTests(unittest.TestCase):
    def make_branching_registry(self):
        partition_spec = DataProductSpec("partitions.main", SpecificResult)
        os_spec = DataProductSpec("metrics.os", SpecificResult)
        mse_spec = DataProductSpec("metrics.mse", SpecificResult)
        combined_spec = DataProductSpec("metrics.combined", SpecificResult)
        clustering = make_registry_stage(
            "clustering",
            provided_products=(partition_spec,),
        )
        os_stage = make_registry_stage(
            "os-stage",
            dependencies=("clustering",),
            required_products=(DataProductSpec("partitions.main", BaseResult),),
            provided_products=(os_spec,),
        )
        mse_stage = make_registry_stage(
            "mse-stage",
            dependencies=("clustering",),
            required_products=(DataProductSpec("partitions.main", BaseResult),),
            provided_products=(mse_spec,),
        )
        combine = make_registry_stage(
            "combine-metrics",
            dependencies=("os-stage", "mse-stage"),
            required_products=(
                DataProductSpec("metrics.os", BaseResult),
                DataProductSpec("metrics.mse", BaseResult),
            ),
            provided_products=(combined_spec,),
        )
        statistics = make_registry_stage(
            "statistics",
            dependencies=("combine-metrics",),
            required_products=(
                DataProductSpec("metrics.combined", BaseResult),
            ),
        )
        plotting = make_registry_stage(
            "plotting",
            dependencies=("combine-metrics",),
            required_products=(
                DataProductSpec("metrics.combined", BaseResult),
            ),
        )
        return StageRegistry((
            clustering,
            os_stage,
            mse_stage,
            combine,
            statistics,
            plotting,
        ))

    def test_builds_stable_dependency_closure_for_requested_stages(self):
        registry = self.make_branching_registry()
        self.assertEqual(registry.execution_order(("statistics", "plotting")), (
            "clustering",
            "os-stage",
            "mse-stage",
            "combine-metrics",
            "statistics",
            "plotting",
        ))
        self.assertEqual(registry.execution_order(("mse-stage",)), (
            "clustering",
            "mse-stage",
        ))

    def test_accepts_external_product_and_subtype_guarantee(self):
        stage = make_registry_stage(
            "statistics",
            required_products=(DataProductSpec("metrics.external", BaseResult),),
        )
        registry = StageRegistry(
            (stage,),
            external_products=(
                DataProductSpec("metrics.external", SpecificResult),
            ),
        )
        self.assertEqual(registry.stage_ids, ("statistics",))

    def test_rejects_missing_dependency_and_cycle(self):
        missing = make_registry_stage(
            "statistics",
            dependencies=("missing-stage",),
        )
        with self.assertRaises(ValueError):
            StageRegistry((missing,))
        first = make_registry_stage("first", dependencies=("second",))
        second = make_registry_stage("second", dependencies=("first",))
        with self.assertRaises(ValueError):
            StageRegistry((first, second))

    def test_rejects_duplicate_stage_and_product_producer(self):
        first = make_registry_stage("first")
        with self.assertRaises(ValueError):
            StageRegistry((first, first))
        shared_spec = DataProductSpec("metrics.shared", BaseResult)
        producer_a = make_registry_stage(
            "producer-a",
            provided_products=(shared_spec,),
        )
        producer_b = make_registry_stage(
            "producer-b",
            provided_products=(shared_spec,),
        )
        with self.assertRaises(ValueError):
            StageRegistry((producer_a, producer_b))

    def test_rejects_requirement_from_non_dependency_or_unknown_source(self):
        metric_spec = DataProductSpec("metrics.main", BaseResult)
        producer = make_registry_stage(
            "metric-stage",
            provided_products=(metric_spec,),
        )
        unrelated = make_registry_stage("unrelated")
        consumer = make_registry_stage(
            "statistics",
            dependencies=("unrelated",),
            required_products=(metric_spec,),
        )
        with self.assertRaises(ValueError):
            StageRegistry((producer, unrelated, consumer))
        unknown_consumer = make_registry_stage(
            "plotting",
            required_products=(DataProductSpec("metrics.unknown", BaseResult),),
        )
        with self.assertRaises(ValueError):
            StageRegistry((unknown_consumer,))

    def test_rejects_incompatible_source_type_and_bad_requests(self):
        external_spec = DataProductSpec("metrics.external", BaseResult)
        stage = make_registry_stage(
            "statistics",
            required_products=(
                DataProductSpec("metrics.external", SpecificResult),
            ),
        )
        with self.assertRaises(TypeError):
            StageRegistry((stage,), external_products=(external_spec,))
        registry = self.make_branching_registry()
        with self.assertRaises(ValueError):
            registry.execution_order(())
        with self.assertRaises(ValueError):
            registry.execution_order(("statistics", "statistics"))
        with self.assertRaises(KeyError):
            registry.execution_order(("missing-stage",))


def timestamp(second):
    return datetime(2026, 8, 9, 12, 0, second, tzinfo=timezone.utc).isoformat()


class OrchestratorStageRecordTests(unittest.TestCase):
    def test_computed_record_preserves_optional_native_report(self):
        native_report = StageRunReport(stage="partition_metric", records=())
        record = OrchestratorStageRecord(
            stage_id="os-stage",
            status="computed",
            dependencies=("clustering",),
            consumed_product_keys=("partitions.main",),
            produced_product_keys=("metrics.os",),
            started_at=timestamp(0),
            ended_at=timestamp(1),
            native_report=native_report,
        )
        self.assertIs(record.native_report, native_report)
        self.assertTrue(record.to_dict()["native_report"]["succeeded"])

    def test_failed_record_requires_error_and_publishes_nothing(self):
        record = OrchestratorStageRecord(
            stage_id="statistics",
            status="failed",
            dependencies=("metric-stage",),
            consumed_product_keys=("metrics.main",),
            produced_product_keys=(),
            started_at=timestamp(1),
            ended_at=timestamp(2),
            error_message="numeric failure",
        )
        self.assertEqual(record.error_message, "numeric failure")
        with self.assertRaises(ValueError):
            OrchestratorStageRecord(
                stage_id="statistics",
                status="failed",
                dependencies=(),
                consumed_product_keys=(),
                produced_product_keys=("statistics.main",),
                started_at=timestamp(1),
                ended_at=timestamp(2),
                error_message="failure",
            )

    def test_rejects_invalid_time_and_computed_failure_fields(self):
        with self.assertRaises(ValueError):
            OrchestratorStageRecord(
                stage_id="statistics",
                status="computed",
                dependencies=(),
                consumed_product_keys=(),
                produced_product_keys=("statistics.main",),
                started_at=timestamp(2),
                ended_at=timestamp(1),
            )
        with self.assertRaises(ValueError):
            OrchestratorStageRecord(
                stage_id="statistics",
                status="computed",
                dependencies=(),
                consumed_product_keys=(),
                produced_product_keys=("statistics.main",),
                started_at=timestamp(1),
                ended_at=timestamp(2),
                error_message="not allowed",
            )


class OrchestratorRunReportTests(unittest.TestCase):
    def make_computed_record(self):
        return OrchestratorStageRecord(
            stage_id="metric-stage",
            status="computed",
            dependencies=(),
            consumed_product_keys=("partitions.external",),
            produced_product_keys=("metrics.main",),
            started_at=timestamp(0),
            ended_at=timestamp(1),
        )

    def make_failed_record(self):
        return OrchestratorStageRecord(
            stage_id="statistics",
            status="failed",
            dependencies=("metric-stage",),
            consumed_product_keys=("metrics.main",),
            produced_product_keys=(),
            started_at=timestamp(1),
            ended_at=timestamp(2),
            error_message="failure",
        )

    def make_store(self):
        return DataProductStore((
            DataProduct(
                DataProductSpec("partitions.external", BaseResult),
                BaseResult(),
            ),
            DataProduct(
                DataProductSpec("metrics.main", BaseResult),
                BaseResult(),
                producer_stage="metric-stage",
            ),
        ))

    def test_summarizes_success_without_serializing_values(self):
        report = OrchestratorRunReport(
            requested_stage_ids=("metric-stage",),
            execution_order=("metric-stage",),
            records=(self.make_computed_record(),),
            final_products=self.make_store(),
        )
        self.assertTrue(report.succeeded)
        self.assertEqual(report.computed_count, 1)
        summary = report.to_dict()
        self.assertEqual(summary["failed_count"], 0)
        self.assertNotIn("value", summary["final_products"][0])

    def test_summarizes_partial_failure_and_preserves_completed_products(self):
        report = OrchestratorRunReport(
            requested_stage_ids=("statistics",),
            execution_order=("metric-stage", "statistics"),
            records=(self.make_computed_record(), self.make_failed_record()),
            final_products=self.make_store(),
        )
        self.assertFalse(report.succeeded)
        self.assertEqual(report.computed_count, 1)
        self.assertEqual(report.failed_count, 1)
        self.assertIn("metrics.main", report.final_products)

    def test_rejects_wrong_record_order_and_missing_final_product(self):
        with self.assertRaises(ValueError):
            OrchestratorRunReport(
                requested_stage_ids=("statistics",),
                execution_order=("metric-stage", "statistics"),
                records=(self.make_failed_record(),),
                final_products=self.make_store(),
            )
        with self.assertRaises(KeyError):
            OrchestratorRunReport(
                requested_stage_ids=("metric-stage",),
                execution_order=("metric-stage",),
                records=(self.make_computed_record(),),
                final_products=DataProductStore(),
            )
        with self.assertRaises(ValueError):
            CallableExperimentStage(
                stage_id="metric-stage",
                dependencies=(),
                required_products=(DataProductSpec("metrics.main", BaseResult),),
                provided_products=(DataProductSpec("metrics.main", BaseResult),),
                executor=lambda store: None,
            )


class SequenceClock:
    def __init__(self):
        self.call_count = 0

    def __call__(self):
        value = datetime(
            2026,
            8,
            9,
            12,
            0,
            self.call_count,
            tzinfo=timezone.utc,
        )
        self.call_count += 1
        return value


class ExperimentOrchestratorTests(unittest.TestCase):
    def make_linear_registry(self, second_executor=None):
        input_spec = DataProductSpec("inputs.main", SpecificResult)
        intermediate_spec = DataProductSpec("metrics.main", SpecificResult)
        final_spec = DataProductSpec("statistics.main", SpecificResult)

        def first_executor(store):
            store.require(DataProductSpec("inputs.main", BaseResult))
            return StageExecution((DataProduct(
                intermediate_spec,
                SpecificResult(),
                producer_stage="metric-stage",
            ),))

        def default_second_executor(store):
            store.require(DataProductSpec("metrics.main", BaseResult))
            return StageExecution((DataProduct(
                final_spec,
                SpecificResult(),
                producer_stage="statistics",
            ),))

        metric_stage = CallableExperimentStage(
            stage_id="metric-stage",
            dependencies=(),
            required_products=(
                DataProductSpec("inputs.main", BaseResult),
            ),
            provided_products=(intermediate_spec,),
            executor=first_executor,
        )
        statistics = CallableExperimentStage(
            stage_id="statistics",
            dependencies=("metric-stage",),
            required_products=(
                DataProductSpec("metrics.main", BaseResult),
            ),
            provided_products=(final_spec,),
            executor=second_executor or default_second_executor,
        )
        registry = StageRegistry(
            (metric_stage, statistics),
            external_products=(input_spec,),
        )
        initial_store = DataProductStore((DataProduct(
            input_spec,
            SpecificResult(),
        ),))
        return registry, initial_store

    def make_failed_native_report(self):
        task = ScaleTask(
            analysis_id="test",
            model_name="model",
            space_name="space",
            scale=2,
            source="synthetic",
        )
        return StageRunReport(
            stage="statistics",
            records=(StageRunRecord(
                task=task,
                stage="statistics",
                status="failed",
                output_paths=(),
                config_hash="test-config",
                started_at=timestamp(0),
                ended_at=timestamp(1),
                error_message="task failure",
            ),),
        )

    def test_executes_dependency_order_and_publishes_products(self):
        registry, initial_store = self.make_linear_registry()
        clock = SequenceClock()
        report = ExperimentOrchestrator(registry, clock).run(
            ("statistics",),
            initial_store,
        )
        self.assertTrue(report.succeeded)
        self.assertEqual(report.execution_order, (
            "metric-stage",
            "statistics",
        ))
        self.assertEqual(report.final_products.keys, (
            "inputs.main",
            "metrics.main",
            "statistics.main",
        ))
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(report.records[0].started_at, timestamp(0))
        self.assertEqual(report.records[1].ended_at, timestamp(3))

    def test_executor_exception_stops_with_partial_store(self):
        calls = []

        def failing_executor(store):
            calls.append(store.keys)
            raise ArithmeticError("numeric failure")

        registry, initial_store = self.make_linear_registry(failing_executor)
        report = ExperimentOrchestrator(registry, SequenceClock()).run(
            ("statistics",),
            initial_store,
        )
        self.assertFalse(report.succeeded)
        self.assertEqual(tuple(record.status for record in report.records), (
            "computed",
            "failed",
        ))
        self.assertEqual(report.final_products.keys, (
            "inputs.main",
            "metrics.main",
        ))
        self.assertEqual(calls, [("inputs.main", "metrics.main")])
        self.assertIn("ArithmeticError", report.records[-1].error_message)

    def test_failed_native_report_does_not_publish_stage_products(self):
        native_report = self.make_failed_native_report()

        def failed_native_executor(store):
            return StageExecution((DataProduct(
                DataProductSpec("statistics.main", SpecificResult),
                SpecificResult(),
                producer_stage="statistics",
            ),), native_report=native_report)

        registry, initial_store = self.make_linear_registry(
            failed_native_executor
        )
        report = ExperimentOrchestrator(registry, SequenceClock()).run(
            ("statistics",),
            initial_store,
        )
        self.assertEqual(report.records[-1].status, "failed")
        self.assertIs(report.records[-1].native_report, native_report)
        self.assertNotIn("statistics.main", report.final_products)

    def test_validation_failure_cannot_leak_any_returned_product(self):
        leaked_value = SpecificResult()

        def wrong_output_executor(store):
            return StageExecution((DataProduct(
                DataProductSpec("statistics.wrong", SpecificResult),
                leaked_value,
                producer_stage="statistics",
            ),))

        registry, initial_store = self.make_linear_registry(
            wrong_output_executor
        )
        report = ExperimentOrchestrator(registry, SequenceClock()).run(
            ("statistics",),
            initial_store,
        )
        self.assertEqual(report.records[-1].status, "failed")
        self.assertNotIn("statistics.wrong", report.final_products)
        self.assertNotIn("statistics.main", report.final_products)

    def test_validates_required_external_products_before_execution(self):
        registry, _ = self.make_linear_registry()
        calls = []
        orchestrator = ExperimentOrchestrator(
            registry,
            lambda: calls.append(True),
        )
        with self.assertRaises(KeyError):
            orchestrator.run(("statistics",), DataProductStore())
        self.assertEqual(calls, [])

    def test_rejects_undeclared_or_non_external_initial_products(self):
        registry, _ = self.make_linear_registry()
        undeclared = DataProductStore((DataProduct(
            DataProductSpec("inputs.other", SpecificResult),
            SpecificResult(),
        ),))
        with self.assertRaises(ValueError):
            ExperimentOrchestrator(registry).run(("statistics",), undeclared)

        wrong_producer = DataProductStore((DataProduct(
            DataProductSpec("inputs.main", SpecificResult),
            SpecificResult(),
            producer_stage="seed-stage",
        ),))
        with self.assertRaises(ValueError):
            ExperimentOrchestrator(registry).run(
                ("statistics",),
                wrong_producer,
            )


if __name__ == "__main__":
    unittest.main()
