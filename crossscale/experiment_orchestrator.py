"""Typed data-product contracts for experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Callable, Protocol, runtime_checkable

from .pipeline_core import StageRunReport


PRODUCT_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
ORCHESTRATOR_STAGE_STATUSES = ("computed", "failed")


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not PRODUCT_KEY_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must start with a letter and contain only letters, "
            "digits, dots, underscores, or hyphens"
        )


@dataclass(frozen=True)
class DataProductSpec:
    """Declare one named runtime-typed data boundary between stages."""

    key: str
    value_type: type

    def __post_init__(self) -> None:
        _validate_identifier(self.key, "key")
        if not isinstance(self.value_type, type):
            raise TypeError("value_type must be a runtime type")
        if self.value_type is object:
            raise ValueError("value_type must be narrower than object")

    @property
    def type_name(self) -> str:
        """Return the fully qualified declared type name."""

        return f"{self.value_type.__module__}.{self.value_type.__qualname__}"

    def validate_value(self, value: object) -> None:
        """Require one value to satisfy the declared runtime type."""

        if not isinstance(value, self.value_type):
            raise TypeError(
                f"product {self.key!r} requires {self.type_name}, "
                f"got {type(value).__module__}.{type(value).__qualname__}"
            )


@dataclass(frozen=True, eq=False)
class DataProduct:
    """Bind one product specification to a value and producer identity."""

    spec: DataProductSpec
    value: object
    producer_stage: str = "external"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, DataProductSpec):
            raise TypeError("spec must be a DataProductSpec")
        _validate_identifier(self.producer_stage, "producer_stage")
        self.spec.validate_value(self.value)

    def to_summary(self) -> dict[str, Any]:
        """Return JSON-compatible product provenance without serializing value."""

        return {
            "key": self.spec.key,
            "declared_type": self.spec.type_name,
            "actual_type": (
                f"{type(self.value).__module__}.{type(self.value).__qualname__}"
            ),
            "producer_stage": self.producer_stage,
        }


@dataclass(frozen=True, eq=False)
class DataProductStore:
    """Hold immutable product bindings and return new stores on publication."""

    products: tuple[DataProduct, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.products, tuple):
            raise TypeError("products must be a tuple")
        if any(not isinstance(product, DataProduct) for product in self.products):
            raise TypeError("products must contain DataProduct objects")
        keys = tuple(product.spec.key for product in self.products)
        if len(keys) != len(set(keys)):
            raise ValueError("data product store contains duplicate keys")

    @property
    def keys(self) -> tuple[str, ...]:
        """Return product keys in publication order."""

        return tuple(product.spec.key for product in self.products)

    def __len__(self) -> int:
        return len(self.products)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.keys

    def product(self, key: str) -> DataProduct:
        """Return one product binding by key."""

        _validate_identifier(key, "key")
        for product in self.products:
            if product.spec.key == key:
                return product
        raise KeyError(f"data product not found: {key}")

    def require(self, spec: DataProductSpec) -> object:
        """Return a value after validating declared producer compatibility."""

        if not isinstance(spec, DataProductSpec):
            raise TypeError("spec must be a DataProductSpec")
        product = self.product(spec.key)
        if not issubclass(product.spec.value_type, spec.value_type):
            raise TypeError(
                f"product {spec.key!r} declares {product.spec.type_name}, "
                f"which does not satisfy required {spec.type_name}"
            )
        spec.validate_value(product.value)
        return product.value

    def publish(self, product: DataProduct) -> "DataProductStore":
        """Return a new store containing one additional unique product."""

        if not isinstance(product, DataProduct):
            raise TypeError("product must be a DataProduct")
        if product.spec.key in self:
            raise ValueError(f"data product key already exists: {product.spec.key}")
        return DataProductStore(self.products + (product,))

    def publish_many(
        self,
        products: tuple[DataProduct, ...],
    ) -> "DataProductStore":
        """Atomically validate and publish multiple products in tuple order."""

        if not isinstance(products, tuple):
            raise TypeError("products must be a tuple")
        candidate = DataProductStore(products)
        overlap = set(self.keys).intersection(candidate.keys)
        if overlap:
            raise ValueError(
                f"data product keys already exist: {sorted(overlap)}"
            )
        return DataProductStore(self.products + candidate.products)

    def to_summary(self) -> tuple[dict[str, Any], ...]:
        """Return product provenance in publication order."""

        return tuple(product.to_summary() for product in self.products)


@dataclass(frozen=True)
class StageExecution:
    """Return products and an optional native task-level report from one stage."""

    products: tuple[DataProduct, ...]
    native_report: StageRunReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.products, tuple) or not self.products:
            raise ValueError("products must be a non-empty tuple")
        DataProductStore(self.products)
        if self.native_report is not None and not isinstance(
            self.native_report,
            StageRunReport,
        ):
            raise TypeError("native_report must be a StageRunReport when set")

    @property
    def product_keys(self) -> tuple[str, ...]:
        """Return produced keys in declared execution order."""

        return tuple(product.spec.key for product in self.products)


@runtime_checkable
class ExperimentStage(Protocol):
    """Execute one bound experiment stage over declared data products."""

    stage_id: str
    dependencies: tuple[str, ...]
    required_products: tuple[DataProductSpec, ...]
    provided_products: tuple[DataProductSpec, ...]

    def execute(self, store: DataProductStore) -> StageExecution:
        """Consume declared products and return exactly declared outputs."""


@dataclass(frozen=True)
class CallableExperimentStage:
    """Bind a typed stage declaration to one narrow execution callable."""

    stage_id: str
    dependencies: tuple[str, ...]
    required_products: tuple[DataProductSpec, ...]
    provided_products: tuple[DataProductSpec, ...]
    executor: Callable[[DataProductStore], StageExecution]

    def __post_init__(self) -> None:
        _validate_identifier(self.stage_id, "stage_id")
        if not isinstance(self.dependencies, tuple):
            raise TypeError("dependencies must be a tuple")
        for dependency in self.dependencies:
            _validate_identifier(dependency, "dependency")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("dependencies must be unique")
        if self.stage_id in self.dependencies:
            raise ValueError("a stage cannot depend on itself")
        for field_name in ("required_products", "provided_products"):
            specs = getattr(self, field_name)
            if not isinstance(specs, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(not isinstance(spec, DataProductSpec) for spec in specs):
                raise TypeError(f"{field_name} must contain DataProductSpec objects")
            keys = tuple(spec.key for spec in specs)
            if len(keys) != len(set(keys)):
                raise ValueError(f"{field_name} must contain unique keys")
        if not self.provided_products:
            raise ValueError("provided_products must be non-empty")
        overlap = {
            spec.key for spec in self.required_products
        }.intersection(spec.key for spec in self.provided_products)
        if overlap:
            raise ValueError(
                f"required and provided product keys must be disjoint: {sorted(overlap)}"
            )
        if not callable(self.executor):
            raise TypeError("executor must be callable")

    def execute(self, store: DataProductStore) -> StageExecution:
        """Validate inputs, call the bound runner, and validate exact outputs."""

        if not isinstance(store, DataProductStore):
            raise TypeError("store must be a DataProductStore")
        for required_spec in self.required_products:
            store.require(required_spec)
        execution = self.executor(store)
        if not isinstance(execution, StageExecution):
            raise TypeError("executor must return StageExecution")
        expected_keys = tuple(spec.key for spec in self.provided_products)
        if execution.product_keys != expected_keys:
            raise ValueError(
                "executor products must exactly match provided_products order"
            )
        for product, provided_spec in zip(
            execution.products,
            self.provided_products,
        ):
            if product.producer_stage != self.stage_id:
                raise ValueError(
                    "executor product producer_stage must match stage_id"
                )
            if not issubclass(product.spec.value_type, provided_spec.value_type):
                raise TypeError(
                    f"product {product.spec.key!r} declares incompatible type "
                    f"{product.spec.type_name}"
                )
            provided_spec.validate_value(product.value)
        return execution


@dataclass(frozen=True)
class StageRegistry:
    """Validate a static typed stage graph and derive stable execution order."""

    stages: tuple[ExperimentStage, ...]
    external_products: tuple[DataProductSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("stages must be a non-empty tuple")
        if any(not isinstance(stage, ExperimentStage) for stage in self.stages):
            raise TypeError("stages must satisfy ExperimentStage")
        for stage in self.stages:
            self._validate_stage_declaration(stage)
        stage_ids = self.stage_ids
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("stage ids must be unique")

        if not isinstance(self.external_products, tuple):
            raise TypeError("external_products must be a tuple")
        if any(
            not isinstance(spec, DataProductSpec)
            for spec in self.external_products
        ):
            raise TypeError("external_products must contain DataProductSpec objects")
        external_keys = tuple(spec.key for spec in self.external_products)
        if len(external_keys) != len(set(external_keys)):
            raise ValueError("external product keys must be unique")

        known_stage_ids = set(stage_ids)
        for stage in self.stages:
            missing = set(stage.dependencies) - known_stage_ids
            if missing:
                raise ValueError(
                    f"stage {stage.stage_id!r} has missing dependencies: "
                    f"{sorted(missing)}"
                )
        self._validate_acyclic()

        producers = self._producer_specs()
        overlap = set(external_keys).intersection(producers)
        if overlap:
            raise ValueError(
                f"external and stage-produced product keys overlap: {sorted(overlap)}"
            )
        self._validate_requirements(producers)

    @property
    def stage_ids(self) -> tuple[str, ...]:
        """Return registered stage ids in stable registration order."""

        return tuple(stage.stage_id for stage in self.stages)

    def stage(self, stage_id: str) -> ExperimentStage:
        """Return one registered stage by id."""

        _validate_identifier(stage_id, "stage_id")
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"stage not registered: {stage_id}")

    def execution_order(
        self,
        requested_stage_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return dependency closure in deterministic requested DFS order."""

        if not isinstance(requested_stage_ids, tuple) or not requested_stage_ids:
            raise ValueError("requested_stage_ids must be a non-empty tuple")
        for stage_id in requested_stage_ids:
            _validate_identifier(stage_id, "requested stage id")
        if len(requested_stage_ids) != len(set(requested_stage_ids)):
            raise ValueError("requested stage ids must be unique")
        missing = set(requested_stage_ids) - set(self.stage_ids)
        if missing:
            raise KeyError(f"requested stages are not registered: {sorted(missing)}")

        ordered = []
        visited = set()

        def visit(stage_id: str) -> None:
            if stage_id in visited:
                return
            stage = self.stage(stage_id)
            for dependency in stage.dependencies:
                visit(dependency)
            visited.add(stage_id)
            ordered.append(stage_id)

        for requested_stage_id in requested_stage_ids:
            visit(requested_stage_id)
        return tuple(ordered)

    @staticmethod
    def _validate_stage_declaration(stage: ExperimentStage) -> None:
        _validate_identifier(stage.stage_id, "stage_id")
        if not isinstance(stage.dependencies, tuple):
            raise TypeError("stage dependencies must be a tuple")
        for dependency in stage.dependencies:
            _validate_identifier(dependency, "dependency")
        if len(stage.dependencies) != len(set(stage.dependencies)):
            raise ValueError("stage dependencies must be unique")
        if stage.stage_id in stage.dependencies:
            raise ValueError("a stage cannot depend on itself")
        for field_name in ("required_products", "provided_products"):
            specs = getattr(stage, field_name)
            if not isinstance(specs, tuple):
                raise TypeError(f"stage {field_name} must be a tuple")
            if any(not isinstance(spec, DataProductSpec) for spec in specs):
                raise TypeError(
                    f"stage {field_name} must contain DataProductSpec objects"
                )
            keys = tuple(spec.key for spec in specs)
            if len(keys) != len(set(keys)):
                raise ValueError(f"stage {field_name} must contain unique keys")
        if not stage.provided_products:
            raise ValueError("stage provided_products must be non-empty")
        overlap = {
            spec.key for spec in stage.required_products
        }.intersection(spec.key for spec in stage.provided_products)
        if overlap:
            raise ValueError(
                f"stage required and provided product keys overlap: {sorted(overlap)}"
            )

    def _validate_acyclic(self) -> None:
        state: dict[str, str] = {}
        path: list[str] = []

        def visit(stage_id: str) -> None:
            current_state = state.get(stage_id)
            if current_state == "visited":
                return
            if current_state == "visiting":
                cycle_start = path.index(stage_id)
                cycle = path[cycle_start:] + [stage_id]
                raise ValueError(
                    f"stage dependency graph contains a cycle: {' -> '.join(cycle)}"
                )
            state[stage_id] = "visiting"
            path.append(stage_id)
            for dependency in self.stage(stage_id).dependencies:
                visit(dependency)
            path.pop()
            state[stage_id] = "visited"

        for stage_id in self.stage_ids:
            visit(stage_id)

    def _producer_specs(self) -> dict[str, tuple[str, DataProductSpec]]:
        producers: dict[str, tuple[str, DataProductSpec]] = {}
        for stage in self.stages:
            for spec in stage.provided_products:
                if spec.key in producers:
                    other_stage = producers[spec.key][0]
                    raise ValueError(
                        f"product {spec.key!r} has multiple producers: "
                        f"{other_stage!r}, {stage.stage_id!r}"
                    )
                producers[spec.key] = (stage.stage_id, spec)
        return producers

    def _validate_requirements(
        self,
        producers: dict[str, tuple[str, DataProductSpec]],
    ) -> None:
        external = {spec.key: spec for spec in self.external_products}
        for stage in self.stages:
            ancestors = self._ancestor_ids(stage.stage_id)
            for required_spec in stage.required_products:
                if required_spec.key in external:
                    source_spec = external[required_spec.key]
                    source_name = "external"
                elif required_spec.key in producers:
                    producer_id, source_spec = producers[required_spec.key]
                    source_name = producer_id
                    if producer_id not in ancestors:
                        raise ValueError(
                            f"stage {stage.stage_id!r} requires product "
                            f"{required_spec.key!r} from non-dependency stage "
                            f"{producer_id!r}"
                        )
                else:
                    raise ValueError(
                        f"stage {stage.stage_id!r} requires product "
                        f"{required_spec.key!r} with no declared source"
                    )
                if not issubclass(source_spec.value_type, required_spec.value_type):
                    raise TypeError(
                        f"product {required_spec.key!r} from {source_name!r} "
                        f"declares {source_spec.type_name}, which does not satisfy "
                        f"{required_spec.type_name}"
                    )

    def _ancestor_ids(self, stage_id: str) -> set[str]:
        ancestors = set()

        def collect(current_id: str) -> None:
            for dependency in self.stage(current_id).dependencies:
                if dependency not in ancestors:
                    ancestors.add(dependency)
                    collect(dependency)

        collect(stage_id)
        return ancestors


def _validated_timestamp(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return parsed


@dataclass(frozen=True)
class OrchestratorStageRecord:
    """Describe one stage-level orchestration outcome."""

    stage_id: str
    status: str
    dependencies: tuple[str, ...]
    consumed_product_keys: tuple[str, ...]
    produced_product_keys: tuple[str, ...]
    started_at: str
    ended_at: str
    error_message: str | None = None
    native_report: StageRunReport | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.stage_id, "stage_id")
        if self.status not in ORCHESTRATOR_STAGE_STATUSES:
            raise ValueError(
                f"status must be one of {ORCHESTRATOR_STAGE_STATUSES}"
            )
        for field_name in (
            "dependencies",
            "consumed_product_keys",
            "produced_product_keys",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            for value in values:
                _validate_identifier(value, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique values")
        if self.stage_id in self.dependencies:
            raise ValueError("a stage record cannot depend on itself")
        overlap = set(self.consumed_product_keys).intersection(
            self.produced_product_keys
        )
        if overlap:
            raise ValueError(
                f"consumed and produced product keys overlap: {sorted(overlap)}"
            )
        started = _validated_timestamp(self.started_at, "started_at")
        ended = _validated_timestamp(self.ended_at, "ended_at")
        if ended < started:
            raise ValueError("ended_at must not precede started_at")
        if self.native_report is not None and not isinstance(
            self.native_report,
            StageRunReport,
        ):
            raise TypeError("native_report must be a StageRunReport when set")
        if self.status == "computed":
            if not self.produced_product_keys:
                raise ValueError("computed records require produced products")
            if self.error_message is not None:
                raise ValueError("computed records cannot contain an error message")
            if self.native_report is not None and not self.native_report.succeeded:
                raise ValueError(
                    "computed records cannot contain a failed native report"
                )
        else:
            if self.produced_product_keys:
                raise ValueError("failed records cannot publish products")
            if (
                not isinstance(self.error_message, str)
                or not self.error_message.strip()
            ):
                raise ValueError("failed records require a non-empty error message")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible stage-level record."""

        return {
            "stage_id": self.stage_id,
            "status": self.status,
            "dependencies": list(self.dependencies),
            "consumed_product_keys": list(self.consumed_product_keys),
            "produced_product_keys": list(self.produced_product_keys),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error_message": self.error_message,
            "native_report": (
                None if self.native_report is None else self.native_report.to_dict()
            ),
        }


@dataclass(frozen=True)
class OrchestratorRunReport:
    """Collect ordered stage outcomes and the final typed product store."""

    requested_stage_ids: tuple[str, ...]
    execution_order: tuple[str, ...]
    records: tuple[OrchestratorStageRecord, ...]
    final_products: DataProductStore

    def __post_init__(self) -> None:
        for field_name in ("requested_stage_ids", "execution_order"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{field_name} must be a non-empty tuple")
            for value in values:
                _validate_identifier(value, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique values")
        if not set(self.requested_stage_ids).issubset(self.execution_order):
            raise ValueError("requested stages must appear in execution_order")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple")
        if any(
            not isinstance(record, OrchestratorStageRecord)
            for record in self.records
        ):
            raise TypeError("records must contain OrchestratorStageRecord objects")
        record_ids = tuple(record.stage_id for record in self.records)
        if record_ids != self.execution_order[:len(record_ids)]:
            raise ValueError(
                "record ids must be an ordered prefix of execution_order"
            )
        failed_indices = tuple(
            index
            for index, record in enumerate(self.records)
            if record.status == "failed"
        )
        if len(failed_indices) > 1:
            raise ValueError("an orchestrator report can contain at most one failure")
        if failed_indices and failed_indices[0] != len(self.records) - 1:
            raise ValueError("a failed record must be the final record")
        if not isinstance(self.final_products, DataProductStore):
            raise TypeError("final_products must be a DataProductStore")
        for record in self.records:
            if record.status != "computed":
                continue
            for key in record.produced_product_keys:
                product = self.final_products.product(key)
                if product.producer_stage != record.stage_id:
                    raise ValueError(
                        "final product producer does not match its stage record"
                    )

    @property
    def computed_count(self) -> int:
        return sum(record.status == "computed" for record in self.records)

    @property
    def failed_count(self) -> int:
        return sum(record.status == "failed" for record in self.records)

    @property
    def succeeded(self) -> bool:
        return self.failed_count == 0 and len(self.records) == len(self.execution_order)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report without serializing product values."""

        return {
            "requested_stage_ids": list(self.requested_stage_ids),
            "execution_order": list(self.execution_order),
            "records": [record.to_dict() for record in self.records],
            "computed_count": self.computed_count,
            "failed_count": self.failed_count,
            "succeeded": self.succeeded,
            "final_products": list(self.final_products.to_summary()),
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ExperimentOrchestrator:
    """Execute a validated stage closure and publish products atomically."""

    registry: StageRegistry
    clock: Callable[[], datetime] = field(
        default=_utc_now,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.registry, StageRegistry):
            raise TypeError("registry must be a StageRegistry")
        if not callable(self.clock):
            raise TypeError("clock must be callable")

    def run(
        self,
        requested_stage_ids: tuple[str, ...],
        initial_products: DataProductStore | None = None,
    ) -> OrchestratorRunReport:
        """Run the requested dependency closure until completion or failure."""

        execution_order = self.registry.execution_order(requested_stage_ids)
        store = initial_products or DataProductStore()
        if not isinstance(store, DataProductStore):
            raise TypeError("initial_products must be a DataProductStore")
        self._validate_initial_products(store, execution_order)

        records = []
        for stage_id in execution_order:
            stage = self.registry.stage(stage_id)
            started_at = self._timestamp()
            native_report = None
            try:
                execution = stage.execute(store)
                native_report = execution.native_report
                if native_report is not None and not native_report.succeeded:
                    raise RuntimeError(
                        "native stage report contains "
                        f"{native_report.failed_count} failed task(s)"
                    )
                next_store = store.publish_many(execution.products)
            except Exception as error:
                ended_at = self._timestamp()
                records.append(OrchestratorStageRecord(
                    stage_id=stage.stage_id,
                    status="failed",
                    dependencies=stage.dependencies,
                    consumed_product_keys=tuple(
                        spec.key for spec in stage.required_products
                    ),
                    produced_product_keys=(),
                    started_at=started_at,
                    ended_at=ended_at,
                    error_message=f"{type(error).__name__}: {error}",
                    native_report=native_report,
                ))
                break
            ended_at = self._timestamp()
            store = next_store
            records.append(OrchestratorStageRecord(
                stage_id=stage.stage_id,
                status="computed",
                dependencies=stage.dependencies,
                consumed_product_keys=tuple(
                    spec.key for spec in stage.required_products
                ),
                produced_product_keys=execution.product_keys,
                started_at=started_at,
                ended_at=ended_at,
                native_report=native_report,
            ))

        return OrchestratorRunReport(
            requested_stage_ids=requested_stage_ids,
            execution_order=execution_order,
            records=tuple(records),
            final_products=store,
        )

    def _validate_initial_products(
        self,
        store: DataProductStore,
        execution_order: tuple[str, ...],
    ) -> None:
        external_specs = {
            spec.key: spec for spec in self.registry.external_products
        }
        undeclared = set(store.keys) - set(external_specs)
        if undeclared:
            raise ValueError(
                f"initial products are not declared external: {sorted(undeclared)}"
            )
        for product in store.products:
            if product.producer_stage != "external":
                raise ValueError(
                    "initial products must use producer_stage='external'"
                )
            store.require(external_specs[product.spec.key])

        required_external_keys = {
            spec.key
            for stage_id in execution_order
            for spec in self.registry.stage(stage_id).required_products
            if spec.key in external_specs
        }
        missing = required_external_keys - set(store.keys)
        if missing:
            raise KeyError(
                f"required external products are missing: {sorted(missing)}"
            )

    def _timestamp(self) -> str:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetime")
        return value.isoformat()
