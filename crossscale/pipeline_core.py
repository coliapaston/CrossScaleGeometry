"""Shared contracts for modular analysis pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Optional


EXISTING_OUTPUT_MODES = frozenset({"error", "resume", "replace"})
STAGE_STATUSES = frozenset({"computed", "reused", "failed"})


def stable_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible configuration independently of mapping order."""

    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScaleTask:
    """Identify one model-space-scale unit of pipeline work."""

    analysis_id: str
    model_name: str
    space_name: str
    scale: int
    source: str
    seed_type: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        for field_name in ("analysis_id", "model_name", "space_name", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if isinstance(self.scale, bool) or not isinstance(self.scale, int):
            raise TypeError("scale must be an integer")
        if self.scale <= 0:
            raise ValueError("scale must be greater than zero")

        if (self.seed_type is None) != (self.seed is None):
            raise ValueError("seed_type and seed must either both be set or both be None")

        if self.seed_type is not None:
            if not isinstance(self.seed_type, str) or not self.seed_type.strip():
                raise ValueError("seed_type must be a non-empty string when seed is set")
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer when seed_type is set")
            if self.seed < 0:
                raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "analysis_id": self.analysis_id,
            "model_name": self.model_name,
            "space_name": self.space_name,
            "scale": self.scale,
            "source": self.source,
            "seed_type": self.seed_type,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScaleTask":
        """Construct a task from its serialized representation."""

        expected_fields = {
            "analysis_id",
            "model_name",
            "space_name",
            "scale",
            "source",
            "seed_type",
            "seed",
        }
        provided_fields = set(data)
        if provided_fields != expected_fields:
            missing = sorted(expected_fields - provided_fields)
            extra = sorted(provided_fields - expected_fields)
            raise ValueError(f"Invalid ScaleTask fields: missing={missing}, extra={extra}")
        return cls(**dict(data))

    def canonical_json(self) -> str:
        """Serialize identity fields with deterministic ordering and spacing."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def task_key(self) -> str:
        """Return a deterministic content-derived task identifier."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunPolicy:
    """Define how a pipeline handles outputs and task failures."""

    existing_output: str = "error"
    validate_existing: bool = True
    stop_on_error: bool = True
    atomic_write: bool = True

    def __post_init__(self) -> None:
        if self.existing_output not in EXISTING_OUTPUT_MODES:
            supported = ", ".join(sorted(EXISTING_OUTPUT_MODES))
            raise ValueError(
                f"existing_output must be one of {{{supported}}}, "
                f"got {self.existing_output!r}"
            )

        for field_name in ("validate_existing", "stop_on_error", "atomic_write"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")

        if self.existing_output == "resume" and not self.validate_existing:
            raise ValueError("resume requires validate_existing=True")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy representation."""

        return {
            "existing_output": self.existing_output,
            "validate_existing": self.validate_existing,
            "stop_on_error": self.stop_on_error,
            "atomic_write": self.atomic_write,
        }


@dataclass(frozen=True)
class StageRunRecord:
    """Describe the outcome of one stage task."""

    task: ScaleTask
    stage: str
    status: str
    output_paths: tuple[str, ...]
    config_hash: str
    started_at: str
    ended_at: str
    error_message: Optional[str] = None
    reused: bool = False
    computed: bool = False

    def __post_init__(self) -> None:
        for field_name in ("stage", "config_hash", "started_at", "ended_at"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        if self.status not in STAGE_STATUSES:
            supported = ", ".join(sorted(STAGE_STATUSES))
            raise ValueError(f"status must be one of {{{supported}}}, got {self.status!r}")

        if not isinstance(self.output_paths, tuple):
            raise TypeError("output_paths must be a tuple")
        if any(not isinstance(path, str) or not path.strip() for path in self.output_paths):
            raise ValueError("output_paths must contain only non-empty strings")

        if not isinstance(self.reused, bool) or not isinstance(self.computed, bool):
            raise TypeError("reused and computed must be booleans")
        if self.reused and self.computed:
            raise ValueError("a stage record cannot be both reused and computed")

        expected_flags = {
            "computed": (False, True),
            "reused": (True, False),
            "failed": (False, False),
        }
        if (self.reused, self.computed) != expected_flags[self.status]:
            raise ValueError(f"status={self.status!r} is inconsistent with outcome flags")

        if self.status == "failed":
            if not isinstance(self.error_message, str) or not self.error_message.strip():
                raise ValueError("failed records require a non-empty error_message")
        elif self.error_message is not None:
            raise ValueError("successful records cannot contain an error_message")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record representation."""

        return {
            "task": self.task.to_dict(),
            "task_key": self.task.task_key,
            "stage": self.stage,
            "status": self.status,
            "output_paths": list(self.output_paths),
            "config_hash": self.config_hash,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error_message": self.error_message,
            "reused": self.reused,
            "computed": self.computed,
        }


@dataclass(frozen=True)
class StageRunReport:
    """Collect all task outcomes from one pipeline stage run."""

    stage: str
    records: tuple[StageRunRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple")
        if any(record.stage != self.stage for record in self.records):
            raise ValueError("all records must belong to the report stage")

        task_keys = [record.task.task_key for record in self.records]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("a stage report cannot contain duplicate task identities")

    @property
    def computed_count(self) -> int:
        return sum(record.computed for record in self.records)

    @property
    def reused_count(self) -> int:
        return sum(record.reused for record in self.records)

    @property
    def failed_count(self) -> int:
        return sum(record.status == "failed" for record in self.records)

    @property
    def succeeded(self) -> bool:
        return self.failed_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report representation."""

        return {
            "stage": self.stage,
            "records": [record.to_dict() for record in self.records],
            "computed_count": self.computed_count,
            "reused_count": self.reused_count,
            "failed_count": self.failed_count,
            "succeeded": self.succeeded,
        }
