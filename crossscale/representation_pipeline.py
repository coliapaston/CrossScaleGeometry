"""Representation-construction pipelines with persistent output contracts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .pipeline_core import (
    RunPolicy,
    ScaleTask,
    StageRunRecord,
    StageRunReport,
    stable_config_hash,
)


RANDOMIZED_PCA_FILENAMES = {
    "matrix": "X_pca.npy",
    "components": "components.npy",
    "explained_ratio": "explained_ratio.npy",
    "mean": "mean.npy",
    "metadata": "meta.json",
}

REPRESENTATION_MANIFEST_COLUMNS = (
    "analysis_id",
    "model_name",
    "space_name",
    "scale",
    "source",
    "seed_type",
    "seed",
    "builder",
    "matrix_path",
    "components_path",
    "explained_ratio_path",
    "mean_path",
    "metadata_path",
    "n_samples",
    "n_features",
    "config_hash",
    "reused",
)


@dataclass(frozen=True)
class RandomizedPCAOutputPaths:
    """Hold the five files that make one randomized-PCA output complete."""

    task_dir: Path
    matrix: Path
    components: Path
    explained_ratio: Path
    mean: Path
    metadata: Path

    @property
    def all_files(self) -> tuple[Path, ...]:
        """Return data files first and the completion metadata file last."""

        return (
            self.matrix,
            self.components,
            self.explained_ratio,
            self.mean,
            self.metadata,
        )


class RandomizedPCAOutputLayout:
    """Resolve the repository's unpadded randomized-PCA directory layout."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def paths_for(self, task: ScaleTask) -> RandomizedPCAOutputPaths:
        """Resolve all saved files for one seeded scale task."""

        if task.seed is None:
            raise ValueError("randomized PCA tasks require a seed")
        task_dir = (
            self.output_root
            / task.model_name
            / "random_pca"
            / f"seed_{task.seed}"
            / f"pca_dim_{task.scale}"
        )
        return RandomizedPCAOutputPaths(
            task_dir=task_dir,
            matrix=task_dir / RANDOMIZED_PCA_FILENAMES["matrix"],
            components=task_dir / RANDOMIZED_PCA_FILENAMES["components"],
            explained_ratio=task_dir / RANDOMIZED_PCA_FILENAMES["explained_ratio"],
            mean=task_dir / RANDOMIZED_PCA_FILENAMES["mean"],
            metadata=task_dir / RANDOMIZED_PCA_FILENAMES["metadata"],
        )

    def manifest_path(self, analysis_id: str) -> Path:
        """Return the aggregate representation manifest path."""

        if not isinstance(analysis_id, str) or not analysis_id.strip():
            raise ValueError("analysis_id must be a non-empty string")
        return (
            self.output_root
            / analysis_id
            / "manifests"
            / "randomized_pca_representation_manifest.csv"
        )


def randomized_pca_matrix_path(root: Path, task: ScaleTask) -> Path:
    """Resolve a matrix path for ``SavedPerScaleProvider`` compatibility."""

    return RandomizedPCAOutputLayout(root).paths_for(task).matrix


RandomizedPCARunner = Callable[..., Any]


@runtime_checkable
class RepresentationBuilder(Protocol):
    """Construct and persist one scale-specific representation."""

    name: str

    def config(self) -> Mapping[str, Any]:
        """Return the task-independent numerical configuration."""

    def build(self, matrix: np.ndarray, task: ScaleTask, output_root: Path) -> None:
        """Build one task beneath the supplied output root."""


class RandomizedPCABuilder:
    """Adapt the existing independent randomized-PCA runner to one task."""

    name = "randomized_pca"

    def __init__(
        self,
        *,
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "auto",
        iterated_power: str | int = "auto",
        runner: RandomizedPCARunner | None = None,
    ) -> None:
        if isinstance(n_oversamples, bool) or not isinstance(n_oversamples, int):
            raise TypeError("n_oversamples must be an integer")
        if n_oversamples < 1:
            raise ValueError("n_oversamples must be greater than zero")
        if not isinstance(power_iteration_normalizer, str):
            raise TypeError("power_iteration_normalizer must be a string")
        if not (
            iterated_power == "auto"
            or (
                not isinstance(iterated_power, bool)
                and isinstance(iterated_power, int)
                and iterated_power >= 0
            )
        ):
            raise ValueError("iterated_power must be 'auto' or a non-negative integer")
        if runner is not None and not callable(runner):
            raise TypeError("runner must be callable or None")

        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.iterated_power = iterated_power
        self._runner = runner

    def config(self) -> Mapping[str, Any]:
        """Return the randomized solver settings used for configuration hashing."""

        return {
            "name": self.name,
            "svd_solver": "randomized",
            "n_oversamples": self.n_oversamples,
            "power_iteration_normalizer": self.power_iteration_normalizer,
            "iterated_power": self.iterated_power,
            "fit_strategy": "independent_per_scale",
        }

    def build(self, matrix: np.ndarray, task: ScaleTask, output_root: Path) -> None:
        """Fit randomized PCA independently at the task's exact scale and seed."""

        _validate_source_matrix(matrix)
        if task.seed is None:
            raise ValueError("randomized PCA tasks require a seed")
        max_dimension = min(matrix.shape)
        if task.scale > max_dimension:
            raise ValueError(
                f"task scale {task.scale} exceeds maximum PCA dimension "
                f"{max_dimension} for model {task.model_name!r}"
            )

        runner = self._runner
        if runner is None:
            from .pca_hdbscan import run_randomized_pca

            runner = run_randomized_pca
        runner(
            matrix,
            pca_dim=task.scale,
            pca_seed=task.seed,
            model_name=task.model_name,
            out_root=str(output_root),
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
            iterated_power=self.iterated_power,
        )


@dataclass(frozen=True)
class RepresentationManifestRow:
    """Describe one complete representation bundle for downstream stages."""

    task: ScaleTask
    builder: str
    matrix_path: str
    components_path: str
    explained_ratio_path: str
    mean_path: str
    metadata_path: str
    n_samples: int
    n_features: int
    config_hash: str
    reused: bool

    def __post_init__(self) -> None:
        for field_name in (
            "builder",
            "matrix_path",
            "components_path",
            "explained_ratio_path",
            "mean_path",
            "metadata_path",
            "config_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("n_samples", "n_features"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 1:
                raise ValueError(f"{field_name} must be greater than zero")
        if self.n_features != self.task.scale:
            raise ValueError("n_features must equal the task scale")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return one flat CSV-compatible manifest row."""

        return {
            **self.task.to_dict(),
            "builder": self.builder,
            "matrix_path": self.matrix_path,
            "components_path": self.components_path,
            "explained_ratio_path": self.explained_ratio_path,
            "mean_path": self.mean_path,
            "metadata_path": self.metadata_path,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "config_hash": self.config_hash,
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepresentationManifestRow":
        """Construct one row from CSV-compatible scalar values."""

        provided = set(data)
        expected = set(REPRESENTATION_MANIFEST_COLUMNS)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(
                f"Invalid representation manifest fields: missing={missing}, "
                f"extra={extra}"
            )
        seed_type = None if pd.isna(data["seed_type"]) else str(data["seed_type"])
        seed = _optional_csv_integer(data["seed"], "seed")
        task = ScaleTask(
            analysis_id=str(data["analysis_id"]),
            model_name=str(data["model_name"]),
            space_name=str(data["space_name"]),
            scale=_required_csv_integer(data["scale"], "scale"),
            source=str(data["source"]),
            seed_type=seed_type,
            seed=seed,
        )
        return cls(
            task=task,
            builder=str(data["builder"]),
            matrix_path=str(data["matrix_path"]),
            components_path=str(data["components_path"]),
            explained_ratio_path=str(data["explained_ratio_path"]),
            mean_path=str(data["mean_path"]),
            metadata_path=str(data["metadata_path"]),
            n_samples=_required_csv_integer(data["n_samples"], "n_samples"),
            n_features=_required_csv_integer(data["n_features"], "n_features"),
            config_hash=str(data["config_hash"]),
            reused=_csv_boolean(data["reused"], "reused"),
        )


@dataclass(frozen=True)
class RepresentationManifest:
    """Collect complete representation bundles for downstream consumers."""

    rows: tuple[RepresentationManifestRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        task_keys = [row.task.task_key for row in self.rows]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("representation manifest contains duplicate task identities")

    def to_frame(self) -> pd.DataFrame:
        """Return the canonical long manifest table."""

        frame = pd.DataFrame(
            [row.to_dict() for row in self.rows],
            columns=REPRESENTATION_MANIFEST_COLUMNS,
        )
        if not frame.empty:
            frame["seed"] = pd.array(frame["seed"], dtype="Int64")
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "RepresentationManifest":
        """Validate and construct a representation manifest from a DataFrame."""

        if tuple(frame.columns) != REPRESENTATION_MANIFEST_COLUMNS:
            raise ValueError(
                "representation manifest columns must exactly match canonical order"
            )
        return cls(
            rows=tuple(
                RepresentationManifestRow.from_dict(record)
                for record in frame.to_dict(orient="records")
            )
        )

    def write_csv(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the manifest without silently replacing an existing file."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Representation manifest already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(destination, index=False, encoding="utf-8")
        return destination

    @classmethod
    def read_csv(cls, path: str | Path) -> "RepresentationManifest":
        """Read and validate a canonical representation manifest."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Representation manifest not found: {source}")
        return cls.from_frame(pd.read_csv(source))


def validate_randomized_pca_output(
    paths: RandomizedPCAOutputPaths,
    *,
    expected_task: ScaleTask,
    expected_builder: str,
    expected_config_hash: str,
    reused: bool,
    canonical_paths: RandomizedPCAOutputPaths | None = None,
) -> RepresentationManifestRow:
    """Validate all arrays and provenance metadata for one complete bundle."""

    missing = [str(path) for path in paths.all_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete randomized PCA output; missing files: {missing}"
        )
    with paths.metadata.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("representation_schema_version") != 1:
        raise ValueError("Unsupported randomized PCA representation schema version")
    if metadata.get("task") != expected_task.to_dict():
        raise ValueError("saved task identity does not match expected task")
    if metadata.get("task_key") != expected_task.task_key:
        raise ValueError("saved task key does not match expected task")
    if metadata.get("builder") != expected_builder:
        raise ValueError("saved builder does not match expected builder")
    if metadata.get("config_hash") != expected_config_hash:
        raise ValueError("saved config hash does not match expected configuration")
    expected_paths = _canonical_path_metadata(canonical_paths or paths)
    if metadata.get("output_paths") != expected_paths:
        raise ValueError("saved output paths do not match the output layout")

    matrix = np.load(paths.matrix, mmap_mode="r")
    components = np.load(paths.components, mmap_mode="r")
    explained_ratio = np.load(paths.explained_ratio, mmap_mode="r")
    mean = np.load(paths.mean, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[1] != expected_task.scale:
        raise ValueError("saved PCA matrix shape does not match the task scale")
    n_samples = int(matrix.shape[0])
    n_input_features = int(mean.shape[0]) if mean.ndim == 1 else -1
    expected_shapes = {
        "components": (expected_task.scale, n_input_features),
        "explained_ratio": (expected_task.scale,),
        "mean": (n_input_features,),
    }
    actual_shapes = {
        "components": components.shape,
        "explained_ratio": explained_ratio.shape,
        "mean": mean.shape,
    }
    if n_input_features < 1 or actual_shapes != expected_shapes:
        raise ValueError(
            f"saved PCA bundle shapes are inconsistent: {actual_shapes}"
        )
    for name, array in (
        ("matrix", matrix),
        ("components", components),
        ("explained_ratio", explained_ratio),
        ("mean", mean),
    ):
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"saved {name} array must use a numeric dtype")
    if metadata.get("input_shape") != [n_samples, n_input_features]:
        raise ValueError("saved input shape does not match array shapes")
    if metadata.get("output_shape") != [n_samples, expected_task.scale]:
        raise ValueError("saved output shape does not match the PCA matrix")

    return RepresentationManifestRow(
        task=expected_task,
        builder=expected_builder,
        matrix_path=str(paths.matrix),
        components_path=str(paths.components),
        explained_ratio_path=str(paths.explained_ratio),
        mean_path=str(paths.mean),
        metadata_path=str(paths.metadata),
        n_samples=n_samples,
        n_features=expected_task.scale,
        config_hash=expected_config_hash,
        reused=reused,
    )


@dataclass(frozen=True)
class RepresentationPipelineRun:
    """Return the representation manifest and execution report."""

    manifest: RepresentationManifest
    report: StageRunReport
    manifest_path: Path


ModelMatrixLoader = Callable[[str], np.ndarray]


class RandomizedPCARepresentationPipeline:
    """Schedule independent model-scale-seed randomized-PCA fits."""

    stage = "representation_construction"

    def __init__(
        self,
        *,
        builder: RepresentationBuilder,
        output_layout: RandomizedPCAOutputLayout,
        matrix_loader: ModelMatrixLoader,
        run_policy: RunPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(builder, RepresentationBuilder):
            raise TypeError("builder must satisfy RepresentationBuilder")
        if not callable(matrix_loader):
            raise TypeError("matrix_loader must be callable")
        self.builder = builder
        self.output_layout = output_layout
        self.matrix_loader = matrix_loader
        self.run_policy = run_policy or RunPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def config(self) -> dict[str, Any]:
        """Return the complete task-independent construction configuration."""

        return {
            "schema_version": 1,
            "stage": self.stage,
            "builder": dict(self.builder.config()),
        }

    def run(self, tasks: tuple[ScaleTask, ...]) -> RepresentationPipelineRun:
        """Build or validate a homogeneous set of randomized-PCA tasks."""

        self._validate_tasks(tasks)
        analysis_id = tasks[0].analysis_id
        manifest_path = self.output_layout.manifest_path(analysis_id)
        config = self.config()
        config_hash = stable_config_hash(config)
        if self.run_policy.existing_output == "error":
            self._preflight_error_mode(tasks, manifest_path)

        rows: dict[str, RepresentationManifestRow] = {}
        records: dict[str, StageRunRecord] = {}
        pending_by_model: dict[str, list[ScaleTask]] = defaultdict(list)

        for task in tasks:
            paths = self.output_layout.paths_for(task)
            if self.run_policy.existing_output == "resume" and any(
                path.exists() for path in paths.all_files
            ):
                started_at = self._timestamp()
                try:
                    row = validate_randomized_pca_output(
                        paths,
                        expected_task=task,
                        expected_builder=self.builder.name,
                        expected_config_hash=config_hash,
                        reused=True,
                    )
                    rows[task.task_key] = row
                    records[task.task_key] = self._record(
                        task,
                        paths,
                        config_hash,
                        started_at,
                        status="reused",
                    )
                except Exception as exc:
                    records[task.task_key] = self._record(
                        task,
                        paths,
                        config_hash,
                        started_at,
                        status="failed",
                        error=exc,
                    )
                    if self.run_policy.stop_on_error:
                        raise
            else:
                pending_by_model[task.model_name].append(task)

        for model_name, model_tasks in pending_by_model.items():
            try:
                source_matrix = self.matrix_loader(model_name)
                _validate_source_matrix(source_matrix)
            except Exception as exc:
                for task in model_tasks:
                    paths = self.output_layout.paths_for(task)
                    records[task.task_key] = self._record(
                        task,
                        paths,
                        config_hash,
                        self._timestamp(),
                        status="failed",
                        error=exc,
                    )
                if self.run_policy.stop_on_error:
                    raise
                continue

            try:
                for task in model_tasks:
                    started_at = self._timestamp()
                    paths = self.output_layout.paths_for(task)
                    try:
                        self._build_one(
                            source_matrix,
                            task,
                            paths,
                            config,
                            config_hash,
                        )
                        row = validate_randomized_pca_output(
                            paths,
                            expected_task=task,
                            expected_builder=self.builder.name,
                            expected_config_hash=config_hash,
                            reused=False,
                        )
                        rows[task.task_key] = row
                        records[task.task_key] = self._record(
                            task,
                            paths,
                            config_hash,
                            started_at,
                            status="computed",
                        )
                    except Exception as exc:
                        records[task.task_key] = self._record(
                            task,
                            paths,
                            config_hash,
                            started_at,
                            status="failed",
                            error=exc,
                        )
                        if self.run_policy.stop_on_error:
                            raise
                    finally:
                        gc.collect()
            finally:
                del source_matrix
                gc.collect()

        ordered_rows = tuple(
            rows[task.task_key] for task in tasks if task.task_key in rows
        )
        ordered_records = tuple(
            records[task.task_key] for task in tasks if task.task_key in records
        )
        manifest = RepresentationManifest(rows=ordered_rows)
        _write_manifest_atomic(manifest, manifest_path)
        return RepresentationPipelineRun(
            manifest=manifest,
            report=StageRunReport(stage=self.stage, records=ordered_records),
            manifest_path=manifest_path,
        )

    def _validate_tasks(self, tasks: tuple[ScaleTask, ...]) -> None:
        if not isinstance(tasks, tuple) or not tasks:
            raise ValueError("tasks must be a non-empty tuple")
        task_keys = [task.task_key for task in tasks]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("tasks must have unique identities")
        if len({task.analysis_id for task in tasks}) != 1:
            raise ValueError("one representation run must use a single analysis_id")
        for task in tasks:
            if task.seed is None:
                raise ValueError("all randomized PCA tasks require a seed")

    def _preflight_error_mode(
        self,
        tasks: tuple[ScaleTask, ...],
        manifest_path: Path,
    ) -> None:
        existing = [str(manifest_path)] if manifest_path.exists() else []
        for task in tasks:
            existing.extend(
                str(path)
                for path in self.output_layout.paths_for(task).all_files
                if path.exists()
            )
        if existing:
            raise FileExistsError(
                f"Randomized PCA outputs already exist: {sorted(existing)}"
            )

    def _build_one(
        self,
        matrix: np.ndarray,
        task: ScaleTask,
        final_paths: RandomizedPCAOutputPaths,
        config: Mapping[str, Any],
        config_hash: str,
    ) -> None:
        if self.run_policy.atomic_write:
            final_paths.task_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_root = Path(
                tempfile.mkdtemp(
                    dir=self.output_layout.output_root.parent,
                    prefix=f".{self.output_layout.output_root.name}.pca.",
                )
            )
            try:
                staging_layout = RandomizedPCAOutputLayout(staging_root)
                staging_paths = staging_layout.paths_for(task)
                self.builder.build(matrix, task, staging_root)
                self._finalize_metadata(
                    staging_paths,
                    final_paths,
                    task,
                    matrix.shape,
                    config,
                    config_hash,
                )
                validate_randomized_pca_output(
                    staging_paths,
                    expected_task=task,
                    expected_builder=self.builder.name,
                    expected_config_hash=config_hash,
                    reused=False,
                    canonical_paths=final_paths,
                )
                final_paths.task_dir.mkdir(parents=True, exist_ok=True)
                for source, destination in zip(
                    staging_paths.all_files,
                    final_paths.all_files,
                    strict=True,
                ):
                    os.replace(source, destination)
            finally:
                shutil.rmtree(staging_root, ignore_errors=True)
        else:
            self.builder.build(matrix, task, self.output_layout.output_root)
            self._finalize_metadata(
                final_paths,
                final_paths,
                task,
                matrix.shape,
                config,
                config_hash,
            )

    def _finalize_metadata(
        self,
        written_paths: RandomizedPCAOutputPaths,
        canonical_paths: RandomizedPCAOutputPaths,
        task: ScaleTask,
        input_shape: tuple[int, ...],
        config: Mapping[str, Any],
        config_hash: str,
    ) -> None:
        if not written_paths.metadata.is_file():
            raise FileNotFoundError(
                f"Randomized PCA runner did not write metadata: {written_paths.metadata}"
            )
        with written_paths.metadata.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(
            {
                "representation_schema_version": 1,
                "task": task.to_dict(),
                "task_key": task.task_key,
                "builder": self.builder.name,
                "builder_config": dict(self.builder.config()),
                "pipeline_config": dict(config),
                "config_hash": config_hash,
                "input_shape": [int(value) for value in input_shape],
                "output_shape": [int(input_shape[0]), task.scale],
                "output_paths": _canonical_path_metadata(canonical_paths),
            }
        )
        _write_json(written_paths.metadata, metadata)

    def _record(
        self,
        task: ScaleTask,
        paths: RandomizedPCAOutputPaths,
        config_hash: str,
        started_at: str,
        *,
        status: str,
        error: Exception | None = None,
    ) -> StageRunRecord:
        return StageRunRecord(
            task=task,
            stage=self.stage,
            status=status,
            output_paths=tuple(str(path) for path in paths.all_files),
            config_hash=config_hash,
            started_at=started_at,
            ended_at=self._timestamp(),
            error_message=(
                None if error is None else f"{type(error).__name__}: {error}"
            ),
            reused=status == "reused",
            computed=status == "computed",
        )

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return a datetime")
        return value.astimezone(timezone.utc).isoformat()


def _validate_source_matrix(matrix: np.ndarray) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix_loader must return a NumPy array")
    if matrix.ndim != 2:
        raise ValueError(f"source matrix must be 2D, got shape={matrix.shape}")
    if min(matrix.shape) < 1:
        raise ValueError("source matrix dimensions must be greater than zero")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("source matrix must use a numeric dtype")


def _canonical_path_metadata(paths: RandomizedPCAOutputPaths) -> dict[str, str]:
    return {
        "matrix": str(paths.matrix),
        "components": str(paths.components),
        "explained_ratio": str(paths.explained_ratio),
        "mean": str(paths.mean),
        "metadata": str(paths.metadata),
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2, allow_nan=False)


def _write_manifest_atomic(
    manifest: RepresentationManifest,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        manifest.to_frame().to_csv(temporary_path, index=False, encoding="utf-8")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _required_csv_integer(value: Any, field_name: str) -> int:
    if pd.isna(value) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    integer = int(value)
    if float(value) != integer:
        raise ValueError(f"{field_name} must be an integer")
    return integer


def _optional_csv_integer(value: Any, field_name: str) -> int | None:
    if pd.isna(value):
        return None
    return _required_csv_integer(value, field_name)


def _csv_boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field_name} must be a boolean")
