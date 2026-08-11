"""Composable representation and clustering pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
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


@dataclass(frozen=True)
class RepresentationResult:
    """Hold one scale-specific representation matrix and its provenance."""

    task: ScaleTask
    matrix: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray):
            raise TypeError("matrix must be a NumPy array")
        if self.matrix.ndim != 2:
            raise ValueError(f"matrix must be 2D, got shape={self.matrix.shape}")
        if self.matrix.shape[1] != self.task.scale:
            raise ValueError(
                f"matrix width {self.matrix.shape[1]} does not match task scale "
                f"{self.task.scale}"
            )


@runtime_checkable
class RepresentationProvider(Protocol):
    """Load a scale-specific matrix for one task."""

    def load(self, task: ScaleTask) -> RepresentationResult:
        """Load and return the representation for a task."""


class FullPCAPrefixProvider:
    """Serve prefix slices from one cached full PCA matrix per model."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        mmap_mode: str | None = "r",
        array_loader: Callable[..., np.ndarray] = np.load,
    ) -> None:
        self.source_root = Path(source_root)
        self.mmap_mode = mmap_mode
        self._array_loader = array_loader
        self._cache: dict[str, np.ndarray] = {}
        self._paths: dict[str, Path] = {}

    def matrix_path(self, task: ScaleTask) -> Path:
        """Return the canonical full PCA matrix path for a task."""

        return self.source_root / task.model_name / "full_pca" / "X_pca.npy"

    def _load_full_matrix(self, task: ScaleTask) -> tuple[np.ndarray, Path]:
        if task.model_name not in self._cache:
            path = self.matrix_path(task)
            if not path.is_file():
                raise FileNotFoundError(f"Full PCA matrix not found: {path}")
            matrix = self._array_loader(path, mmap_mode=self.mmap_mode)
            if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                shape = getattr(matrix, "shape", None)
                raise ValueError(f"Full PCA matrix must be 2D, got shape={shape}")
            self._cache[task.model_name] = matrix
            self._paths[task.model_name] = path
        return self._cache[task.model_name], self._paths[task.model_name]

    def load(self, task: ScaleTask) -> RepresentationResult:
        """Return the exact first-k column view of a model's full PCA matrix."""

        full_matrix, path = self._load_full_matrix(task)
        if task.scale > full_matrix.shape[1]:
            raise ValueError(
                f"task scale {task.scale} exceeds full PCA width {full_matrix.shape[1]} "
                f"for model {task.model_name!r}"
            )
        matrix = full_matrix[:, : task.scale]
        return RepresentationResult(
            task=task,
            matrix=matrix,
            metadata={
                "provider": type(self).__name__,
                "source_path": str(path),
                "full_shape": [int(value) for value in full_matrix.shape],
                "is_prefix_view": bool(np.shares_memory(matrix, full_matrix)),
            },
        )

    def clear_cache(self) -> None:
        """Release references to cached full PCA matrices."""

        self._cache.clear()
        self._paths.clear()


PerScalePathResolver = Callable[[Path, ScaleTask], Path]


def default_saved_per_scale_path(root: Path, task: ScaleTask) -> Path:
    """Resolve the repository's existing randomized-PCA matrix layout."""

    if task.seed is None:
        raise ValueError("the default saved-per-scale layout requires a seed")
    return (
        root
        / task.model_name
        / "random_pca"
        / f"seed_{task.seed:05d}"
        / f"pca_dim_{task.scale}"
        / "X_pca.npy"
    )


class SavedPerScaleProvider:
    """Load independently saved matrices through a configurable path resolver."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        path_resolver: PerScalePathResolver = default_saved_per_scale_path,
        mmap_mode: str | None = "r",
        array_loader: Callable[..., np.ndarray] = np.load,
    ) -> None:
        self.source_root = Path(source_root)
        self.path_resolver = path_resolver
        self.mmap_mode = mmap_mode
        self._array_loader = array_loader

    def matrix_path(self, task: ScaleTask) -> Path:
        """Resolve the saved matrix path for a task."""

        return Path(self.path_resolver(self.source_root, task))

    def load(self, task: ScaleTask) -> RepresentationResult:
        """Load and validate one already materialized scale matrix."""

        path = self.matrix_path(task)
        if not path.is_file():
            raise FileNotFoundError(f"Saved scale matrix not found: {path}")
        matrix = self._array_loader(path, mmap_mode=self.mmap_mode)
        return RepresentationResult(
            task=task,
            matrix=matrix,
            metadata={
                "provider": type(self).__name__,
                "source_path": str(path),
            },
        )


class InMemoryRepresentationProvider:
    """Serve task-keyed matrices without filesystem access."""

    def __init__(self, matrices: Mapping[str | ScaleTask, np.ndarray]) -> None:
        self._matrices: dict[str, np.ndarray] = {}
        for identity, matrix in matrices.items():
            task_key = identity.task_key if isinstance(identity, ScaleTask) else identity
            if not isinstance(task_key, str) or not task_key:
                raise ValueError("in-memory matrix keys must be ScaleTask objects or strings")
            if task_key in self._matrices:
                raise ValueError(f"duplicate in-memory task key: {task_key}")
            self._matrices[task_key] = matrix

    def load(self, task: ScaleTask) -> RepresentationResult:
        """Return the matrix registered for a task identity."""

        try:
            matrix = self._matrices[task.task_key]
        except KeyError as exc:
            raise KeyError(f"No in-memory representation for task {task.task_key}") from exc
        return RepresentationResult(
            task=task,
            matrix=matrix,
            metadata={"provider": type(self).__name__},
        )


@runtime_checkable
class MatrixTransform(Protocol):
    """Transform a representation matrix before downstream analysis."""

    name: str

    def transform(self, matrix: np.ndarray, task: ScaleTask) -> np.ndarray:
        """Apply the transform to one task matrix."""

    def config(self) -> Mapping[str, Any]:
        """Return the transform configuration used for hashing."""


class IdentityTransform:
    """Return a matrix unchanged."""

    name = "identity"

    def transform(self, matrix: np.ndarray, task: ScaleTask) -> np.ndarray:
        """Return the exact input array object after shape validation."""

        _validate_transform_input(matrix, task)
        return matrix

    def config(self) -> Mapping[str, Any]:
        return {"name": self.name}


class RowL2Normalizer:
    """Apply the repository's existing row-wise L2 normalization."""

    name = "row_l2_normalization"

    def __init__(self, eps: float = 1e-12) -> None:
        if not np.isscalar(eps) or not np.isfinite(eps) or float(eps) < 0.0:
            raise ValueError("eps must be a finite non-negative scalar")
        self.eps = float(eps)

    def transform(self, matrix: np.ndarray, task: ScaleTask) -> np.ndarray:
        """Normalize rows using the canonical PCA pipeline function."""

        _validate_transform_input(matrix, task)
        from .pca_hdbscan import apply_l2_normalization

        return apply_l2_normalization(matrix, l2_norm=True, eps=self.eps)

    def config(self) -> Mapping[str, Any]:
        return {"name": self.name, "eps": self.eps}


def _validate_transform_input(matrix: np.ndarray, task: ScaleTask) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={matrix.shape}")
    if matrix.shape[1] != task.scale:
        raise ValueError(
            f"matrix width {matrix.shape[1]} does not match task scale {task.scale}"
        )


@dataclass(frozen=True)
class PartitionResult:
    """Hold standardized token assignments from a partition method."""

    task: ScaleTask
    method: str
    labels: np.ndarray
    probabilities: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if not isinstance(self.labels, np.ndarray):
            raise TypeError("labels must be a NumPy array")
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape={self.labels.shape}")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise TypeError("labels must use an integer dtype")
        if self.probabilities is not None:
            if not isinstance(self.probabilities, np.ndarray):
                raise TypeError("probabilities must be a NumPy array or None")
            if self.probabilities.ndim != 1:
                raise ValueError(
                    f"probabilities must be 1D, got shape={self.probabilities.shape}"
                )
            if self.probabilities.shape[0] != self.labels.shape[0]:
                raise ValueError("labels and probabilities must have equal lengths")
            if not np.issubdtype(self.probabilities.dtype, np.number):
                raise TypeError("probabilities must use a numeric dtype")
            if not np.all(np.isfinite(self.probabilities)):
                raise ValueError("probabilities must contain only finite values")
            if np.any(self.probabilities < 0.0) or np.any(self.probabilities > 1.0):
                raise ValueError("probabilities must be within [0, 1]")

    @property
    def n_tokens(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.labels[self.labels >= 0]).size)

    @property
    def n_noise(self) -> int:
        return int(np.sum(self.labels < 0))

    @property
    def noise_fraction(self) -> float:
        if self.n_tokens == 0:
            return 0.0
        return self.n_noise / self.n_tokens

    def to_frame(self) -> pd.DataFrame:
        """Return the canonical token-level partition table."""

        probabilities = (
            self.probabilities.astype(np.float32, copy=False)
            if self.probabilities is not None
            else np.full(self.n_tokens, np.nan, dtype=np.float32)
        )
        return pd.DataFrame(
            {
                "token_id": np.arange(self.n_tokens, dtype=np.int64),
                "cluster_id": self.labels.astype(np.int64, copy=False),
                "probability": probabilities,
            }
        )


@runtime_checkable
class PartitionMethod(Protocol):
    """Produce a standardized partition from one representation matrix."""

    name: str

    def fit_partition(self, matrix: np.ndarray, task: ScaleTask) -> PartitionResult:
        """Fit the method and return token assignments."""

    def config(self) -> Mapping[str, Any]:
        """Return the method configuration used for hashing."""


HDBSCANRunner = Callable[..., tuple[np.ndarray, np.ndarray]]
TokenTableBuilder = Callable[[np.ndarray, np.ndarray], pd.DataFrame]


def _run_existing_gpu_hdbscan(
    matrix: np.ndarray,
    **parameters: Any,
) -> tuple[np.ndarray, np.ndarray]:
    from .pca_hdbscan import run_gpu_hdbscan

    return run_gpu_hdbscan(matrix, **parameters)


def _build_existing_token_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    from .pca_hdbscan import build_token_cluster_df

    return build_token_cluster_df(labels, probabilities)


class HDBSCANAdapter:
    """Adapt the repository's existing GPU HDBSCAN implementation."""

    name = "hdbscan"

    def __init__(
        self,
        *,
        min_cluster_size: int = 5,
        min_samples: int = 5,
        cluster_selection_method: str = "eom",
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        runner: HDBSCANRunner = _run_existing_gpu_hdbscan,
        token_table_builder: TokenTableBuilder = _build_existing_token_table,
    ) -> None:
        for field_name, value in (
            ("min_cluster_size", min_cluster_size),
            ("min_samples", min_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        for field_name, value in (
            ("cluster_selection_method", cluster_selection_method),
            ("metric", metric),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            not np.isscalar(cluster_selection_epsilon)
            or not np.isfinite(cluster_selection_epsilon)
            or float(cluster_selection_epsilon) < 0.0
        ):
            raise ValueError(
                "cluster_selection_epsilon must be a finite non-negative scalar"
            )

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.cluster_selection_epsilon = float(cluster_selection_epsilon)
        self._runner = runner
        self._token_table_builder = token_table_builder

    def config(self) -> Mapping[str, Any]:
        """Return parameters passed to the existing HDBSCAN runner."""

        return {
            "name": self.name,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_method": self.cluster_selection_method,
            "metric": self.metric,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
        }

    def fit_partition(self, matrix: np.ndarray, task: ScaleTask) -> PartitionResult:
        """Run existing GPU HDBSCAN and standardize its token assignments."""

        _validate_transform_input(matrix, task)
        parameters = dict(self.config())
        parameters.pop("name")
        labels, probabilities = self._runner(matrix, **parameters)
        token_table = self._token_table_builder(labels, probabilities)
        required_columns = {"token_id", "cluster_id", "probability"}
        if not required_columns.issubset(token_table.columns):
            missing = sorted(required_columns - set(token_table.columns))
            raise ValueError(f"token table is missing required columns: {missing}")
        expected_ids = np.arange(len(token_table), dtype=np.int64)
        if not np.array_equal(token_table["token_id"].to_numpy(), expected_ids):
            raise ValueError("token table token_id values must be contiguous and zero-based")

        return PartitionResult(
            task=task,
            method=self.name,
            labels=token_table["cluster_id"].to_numpy(dtype=np.int64, copy=True),
            probabilities=token_table["probability"].to_numpy(
                dtype=np.float32,
                copy=True,
            ),
            metadata={"parameters": self.config()},
        )


PARTITION_MANIFEST_COLUMNS = (
    "analysis_id",
    "model_name",
    "space_name",
    "scale",
    "source",
    "seed_type",
    "seed",
    "method",
    "partition_csv_path",
    "n_tokens",
    "n_clusters",
    "n_noise",
    "noise_fraction",
    "config_hash",
    "reused",
)


@dataclass(frozen=True)
class PartitionManifestRow:
    """Describe one saved token partition in a method-independent form."""

    task: ScaleTask
    method: str
    partition_csv_path: str
    n_tokens: int
    n_clusters: int
    n_noise: int
    noise_fraction: float
    config_hash: str
    reused: bool

    def __post_init__(self) -> None:
        for field_name in ("method", "partition_csv_path", "config_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("n_tokens", "n_clusters", "n_noise"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.n_noise > self.n_tokens:
            raise ValueError("n_noise cannot exceed n_tokens")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be a boolean")
        if not np.isscalar(self.noise_fraction) or not np.isfinite(self.noise_fraction):
            raise ValueError("noise_fraction must be finite")

        expected_fraction = self.n_noise / self.n_tokens if self.n_tokens else 0.0
        if not np.isclose(
            float(self.noise_fraction),
            expected_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "noise_fraction is inconsistent with n_noise and n_tokens"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return one flat manifest row."""

        return {
            **self.task.to_dict(),
            "method": self.method,
            "partition_csv_path": self.partition_csv_path,
            "n_tokens": self.n_tokens,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "noise_fraction": float(self.noise_fraction),
            "config_hash": self.config_hash,
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionManifestRow":
        """Construct one row from CSV-compatible scalar values."""

        provided = set(data)
        expected = set(PARTITION_MANIFEST_COLUMNS)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(
                f"Invalid partition manifest fields: missing={missing}, extra={extra}"
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
            method=str(data["method"]),
            partition_csv_path=str(data["partition_csv_path"]),
            n_tokens=_required_csv_integer(data["n_tokens"], "n_tokens"),
            n_clusters=_required_csv_integer(data["n_clusters"], "n_clusters"),
            n_noise=_required_csv_integer(data["n_noise"], "n_noise"),
            noise_fraction=float(data["noise_fraction"]),
            config_hash=str(data["config_hash"]),
            reused=_csv_boolean(data["reused"], "reused"),
        )


@dataclass(frozen=True)
class PartitionManifest:
    """Collect and serialize partition locations for downstream pipelines."""

    rows: tuple[PartitionManifestRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        identities = [(row.task.task_key, row.method) for row in self.rows]
        if len(identities) != len(set(identities)):
            raise ValueError("partition manifest contains duplicate task-method identities")

    def to_frame(self) -> pd.DataFrame:
        """Return the canonical long manifest table."""

        frame = pd.DataFrame(
            [row.to_dict() for row in self.rows],
            columns=PARTITION_MANIFEST_COLUMNS,
        )
        if not frame.empty:
            frame["seed"] = pd.array(frame["seed"], dtype="Int64")
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "PartitionManifest":
        """Validate and construct a manifest from a DataFrame."""

        if tuple(frame.columns) != PARTITION_MANIFEST_COLUMNS:
            raise ValueError(
                "partition manifest columns must exactly match the canonical order"
            )
        rows = tuple(
            PartitionManifestRow.from_dict(record)
            for record in frame.to_dict(orient="records")
        )
        return cls(rows=rows)

    def write_csv(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write a manifest without silently replacing an existing file."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Partition manifest already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(destination, index=False, encoding="utf-8")
        return destination

    @classmethod
    def read_csv(cls, path: str | Path) -> "PartitionManifest":
        """Read and validate a canonical manifest CSV."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Partition manifest not found: {source}")
        frame = pd.read_csv(source)
        return cls.from_frame(frame)


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


@dataclass(frozen=True)
class ClusteringOutputPaths:
    """Hold all files that make one partition output complete."""

    task_dir: Path
    partition_csv: Path
    metadata_json: Path


class ClusteringOutputLayout:
    """Resolve isolated clustering output paths without inspecting inputs."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def paths_for(self, task: ScaleTask, method: str) -> ClusteringOutputPaths:
        """Resolve paths for one task and partition method."""

        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        if Path(method).name != method:
            raise ValueError("method must be a single path-safe segment")

        task_dir = (
            self.output_root
            / task.analysis_id
            / "partitions"
            / task.model_name
            / task.space_name
            / task.source
        )
        if task.seed is not None:
            task_dir = task_dir / str(task.seed_type) / f"seed_{task.seed:05d}"
        task_dir = task_dir / method / f"scale_{task.scale}"
        return ClusteringOutputPaths(
            task_dir=task_dir,
            partition_csv=task_dir / "partition.csv",
            metadata_json=task_dir / "metadata.json",
        )

    def manifest_path(self, analysis_id: str, method: str) -> Path:
        """Return the aggregate partition manifest path for one stage run."""

        if not analysis_id.strip():
            raise ValueError("analysis_id must be a non-empty string")
        if Path(method).name != method:
            raise ValueError("method must be a single path-safe segment")
        return (
            self.output_root
            / analysis_id
            / "manifests"
            / f"{method}_partition_manifest.csv"
        )


def write_partition_output(
    result: PartitionResult,
    paths: ClusteringOutputPaths,
    *,
    config_hash: str,
    config: Mapping[str, Any],
    atomic_write: bool = True,
    replace: bool = False,
) -> PartitionManifestRow:
    """Safely persist one partition, using metadata as the completion marker."""

    for destination in (paths.partition_csv, paths.metadata_json):
        if destination.exists() and not replace:
            raise FileExistsError(f"Clustering output already exists: {destination}")
    if not isinstance(config_hash, str) or not config_hash.strip():
        raise ValueError("config_hash must be a non-empty string")

    paths.task_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "task": result.task.to_dict(),
        "task_key": result.task.task_key,
        "method": result.method,
        "config_hash": config_hash,
        "config": dict(config),
        "method_metadata": dict(result.metadata),
        "partition_csv_path": str(paths.partition_csv),
        "n_tokens": result.n_tokens,
        "n_clusters": result.n_clusters,
        "n_noise": result.n_noise,
        "noise_fraction": result.noise_fraction,
    }

    if atomic_write:
        partition_temp = _temporary_sibling(paths.partition_csv)
        metadata_temp = _temporary_sibling(paths.metadata_json)
        try:
            result.to_frame().to_csv(
                partition_temp,
                index=False,
                encoding="utf-8",
            )
            _write_json(metadata_temp, metadata)
            os.replace(partition_temp, paths.partition_csv)
            os.replace(metadata_temp, paths.metadata_json)
        finally:
            for temporary_path in (partition_temp, metadata_temp):
                if temporary_path.exists():
                    temporary_path.unlink()
    else:
        result.to_frame().to_csv(paths.partition_csv, index=False, encoding="utf-8")
        _write_json(paths.metadata_json, metadata)

    return PartitionManifestRow(
        task=result.task,
        method=result.method,
        partition_csv_path=str(paths.partition_csv),
        n_tokens=result.n_tokens,
        n_clusters=result.n_clusters,
        n_noise=result.n_noise,
        noise_fraction=result.noise_fraction,
        config_hash=config_hash,
        reused=False,
    )


def validate_partition_output(
    paths: ClusteringOutputPaths,
    *,
    expected_task: ScaleTask,
    expected_method: str,
    expected_config_hash: str,
) -> PartitionManifestRow:
    """Validate a complete saved partition before allowing resume."""

    missing = [
        str(path)
        for path in (paths.partition_csv, paths.metadata_json)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete clustering output; missing files: {missing}")

    with paths.metadata_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported clustering metadata schema version")
    if metadata.get("task") != expected_task.to_dict():
        raise ValueError("saved task identity does not match expected task")
    if metadata.get("task_key") != expected_task.task_key:
        raise ValueError("saved task key does not match expected task")
    if metadata.get("method") != expected_method:
        raise ValueError("saved method does not match expected method")
    if metadata.get("config_hash") != expected_config_hash:
        raise ValueError("saved config hash does not match expected configuration")
    if metadata.get("partition_csv_path") != str(paths.partition_csv):
        raise ValueError("saved partition path does not match output layout")

    result = load_partition_result(
        paths.partition_csv,
        task=expected_task,
        method=expected_method,
        metadata=metadata.get("method_metadata", {}),
    )
    expected_summary = {
        "n_tokens": result.n_tokens,
        "n_clusters": result.n_clusters,
        "n_noise": result.n_noise,
        "noise_fraction": result.noise_fraction,
    }
    for field_name, actual_value in expected_summary.items():
        saved_value = metadata.get(field_name)
        if field_name == "noise_fraction":
            matches = saved_value is not None and np.isclose(
                float(saved_value),
                actual_value,
                rtol=0.0,
                atol=1e-12,
            )
        else:
            matches = saved_value == actual_value
        if not matches:
            raise ValueError(f"saved {field_name} does not match partition table")

    return PartitionManifestRow(
        task=expected_task,
        method=expected_method,
        partition_csv_path=str(paths.partition_csv),
        n_tokens=result.n_tokens,
        n_clusters=result.n_clusters,
        n_noise=result.n_noise,
        noise_fraction=result.noise_fraction,
        config_hash=expected_config_hash,
        reused=True,
    )


def load_partition_result(
    path: str | Path,
    *,
    task: ScaleTask,
    method: str,
    metadata: Mapping[str, Any] | None = None,
) -> PartitionResult:
    """Load and validate the canonical token-level partition table."""

    frame = pd.read_csv(path)
    expected_columns = ["token_id", "cluster_id", "probability"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"partition columns must be exactly {expected_columns}, "
            f"got {frame.columns.tolist()}"
        )
    token_ids = frame["token_id"].to_numpy()
    if not np.array_equal(token_ids, np.arange(len(frame), dtype=np.int64)):
        raise ValueError("partition token_id values must be contiguous and zero-based")
    cluster_values = frame["cluster_id"].to_numpy()
    if not np.issubdtype(cluster_values.dtype, np.integer):
        raise TypeError("partition cluster_id values must use an integer dtype")
    probability_values = frame["probability"].to_numpy(dtype=np.float32)
    probabilities = None if np.isnan(probability_values).all() else probability_values
    if probabilities is not None and np.isnan(probabilities).any():
        raise ValueError("partition probabilities must be either all missing or all present")
    return PartitionResult(
        task=task,
        method=method,
        labels=cluster_values.astype(np.int64, copy=False),
        probabilities=probabilities,
        metadata={} if metadata is None else dict(metadata),
    )


def _temporary_sibling(destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2, allow_nan=False)


@dataclass(frozen=True)
class ClusteringPipelineRun:
    """Return the data products and execution report from a clustering run."""

    manifest: PartitionManifest
    report: StageRunReport
    manifest_path: Path


class ClusteringPipeline:
    """Schedule representation-to-partition tasks through narrow adapters."""

    stage = "clustering"

    def __init__(
        self,
        *,
        provider: RepresentationProvider,
        transforms: tuple[MatrixTransform, ...],
        method: PartitionMethod,
        output_layout: ClusteringOutputLayout,
        run_policy: RunPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(transforms, tuple):
            raise TypeError("transforms must be a tuple")
        if not isinstance(provider, RepresentationProvider):
            raise TypeError("provider must satisfy RepresentationProvider")
        if any(not isinstance(transform, MatrixTransform) for transform in transforms):
            raise TypeError("all transforms must satisfy MatrixTransform")
        if not isinstance(method, PartitionMethod):
            raise TypeError("method must satisfy PartitionMethod")

        self.provider = provider
        self.transforms = transforms
        self.method = method
        self.output_layout = output_layout
        self.run_policy = run_policy or RunPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def config(self) -> dict[str, Any]:
        """Return the complete task-independent computation configuration."""

        return {
            "schema_version": 1,
            "stage": self.stage,
            "provider": {"name": type(self.provider).__name__},
            "transforms": [dict(transform.config()) for transform in self.transforms],
            "method": dict(self.method.config()),
        }

    def run(self, tasks: tuple[ScaleTask, ...]) -> ClusteringPipelineRun:
        """Run or validate a homogeneous set of scale tasks."""

        if not isinstance(tasks, tuple) or not tasks:
            raise ValueError("tasks must be a non-empty tuple")
        task_keys = [task.task_key for task in tasks]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("tasks must have unique identities")
        analysis_ids = {task.analysis_id for task in tasks}
        if len(analysis_ids) != 1:
            raise ValueError("one clustering run must use a single analysis_id")

        analysis_id = next(iter(analysis_ids))
        manifest_path = self.output_layout.manifest_path(analysis_id, self.method.name)
        config = self.config()
        config_hash = stable_config_hash(config)
        if self.run_policy.existing_output == "error":
            self._preflight_error_mode(tasks, manifest_path)

        rows: list[PartitionManifestRow] = []
        records: list[StageRunRecord] = []
        for task in tasks:
            started_at = self._timestamp()
            paths = self.output_layout.paths_for(task, self.method.name)
            try:
                existing = paths.partition_csv.exists() or paths.metadata_json.exists()
                if self.run_policy.existing_output == "resume" and existing:
                    row = validate_partition_output(
                        paths,
                        expected_task=task,
                        expected_method=self.method.name,
                        expected_config_hash=config_hash,
                    )
                    status = "reused"
                else:
                    representation = self.provider.load(task)
                    matrix = representation.matrix
                    for transform in self.transforms:
                        matrix = transform.transform(matrix, task)
                    result = self.method.fit_partition(matrix, task)
                    self._validate_method_result(result, task, matrix.shape[0])
                    row = write_partition_output(
                        result,
                        paths,
                        config_hash=config_hash,
                        config=config,
                        atomic_write=self.run_policy.atomic_write,
                        replace=self.run_policy.existing_output == "replace",
                    )
                    status = "computed"
                rows.append(row)
                records.append(
                    StageRunRecord(
                        task=task,
                        stage=self.stage,
                        status=status,
                        output_paths=(str(paths.partition_csv), str(paths.metadata_json)),
                        config_hash=config_hash,
                        started_at=started_at,
                        ended_at=self._timestamp(),
                        reused=status == "reused",
                        computed=status == "computed",
                    )
                )
            except Exception as exc:
                records.append(
                    StageRunRecord(
                        task=task,
                        stage=self.stage,
                        status="failed",
                        output_paths=(str(paths.partition_csv), str(paths.metadata_json)),
                        config_hash=config_hash,
                        started_at=started_at,
                        ended_at=self._timestamp(),
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
                if self.run_policy.stop_on_error:
                    raise
            finally:
                gc.collect()

        manifest = PartitionManifest(rows=tuple(rows))
        _write_manifest_for_run(
            manifest,
            manifest_path,
            atomic_write=self.run_policy.atomic_write,
        )
        report = StageRunReport(stage=self.stage, records=tuple(records))
        return ClusteringPipelineRun(
            manifest=manifest,
            report=report,
            manifest_path=manifest_path,
        )

    def _preflight_error_mode(
        self,
        tasks: tuple[ScaleTask, ...],
        manifest_path: Path,
    ) -> None:
        existing_paths = []
        if manifest_path.exists():
            existing_paths.append(str(manifest_path))
        for task in tasks:
            paths = self.output_layout.paths_for(task, self.method.name)
            existing_paths.extend(
                str(path)
                for path in (paths.partition_csv, paths.metadata_json)
                if path.exists()
            )
        if existing_paths:
            raise FileExistsError(
                f"Clustering outputs already exist: {sorted(existing_paths)}"
            )

    def _validate_method_result(
        self,
        result: PartitionResult,
        task: ScaleTask,
        expected_tokens: int,
    ) -> None:
        if result.task != task:
            raise ValueError("partition method returned a result for a different task")
        if result.method != self.method.name:
            raise ValueError("partition method result name does not match adapter name")
        if result.n_tokens != expected_tokens:
            raise ValueError(
                "partition method result token count does not match representation rows"
            )

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return datetime objects")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


def _write_manifest_for_run(
    manifest: PartitionManifest,
    destination: Path,
    *,
    atomic_write: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if atomic_write:
        temporary_path = _temporary_sibling(destination)
        try:
            manifest.to_frame().to_csv(temporary_path, index=False, encoding="utf-8")
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    else:
        manifest.to_frame().to_csv(destination, index=False, encoding="utf-8")
