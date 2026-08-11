"""Manifest-to-manifest partition permutation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .clustering_pipeline import (
    ClusteringOutputLayout,
    ClusteringOutputPaths,
    PartitionManifest,
    PartitionManifestRow,
)
from .pipeline_core import (
    RunPolicy,
    ScaleTask,
    StageRunRecord,
    StageRunReport,
    stable_config_hash,
)


@dataclass(frozen=True)
class PermutationModelContext:
    """Describe the model resource needed by a permutation strategy."""

    model_name: str
    tokenizer_path: str

    def __post_init__(self) -> None:
        for field_name in ("model_name", "tokenizer_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "tokenizer_path": self.tokenizer_path,
        }


@runtime_checkable
class PartitionPermutationStrategy(Protocol):
    """Permute token identities while preserving one valid partition structure."""

    name: str

    def config(self) -> Mapping[str, Any]:
        """Return the task-independent permutation configuration."""

    def prepare(self, context: PermutationModelContext) -> Any:
        """Prepare reusable model-level state."""

    def permute(
        self,
        valid_partition: pd.DataFrame,
        *,
        random_state: int,
        model_name: str,
        prepared_state: Any,
    ) -> pd.DataFrame:
        """Return one token-permuted copy of a noise-free partition."""

    def validate(
        self,
        baseline_valid: pd.DataFrame,
        control: pd.DataFrame,
        *,
        model_name: str,
        prepared_state: Any,
    ) -> None:
        """Validate strategy-specific invariants."""

    def finalize(self, prepared_state: Any) -> None:
        """Release prepared model-level state."""


class GlobalTokenPermutation:
    """Shuffle token IDs globally across all non-noise cluster positions."""

    name = "token_permutation"

    def config(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "noise_handling": "remove_before_permutation",
            "permutation_scope": "all_non_noise_positions",
            "random_generator": "numpy.default_rng",
        }

    def prepare(self, context: PermutationModelContext) -> None:
        return None

    def permute(
        self,
        valid_partition: pd.DataFrame,
        *,
        random_state: int,
        model_name: str,
        prepared_state: Any,
    ) -> pd.DataFrame:
        from .partition_permutation_control import (
            build_permuted_cluster_df,
            permute_token_ids,
        )

        permuted_ids = permute_token_ids(
            valid_partition,
            random_state=random_state,
        )
        return build_permuted_cluster_df(valid_partition, permuted_ids)

    def validate(
        self,
        baseline_valid: pd.DataFrame,
        control: pd.DataFrame,
        *,
        model_name: str,
        prepared_state: Any,
    ) -> None:
        return None

    def finalize(self, prepared_state: Any) -> None:
        return None


TokenizerLoader = Callable[[str], Any]


class LengthBucketTokenPermutation:
    """Shuffle token IDs only within fixed original character-length buckets."""

    name = "length_bucket_permutation"

    def __init__(self, *, tokenizer_loader: TokenizerLoader | None = None) -> None:
        if tokenizer_loader is not None and not callable(tokenizer_loader):
            raise TypeError("tokenizer_loader must be callable or None")
        self._tokenizer_loader = tokenizer_loader or self._load_tokenizer

    def config(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "noise_handling": "remove_before_permutation",
            "permutation_scope": "original_character_length_bucket",
            "length_buckets": ["1-2", "3-5", "6-10", "11+"],
            "random_generator": "numpy.default_rng",
        }

    def prepare(self, context: PermutationModelContext) -> dict[str, Any]:
        return {
            "tokenizer": self._tokenizer_loader(context.tokenizer_path),
            "token_text_cache": {},
        }

    def permute(
        self,
        valid_partition: pd.DataFrame,
        *,
        random_state: int,
        model_name: str,
        prepared_state: Any,
    ) -> pd.DataFrame:
        self._validate_state(prepared_state)
        from .permutation_multiseed_pipeline import (
            permute_token_ids_within_length_buckets,
        )

        return permute_token_ids_within_length_buckets(
            valid_partition,
            tokenizer=prepared_state["tokenizer"],
            model_name=model_name,
            random_state=random_state,
            token_text_cache=prepared_state["token_text_cache"],
        )

    def validate(
        self,
        baseline_valid: pd.DataFrame,
        control: pd.DataFrame,
        *,
        model_name: str,
        prepared_state: Any,
    ) -> None:
        self._validate_state(prepared_state)
        from .permutation_multiseed_pipeline import attach_length_bucket_column

        baseline_buckets = attach_length_bucket_column(
            baseline_valid,
            tokenizer=prepared_state["tokenizer"],
            model_name=model_name,
            token_text_cache=prepared_state["token_text_cache"],
        )["length_bucket"].to_numpy()
        control_buckets = attach_length_bucket_column(
            control,
            tokenizer=prepared_state["tokenizer"],
            model_name=model_name,
            token_text_cache=prepared_state["token_text_cache"],
        )["length_bucket"].to_numpy()
        if not np.array_equal(baseline_buckets, control_buckets):
            raise ValueError(
                "length-bucket permutation moved a token across bucket boundaries"
            )

    def finalize(self, prepared_state: Any) -> None:
        if isinstance(prepared_state, dict):
            prepared_state.clear()

    @staticmethod
    def _validate_state(prepared_state: Any) -> None:
        if not isinstance(prepared_state, Mapping):
            raise TypeError("prepared_state must be a mapping")
        if "tokenizer" not in prepared_state or "token_text_cache" not in prepared_state:
            raise ValueError("prepared_state is missing tokenizer resources")

    @staticmethod
    def _load_tokenizer(tokenizer_path: str) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)


PermutationContextFactory = Callable[
    [str, tuple[PartitionManifestRow, ...]],
    PermutationModelContext,
]


@dataclass(frozen=True)
class PartitionPermutationPipelineRun:
    """Return the control partition manifest and stage report."""

    manifest: PartitionManifest
    report: StageRunReport
    manifest_path: Path


class PartitionPermutationPipeline:
    """Convert one baseline manifest into seeded control partition manifests."""

    stage = "partition_permutation"

    def __init__(
        self,
        *,
        strategy: PartitionPermutationStrategy,
        analysis_id: str,
        seeds: tuple[int, ...],
        output_layout: ClusteringOutputLayout,
        source: str | None = None,
        context_factory: PermutationContextFactory | None = None,
        run_policy: RunPolicy | None = None,
        partition_loader: Callable[[str | Path], pd.DataFrame] = pd.read_csv,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(strategy, PartitionPermutationStrategy):
            raise TypeError("strategy must satisfy PartitionPermutationStrategy")
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            raise ValueError("analysis_id must be a non-empty string")
        if not isinstance(seeds, tuple) or not seeds:
            raise ValueError("seeds must be a non-empty tuple")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be unique")
        resolved_source = strategy.name if source is None else source
        if not isinstance(resolved_source, str) or not resolved_source.strip():
            raise ValueError("source must be a non-empty string")
        if not callable(partition_loader):
            raise TypeError("partition_loader must be callable")

        self.strategy = strategy
        self.analysis_id = analysis_id
        self.seeds = seeds
        self.output_layout = output_layout
        self.source = resolved_source
        self.context_factory = context_factory or self._default_context_factory
        self.run_policy = run_policy or RunPolicy()
        self._partition_loader = partition_loader
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def config(self) -> dict[str, Any]:
        """Return the task-independent permutation configuration."""

        return {
            "schema_version": 1,
            "stage": self.stage,
            "analysis_id": self.analysis_id,
            "source": self.source,
            "seed_type": "permutation_seed",
            "seeds": list(self.seeds),
            "strategy": dict(self.strategy.config()),
        }

    def run(
        self,
        baseline_manifest: PartitionManifest | str | Path,
    ) -> PartitionPermutationPipelineRun:
        """Create or validate every baseline-partition and seed control task."""

        baseline = self._coerce_manifest(baseline_manifest)
        if not baseline.rows:
            raise ValueError("baseline_manifest must contain at least one partition")
        tasks = tuple(
            self._control_task(row, seed)
            for row in baseline.rows
            for seed in self.seeds
        )
        if len({task.task_key for task in tasks}) != len(tasks):
            raise ValueError("control tasks must have unique identities")
        manifest_path = self.output_layout.manifest_path(
            self.analysis_id,
            self.strategy.name,
        )
        if self.run_policy.existing_output == "error":
            self._preflight_error_mode(tasks, manifest_path)

        base_config = self.config()
        rows: list[PartitionManifestRow] = []
        records: list[StageRunRecord] = []
        model_rows = self._group_rows_by_model(baseline.rows)

        for model_name, baseline_rows in model_rows.items():
            context = self.context_factory(model_name, baseline_rows)
            if not isinstance(context, PermutationModelContext):
                raise TypeError(
                    "context_factory must return a PermutationModelContext"
                )
            if context.model_name != model_name:
                raise ValueError("permutation context model does not match manifest model")
            prepared_state = self.strategy.prepare(context)
            try:
                for baseline_row in baseline_rows:
                    baseline_valid = self._load_valid_baseline(baseline_row)
                    for seed in self.seeds:
                        task = self._control_task(baseline_row, seed)
                        paths = self.output_layout.paths_for(task, self.strategy.name)
                        task_config = {
                            **base_config,
                            "model_context": context.to_dict(),
                            "baseline_partition": {
                                "task": baseline_row.task.to_dict(),
                                "method": baseline_row.method,
                                "path": baseline_row.partition_csv_path,
                                "config_hash": baseline_row.config_hash,
                            },
                            "task_seed": seed,
                        }
                        config_hash = stable_config_hash(task_config)
                        started_at = self._timestamp()
                        try:
                            existing = (
                                paths.partition_csv.exists()
                                or paths.metadata_json.exists()
                            )
                            if (
                                self.run_policy.existing_output == "resume"
                                and existing
                            ):
                                row = self._validate_saved_output(
                                    paths,
                                    task=task,
                                    baseline_row=baseline_row,
                                    baseline_valid=baseline_valid,
                                    config_hash=config_hash,
                                    prepared_state=prepared_state,
                                )
                                status = "reused"
                            else:
                                control = self.strategy.permute(
                                    baseline_valid.copy(),
                                    random_state=seed,
                                    model_name=model_name,
                                    prepared_state=prepared_state,
                                )
                                self._validate_control(
                                    baseline_valid,
                                    control,
                                    model_name=model_name,
                                    prepared_state=prepared_state,
                                )
                                row = self._write_output(
                                    control,
                                    paths,
                                    task=task,
                                    baseline_row=baseline_row,
                                    config=task_config,
                                    config_hash=config_hash,
                                )
                                status = "computed"
                            rows.append(row)
                            records.append(
                                StageRunRecord(
                                    task=task,
                                    stage=self.stage,
                                    status=status,
                                    output_paths=(
                                        str(paths.partition_csv),
                                        str(paths.metadata_json),
                                    ),
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
                                    output_paths=(
                                        str(paths.partition_csv),
                                        str(paths.metadata_json),
                                    ),
                                    config_hash=config_hash,
                                    started_at=started_at,
                                    ended_at=self._timestamp(),
                                    error_message=f"{type(exc).__name__}: {exc}",
                                )
                            )
                            if self.run_policy.stop_on_error:
                                raise
            finally:
                self.strategy.finalize(prepared_state)

        manifest = PartitionManifest(rows=tuple(rows))
        _write_partition_manifest_atomic(manifest, manifest_path)
        return PartitionPermutationPipelineRun(
            manifest=manifest,
            report=StageRunReport(stage=self.stage, records=tuple(records)),
            manifest_path=manifest_path,
        )

    def _control_task(self, row: PartitionManifestRow, seed: int) -> ScaleTask:
        return ScaleTask(
            analysis_id=self.analysis_id,
            model_name=row.task.model_name,
            space_name=row.task.space_name,
            scale=row.task.scale,
            source=self.source,
            seed_type="permutation_seed",
            seed=seed,
        )

    def _load_valid_baseline(self, row: PartitionManifestRow) -> pd.DataFrame:
        frame = self._partition_loader(row.partition_csv_path)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("partition_loader must return a pandas DataFrame")
        expected_columns = ["token_id", "cluster_id", "probability"]
        if frame.columns.tolist() != expected_columns:
            raise ValueError(
                f"baseline partition columns must be exactly {expected_columns}"
            )
        if len(frame) != row.n_tokens:
            raise ValueError("baseline partition row count does not match its manifest")
        from .partition_permutation_control import filter_valid_cluster_df

        valid = filter_valid_cluster_df(frame)
        valid = valid.loc[:, expected_columns].reset_index(drop=True)
        if valid["token_id"].duplicated().any():
            raise ValueError("baseline valid token IDs must be unique")
        return valid

    def _validate_control(
        self,
        baseline_valid: pd.DataFrame,
        control: pd.DataFrame,
        *,
        model_name: str,
        prepared_state: Any,
    ) -> None:
        if not isinstance(control, pd.DataFrame):
            raise TypeError("permutation strategy must return a pandas DataFrame")
        expected_columns = ["token_id", "cluster_id", "probability"]
        if control.columns.tolist() != expected_columns:
            raise ValueError(
                f"control partition columns must be exactly {expected_columns}"
            )
        if len(control) != len(baseline_valid):
            raise ValueError("control partition changed the valid token count")
        if control["token_id"].duplicated().any():
            raise ValueError("control token IDs must be unique")
        if sorted(control["token_id"].astype(int).tolist()) != sorted(
            baseline_valid["token_id"].astype(int).tolist()
        ):
            raise ValueError("control partition changed the valid token-ID multiset")
        if not np.array_equal(
            control["cluster_id"].to_numpy(dtype=np.int64),
            baseline_valid["cluster_id"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("control partition changed cluster assignments")
        if not control["probability"].reset_index(drop=True).equals(
            baseline_valid["probability"].reset_index(drop=True)
        ):
            raise ValueError("control partition changed membership probabilities")
        if np.any(control["cluster_id"].to_numpy(dtype=np.int64) < 0):
            raise ValueError("control partition must not contain noise rows")
        self.strategy.validate(
            baseline_valid,
            control,
            model_name=model_name,
            prepared_state=prepared_state,
        )

    def _write_output(
        self,
        control: pd.DataFrame,
        paths: ClusteringOutputPaths,
        *,
        task: ScaleTask,
        baseline_row: PartitionManifestRow,
        config: Mapping[str, Any],
        config_hash: str,
    ) -> PartitionManifestRow:
        for destination in (paths.partition_csv, paths.metadata_json):
            if destination.exists() and self.run_policy.existing_output != "replace":
                raise FileExistsError(
                    f"Permutation output already exists: {destination}"
                )
        paths.task_dir.mkdir(parents=True, exist_ok=True)
        n_tokens = int(len(control))
        n_clusters = int(control["cluster_id"].nunique())
        metadata = {
            "schema_version": 1,
            "task": task.to_dict(),
            "task_key": task.task_key,
            "method": self.strategy.name,
            "config_hash": config_hash,
            "config": dict(config),
            "baseline_partition": {
                "task_key": baseline_row.task.task_key,
                "path": baseline_row.partition_csv_path,
                "config_hash": baseline_row.config_hash,
            },
            "partition_csv_path": str(paths.partition_csv),
            "n_tokens": n_tokens,
            "n_clusters": n_clusters,
            "n_noise": 0,
            "noise_fraction": 0.0,
        }
        if self.run_policy.atomic_write:
            partition_temp = _temporary_sibling(paths.partition_csv)
            metadata_temp = _temporary_sibling(paths.metadata_json)
            try:
                control.to_csv(partition_temp, index=False, encoding="utf-8")
                _write_json(metadata_temp, metadata)
                os.replace(partition_temp, paths.partition_csv)
                os.replace(metadata_temp, paths.metadata_json)
            finally:
                for temporary in (partition_temp, metadata_temp):
                    if temporary.exists():
                        temporary.unlink()
        else:
            control.to_csv(paths.partition_csv, index=False, encoding="utf-8")
            _write_json(paths.metadata_json, metadata)
        return PartitionManifestRow(
            task=task,
            method=self.strategy.name,
            partition_csv_path=str(paths.partition_csv),
            n_tokens=n_tokens,
            n_clusters=n_clusters,
            n_noise=0,
            noise_fraction=0.0,
            config_hash=config_hash,
            reused=False,
        )

    def _validate_saved_output(
        self,
        paths: ClusteringOutputPaths,
        *,
        task: ScaleTask,
        baseline_row: PartitionManifestRow,
        baseline_valid: pd.DataFrame,
        config_hash: str,
        prepared_state: Any,
    ) -> PartitionManifestRow:
        missing = [
            str(path)
            for path in (paths.partition_csv, paths.metadata_json)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete permutation output; missing files: {missing}"
            )
        metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported permutation metadata schema version")
        if metadata.get("task") != task.to_dict():
            raise ValueError("saved permutation task does not match expected task")
        if metadata.get("task_key") != task.task_key:
            raise ValueError("saved permutation task key does not match")
        if metadata.get("method") != self.strategy.name:
            raise ValueError("saved permutation method does not match strategy")
        if metadata.get("config_hash") != config_hash:
            raise ValueError("saved permutation config hash does not match")
        if metadata.get("partition_csv_path") != str(paths.partition_csv):
            raise ValueError("saved permutation path does not match output layout")
        baseline_metadata = metadata.get("baseline_partition", {})
        if baseline_metadata != {
            "task_key": baseline_row.task.task_key,
            "path": baseline_row.partition_csv_path,
            "config_hash": baseline_row.config_hash,
        }:
            raise ValueError("saved baseline partition provenance does not match")
        control = pd.read_csv(paths.partition_csv)
        self._validate_control(
            baseline_valid,
            control,
            model_name=task.model_name,
            prepared_state=prepared_state,
        )
        expected_summary = {
            "n_tokens": int(len(control)),
            "n_clusters": int(control["cluster_id"].nunique()),
            "n_noise": 0,
            "noise_fraction": 0.0,
        }
        if any(metadata.get(key) != value for key, value in expected_summary.items()):
            raise ValueError("saved permutation summary does not match partition")
        return PartitionManifestRow(
            task=task,
            method=self.strategy.name,
            partition_csv_path=str(paths.partition_csv),
            n_tokens=expected_summary["n_tokens"],
            n_clusters=expected_summary["n_clusters"],
            n_noise=0,
            noise_fraction=0.0,
            config_hash=config_hash,
            reused=True,
        )

    def _preflight_error_mode(
        self,
        tasks: tuple[ScaleTask, ...],
        manifest_path: Path,
    ) -> None:
        existing = [str(manifest_path)] if manifest_path.exists() else []
        for task in tasks:
            paths = self.output_layout.paths_for(task, self.strategy.name)
            existing.extend(
                str(path)
                for path in (paths.partition_csv, paths.metadata_json)
                if path.exists()
            )
        if existing:
            raise FileExistsError(
                f"Permutation outputs already exist: {sorted(existing)}"
            )

    @staticmethod
    def _coerce_manifest(
        manifest: PartitionManifest | str | Path,
    ) -> PartitionManifest:
        if isinstance(manifest, PartitionManifest):
            return manifest
        if isinstance(manifest, (str, Path)):
            return PartitionManifest.read_csv(manifest)
        raise TypeError("baseline_manifest must be a PartitionManifest or CSV path")

    @staticmethod
    def _group_rows_by_model(
        rows: tuple[PartitionManifestRow, ...],
    ) -> dict[str, tuple[PartitionManifestRow, ...]]:
        grouped: dict[str, list[PartitionManifestRow]] = {}
        for row in rows:
            grouped.setdefault(row.task.model_name, []).append(row)
        return {model: tuple(items) for model, items in grouped.items()}

    @staticmethod
    def _default_context_factory(
        model_name: str,
        rows: tuple[PartitionManifestRow, ...],
    ) -> PermutationModelContext:
        return PermutationModelContext(
            model_name=model_name,
            tokenizer_path=str(Path(model_name) / "tokenizer"),
        )

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return a datetime")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()


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


def _write_partition_manifest_atomic(
    manifest: PartitionManifest,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        manifest.to_frame().to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
