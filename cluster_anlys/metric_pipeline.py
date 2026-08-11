"""Composable partition metric contracts and pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .clustering_pipeline import PartitionManifest, PartitionManifestRow
from .pipeline_core import (
    RunPolicy,
    ScaleTask,
    StageRunRecord,
    StageRunReport,
    stable_config_hash,
)


METRIC_RESULT_COLUMNS = (
    "analysis_id",
    "model_name",
    "space_name",
    "scale",
    "source",
    "seed_type",
    "seed",
    "partition_method",
    "metric",
    "component",
    "aggregation",
    "value",
    "scope",
    "config_hash",
    "reused",
)


@dataclass(frozen=True)
class ModelMetricContext:
    """Describe model-level resources available during metric preparation."""

    model_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) or not key for key in self.metadata):
            raise ValueError("metadata keys must be non-empty strings")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MetricObservation:
    """Describe one scalar emitted by a partition metric evaluation."""

    component: str
    aggregation: str
    value: float
    scope: str = "partition"

    def __post_init__(self) -> None:
        for field_name in ("component", "aggregation", "scope"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not np.isscalar(self.value) or not np.isfinite(self.value):
            raise ValueError("value must be a finite scalar")

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the identity within one metric evaluation."""

        return self.component, self.aggregation, self.scope


@dataclass(frozen=True)
class MetricEvaluation:
    """Hold scalar observations and optional metric-specific detail tables."""

    observations: tuple[MetricObservation, ...]
    detail_tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("observations must be a non-empty tuple")
        if any(not isinstance(item, MetricObservation) for item in self.observations):
            raise TypeError("observations must contain MetricObservation objects")
        identities = [item.identity for item in self.observations]
        if len(identities) != len(set(identities)):
            raise ValueError("metric evaluation contains duplicate observation identities")
        if not isinstance(self.detail_tables, Mapping):
            raise TypeError("detail_tables must be a mapping")
        for name, table in self.detail_tables.items():
            if not isinstance(name, str) or not name:
                raise ValueError("detail table names must be non-empty strings")
            if not isinstance(table, pd.DataFrame):
                raise TypeError("detail table values must be pandas DataFrames")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) or not key for key in self.metadata):
            raise ValueError("metadata keys must be non-empty strings")
        object.__setattr__(self, "detail_tables", dict(self.detail_tables))
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class PartitionMetric(Protocol):
    """Evaluate one metric over token-level partition tables."""

    name: str

    def config(self) -> Mapping[str, Any]:
        """Return the metric configuration used for provenance hashing."""

    def prepare(self, model_context: ModelMetricContext) -> Any:
        """Build reusable state once for one model."""

    def evaluate(
        self,
        partition: pd.DataFrame,
        prepared_state: Any,
    ) -> MetricEvaluation:
        """Evaluate one partition using model-level prepared state."""

    def finalize(self, prepared_state: Any) -> None:
        """Release resources associated with prepared model state."""


class OrthographicSimilarityMetric:
    """Adapt the existing Kondrak cluster score to PartitionMetric."""

    name = "orthographic_similarity"

    def __init__(
        self,
        *,
        token_col: str = "token_str",
        token_id_col: str = "token_id",
        cluster_id_col: str = "cluster_id",
        affix_prefix: str = "##",
        ddof: int = 0,
        tokenizer_loader: Callable[[str], Any] | None = None,
    ) -> None:
        for field_name, value in (
            ("token_col", token_col),
            ("token_id_col", token_id_col),
            ("cluster_id_col", cluster_id_col),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(affix_prefix, str):
            raise TypeError("affix_prefix must be a string")
        if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer")
        self.token_col = token_col
        self.token_id_col = token_id_col
        self.cluster_id_col = cluster_id_col
        self.affix_prefix = affix_prefix
        self.ddof = ddof
        self._tokenizer_loader = tokenizer_loader or self._load_tokenizer

    def config(self) -> Mapping[str, Any]:
        """Return the effective existing O-S computation configuration."""

        return {
            "token_col": self.token_col,
            "token_id_col": self.token_id_col,
            "cluster_id_col": self.cluster_id_col,
            "affix_prefix": self.affix_prefix,
            "ddof": self.ddof,
            "noise_label": -1,
            "cluster_weighting": "equal",
            "large_cluster_approximation": {
                "exact_token_limit": 512,
                "sample_size": 128,
                "n_repeats": 10,
                "random_state": 1813382118,
            },
        }

    def prepare(self, model_context: ModelMetricContext) -> dict[str, Any]:
        """Load one tokenizer for a model unless one is supplied in metadata."""

        tokenizer = model_context.metadata.get("tokenizer")
        if tokenizer is None:
            tokenizer_path = str(
                model_context.metadata.get(
                    "tokenizer_path",
                    f"{model_context.model_name}/tokenizer",
                )
            )
            tokenizer = self._tokenizer_loader(tokenizer_path)
        return {
            "model_name": model_context.model_name,
            "tokenizer": tokenizer,
        }

    def evaluate(
        self,
        partition: pd.DataFrame,
        prepared_state: Any,
    ) -> MetricEvaluation:
        """Compute legacy O-S cluster details and equal-weight summaries."""

        if not isinstance(prepared_state, Mapping):
            raise TypeError("prepared_state must be a mapping")
        model_name = prepared_state.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("prepared_state must contain a model_name")

        from .kondrak_cluster_morphology import (
            attach_token_str_from_tokenizer,
            compute_cluster_level_morphology_df,
            compute_global_cluster_stats,
            filter_valid_clusters,
        )

        valid_partition = filter_valid_clusters(
            partition,
            cluster_id_col=self.cluster_id_col,
        )
        if self.token_col not in valid_partition.columns:
            tokenizer = prepared_state.get("tokenizer")
            if tokenizer is None:
                raise ValueError("prepared_state must contain a tokenizer")
            valid_partition = attach_token_str_from_tokenizer(
                valid_partition,
                tokenizer=tokenizer,
                model_name=model_name,
                token_id_col=self.token_id_col,
                token_str_col=self.token_col,
            )
        cluster_scores = compute_cluster_level_morphology_df(
            valid_partition,
            cluster_id_col=self.cluster_id_col,
            token_col=self.token_col,
            affix_prefix=self.affix_prefix,
            sort_by_cluster_id=True,
        )
        stats = compute_global_cluster_stats(
            cluster_scores,
            score_col="M(c)",
            ddof=self.ddof,
        )
        return MetricEvaluation(
            observations=(
                MetricObservation("score", "mean", stats["global_mean"]),
                MetricObservation("score", "std", stats["global_std"]),
                MetricObservation(
                    "cluster_count",
                    "count",
                    float(stats["n_clusters"]),
                ),
            ),
            detail_tables={"clusters": cluster_scores},
            metadata={
                "n_clusters": int(stats["n_clusters"]),
                "cluster_weighting": "equal",
            },
        )

    def finalize(self, prepared_state: Any) -> None:
        """Complete the no-op lifecycle for tokenizer-backed state."""

    @staticmethod
    def _load_tokenizer(tokenizer_path: str) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)


class MultiScriptEntropyMetric:
    """Adapt the existing multi-script entropy chain to PartitionMetric."""

    name = "multi_script_entropy"

    def __init__(
        self,
        *,
        token_col: str = "token_str",
        token_id_col: str = "token_id",
        cluster_id_col: str = "cluster_id",
        script_col: str = "script",
        tokenizer_loader: Callable[[str], Any] | None = None,
    ) -> None:
        for field_name, value in (
            ("token_col", token_col),
            ("token_id_col", token_id_col),
            ("cluster_id_col", cluster_id_col),
            ("script_col", script_col),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        self.token_col = token_col
        self.token_id_col = token_id_col
        self.cluster_id_col = cluster_id_col
        self.script_col = script_col
        self._tokenizer_loader = tokenizer_loader or self._load_tokenizer

    def config(self) -> Mapping[str, Any]:
        """Return the effective existing MS-E configuration."""

        return {
            "token_col": self.token_col,
            "token_id_col": self.token_id_col,
            "cluster_id_col": self.cluster_id_col,
            "script_col": self.script_col,
            "noise_label": -1,
            "cluster_weighting": "equal",
            "entropy_log_base": "natural",
            "normalization": "log_global_script_set_size",
            "ddof": 0,
        }

    def prepare(self, model_context: ModelMetricContext) -> dict[str, Any]:
        """Load a tokenizer and build the legacy global script cache once."""

        from .script_entropy import build_global_script_cache_from_tokenizer

        tokenizer = model_context.metadata.get("tokenizer")
        if tokenizer is None:
            tokenizer_path = str(
                model_context.metadata.get(
                    "tokenizer_path",
                    f"{model_context.model_name}/tokenizer",
                )
            )
            tokenizer = self._tokenizer_loader(tokenizer_path)
        full_script_cache = build_global_script_cache_from_tokenizer(
            tokenizer,
            model_name=model_context.model_name,
        )
        script_cache = {
            "global_script_set": list(full_script_cache["global_script_set"]),
            "global_script_set_size": int(
                full_script_cache["global_script_set_size"]
            ),
        }
        return {
            "model_name": model_context.model_name,
            "tokenizer": tokenizer,
            "script_cache": script_cache,
        }

    def evaluate(
        self,
        partition: pd.DataFrame,
        prepared_state: Any,
    ) -> MetricEvaluation:
        """Compute legacy script assignments, cluster entropies, and summaries."""

        if not isinstance(prepared_state, Mapping):
            raise TypeError("prepared_state must be a mapping")
        model_name = prepared_state.get("model_name")
        tokenizer = prepared_state.get("tokenizer")
        script_cache = prepared_state.get("script_cache")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("prepared_state must contain a model_name")
        if not isinstance(script_cache, Mapping):
            raise ValueError("prepared_state must contain a script_cache")
        global_script_set = script_cache.get("global_script_set")
        if not isinstance(global_script_set, list) or not global_script_set:
            raise ValueError("script_cache must contain a non-empty global_script_set")

        from .script_entropy import (
            annotate_token_scripts,
            build_cluster_script_count_df,
            compute_cluster_entropy_df,
            filter_noise_clusters,
            partition_entropy_stats,
            standardize_cluster_input_df,
        )

        token_id_partition = partition[
            [self.token_id_col, self.cluster_id_col]
        ].copy()
        standardized = standardize_cluster_input_df(
            token_id_partition,
            model_name=model_name,
            tokenizer=tokenizer,
            cluster_id_col=self.cluster_id_col,
            token_str_col=self.token_col,
            token_id_col=self.token_id_col,
        )
        token_scripts = annotate_token_scripts(
            standardized,
            token_col=self.token_col,
            script_col=self.script_col,
        )
        non_noise_tokens = filter_noise_clusters(
            token_scripts,
            cluster_id_col=self.cluster_id_col,
            noise_cluster_id=-1,
        )
        cluster_counts = build_cluster_script_count_df(
            non_noise_tokens,
            global_script_set=global_script_set,
            cluster_id_col=self.cluster_id_col,
            script_col=self.script_col,
        )
        cluster_entropies = compute_cluster_entropy_df(
            cluster_counts,
            global_script_set=global_script_set,
            cluster_id_col=self.cluster_id_col,
            warn_on_degenerate_global_set=True,
        )
        stats = partition_entropy_stats(
            cluster_entropies,
            raw_entropy_col="H",
            norm_entropy_col="H_norm",
        )
        return MetricEvaluation(
            observations=(
                MetricObservation("raw", "mean", stats["mean_H"]),
                MetricObservation("raw", "std", stats["std_H"]),
                MetricObservation("normalized", "mean", stats["mean_H_norm"]),
                MetricObservation("normalized", "std", stats["std_H_norm"]),
                MetricObservation(
                    "cluster_count",
                    "count",
                    float(stats["num_clusters"]),
                ),
            ),
            detail_tables={"clusters": cluster_entropies},
            metadata={
                "n_clusters": int(stats["num_clusters"]),
                "global_script_set": list(global_script_set),
                "global_script_set_size": int(len(global_script_set)),
                "cluster_weighting": "equal",
            },
        )

    def finalize(self, prepared_state: Any) -> None:
        """Complete the no-op lifecycle for cached script state."""

    @staticmethod
    def _load_tokenizer(tokenizer_path: str) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)


@dataclass(frozen=True)
class MetricResultRow:
    """Represent one scalar partition-metric observation."""

    task: ScaleTask
    partition_method: str
    metric: str
    component: str
    aggregation: str
    value: float
    scope: str
    config_hash: str
    reused: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "partition_method",
            "metric",
            "component",
            "aggregation",
            "scope",
            "config_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not np.isscalar(self.value) or not np.isfinite(self.value):
            raise ValueError("value must be a finite scalar")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be a boolean")

    @property
    def identity(self) -> tuple[Any, ...]:
        """Return the unique identity of one long-form observation."""

        return (
            self.task.task_key,
            self.partition_method,
            self.metric,
            self.component,
            self.aggregation,
            self.scope,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return one flat canonical result row."""

        return {
            **self.task.to_dict(),
            "partition_method": self.partition_method,
            "metric": self.metric,
            "component": self.component,
            "aggregation": self.aggregation,
            "value": float(self.value),
            "scope": self.scope,
            "config_hash": self.config_hash,
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricResultRow":
        """Construct one row from CSV-compatible values."""

        expected = set(METRIC_RESULT_COLUMNS)
        provided = set(data)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(f"Invalid metric result fields: missing={missing}, extra={extra}")
        seed_type = None if pd.isna(data["seed_type"]) else str(data["seed_type"])
        seed = _optional_integer(data["seed"], "seed")
        task = ScaleTask(
            analysis_id=str(data["analysis_id"]),
            model_name=str(data["model_name"]),
            space_name=str(data["space_name"]),
            scale=_required_integer(data["scale"], "scale"),
            source=str(data["source"]),
            seed_type=seed_type,
            seed=seed,
        )
        return cls(
            task=task,
            partition_method=str(data["partition_method"]),
            metric=str(data["metric"]),
            component=str(data["component"]),
            aggregation=str(data["aggregation"]),
            value=float(data["value"]),
            scope=str(data["scope"]),
            config_hash=str(data["config_hash"]),
            reused=_boolean(data["reused"], "reused"),
        )


@dataclass(frozen=True)
class MetricResultTable:
    """Collect long-form metric observations across tasks and metrics."""

    rows: tuple[MetricResultRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        identities = [row.identity for row in self.rows]
        if len(identities) != len(set(identities)):
            raise ValueError("metric result table contains duplicate observation identities")

    def to_frame(self) -> pd.DataFrame:
        """Return the canonical long-form DataFrame."""

        frame = pd.DataFrame(
            [row.to_dict() for row in self.rows],
            columns=METRIC_RESULT_COLUMNS,
        )
        if not frame.empty:
            frame["seed"] = pd.array(frame["seed"], dtype="Int64")
        return frame

    @classmethod
    def combine(cls, *tables: "MetricResultTable") -> "MetricResultTable":
        """Combine metric tables while preserving duplicate validation."""

        if any(not isinstance(table, MetricResultTable) for table in tables):
            raise TypeError("tables must contain MetricResultTable objects")
        return cls(rows=tuple(row for table in tables for row in table.rows))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "MetricResultTable":
        """Validate and construct a result table from a DataFrame."""

        if tuple(frame.columns) != METRIC_RESULT_COLUMNS:
            raise ValueError("metric result columns must exactly match canonical order")
        return cls(
            rows=tuple(
                MetricResultRow.from_dict(record)
                for record in frame.to_dict(orient="records")
            )
        )

    def write_csv(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write results without silently replacing an existing table."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Metric result table already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(destination, index=False, encoding="utf-8")
        return destination

    @classmethod
    def read_csv(cls, path: str | Path) -> "MetricResultTable":
        """Read and validate a canonical metric result CSV."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Metric result table not found: {source}")
        return cls.from_frame(pd.read_csv(source))


def orthographic_similarity_summary_frame(
    results: MetricResultTable,
    *,
    l2_norm: bool = True,
    min_cluster_size: int = 5,
    min_samples: int = 5,
    distance_metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
) -> pd.DataFrame:
    """Return one legacy morphology-style summary row per O-S task."""

    if not isinstance(results, MetricResultTable):
        raise TypeError("results must be a MetricResultTable")
    if not isinstance(l2_norm, bool):
        raise TypeError("l2_norm must be a boolean")
    for field_name, value in (
        ("min_cluster_size", min_cluster_size),
        ("min_samples", min_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
    for field_name, value in (
        ("distance_metric", distance_metric),
        ("cluster_selection_method", cluster_selection_method),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if (
        isinstance(cluster_selection_epsilon, bool)
        or not np.isscalar(cluster_selection_epsilon)
        or not np.isfinite(cluster_selection_epsilon)
        or float(cluster_selection_epsilon) < 0.0
    ):
        raise ValueError("cluster_selection_epsilon must be a finite non-negative scalar")

    selected_rows = tuple(
        row for row in results.rows if row.metric == "orthographic_similarity"
    )
    if not selected_rows:
        raise ValueError("results contain no orthographic_similarity observations")

    expected_observations = {
        ("score", "mean", "partition"),
        ("score", "std", "partition"),
        ("cluster_count", "count", "partition"),
    }
    grouped_rows: dict[tuple[str, str], list[MetricResultRow]] = {}
    for row in selected_rows:
        group_key = (row.task.task_key, row.partition_method)
        grouped_rows.setdefault(group_key, []).append(row)

    seed_types = {row.task.seed_type for row in selected_rows}
    if None in seed_types and len(seed_types) > 1:
        raise ValueError("O-S summary cannot mix seeded and unseeded tasks")
    if len(seed_types) > 1:
        raise ValueError("O-S summary requires one consistent seed type")
    seed_type = next(iter(seed_types))

    records: list[dict[str, Any]] = []
    for group_rows in grouped_rows.values():
        task = group_rows[0].task
        observations = {
            (row.component, row.aggregation, row.scope): float(row.value)
            for row in group_rows
        }
        if set(observations) != expected_observations:
            missing = sorted(expected_observations - set(observations))
            extra = sorted(set(observations) - expected_observations)
            raise ValueError(
                f"Invalid O-S observations for {task.task_key}: "
                f"missing={missing}, extra={extra}"
            )
        n_clusters = observations[("cluster_count", "count", "partition")]
        if not float(n_clusters).is_integer() or n_clusters < 1:
            raise ValueError("O-S cluster count must be a positive integer")

        record: dict[str, Any] = {
            "model_name": task.model_name,
            "space_name": task.space_name,
        }
        if seed_type is not None:
            record[seed_type] = task.seed
        record.update(
            {
                "pca_dim": task.scale,
                "l2_norm": l2_norm,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": distance_metric,
                "cluster_selection_method": cluster_selection_method,
                "cluster_selection_epsilon": float(cluster_selection_epsilon),
                "global_mean": observations[("score", "mean", "partition")],
                "global_std": observations[("score", "std", "partition")],
                "n_clusters": int(n_clusters),
            }
        )
        records.append(record)

    output = pd.DataFrame(records)
    sort_columns = ["model_name"]
    if seed_type is not None:
        sort_columns.append(seed_type)
    sort_columns.append("pca_dim")
    output = output.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if seed_type is not None:
        output[seed_type] = pd.array(output[seed_type], dtype="Int64")
    return output


def multi_script_entropy_summary_frame(
    results: MetricResultTable,
    *,
    l2_norm: bool = True,
    min_cluster_size: int = 5,
    min_samples: int = 5,
    distance_metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
) -> pd.DataFrame:
    """Return one legacy script-entropy-style summary row per MS-E task."""

    if not isinstance(results, MetricResultTable):
        raise TypeError("results must be a MetricResultTable")
    if not isinstance(l2_norm, bool):
        raise TypeError("l2_norm must be a boolean")
    for field_name, value in (
        ("min_cluster_size", min_cluster_size),
        ("min_samples", min_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
    for field_name, value in (
        ("distance_metric", distance_metric),
        ("cluster_selection_method", cluster_selection_method),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if (
        isinstance(cluster_selection_epsilon, bool)
        or not np.isscalar(cluster_selection_epsilon)
        or not np.isfinite(cluster_selection_epsilon)
        or float(cluster_selection_epsilon) < 0.0
    ):
        raise ValueError("cluster_selection_epsilon must be a finite non-negative scalar")

    selected_rows = tuple(
        row for row in results.rows if row.metric == "multi_script_entropy"
    )
    if not selected_rows:
        raise ValueError("results contain no multi_script_entropy observations")

    expected_observations = {
        ("raw", "mean", "partition"),
        ("raw", "std", "partition"),
        ("normalized", "mean", "partition"),
        ("normalized", "std", "partition"),
        ("cluster_count", "count", "partition"),
    }
    grouped_rows: dict[tuple[str, str], list[MetricResultRow]] = {}
    for row in selected_rows:
        group_key = (row.task.task_key, row.partition_method)
        grouped_rows.setdefault(group_key, []).append(row)

    seed_types = {row.task.seed_type for row in selected_rows}
    if None in seed_types and len(seed_types) > 1:
        raise ValueError("MS-E summary cannot mix seeded and unseeded tasks")
    if len(seed_types) > 1:
        raise ValueError("MS-E summary requires one consistent seed type")
    seed_type = next(iter(seed_types))

    records: list[dict[str, Any]] = []
    for group_rows in grouped_rows.values():
        task = group_rows[0].task
        observations = {
            (row.component, row.aggregation, row.scope): float(row.value)
            for row in group_rows
        }
        if set(observations) != expected_observations:
            missing = sorted(expected_observations - set(observations))
            extra = sorted(set(observations) - expected_observations)
            raise ValueError(
                f"Invalid MS-E observations for {task.task_key}: "
                f"missing={missing}, extra={extra}"
            )
        num_clusters = observations[("cluster_count", "count", "partition")]
        if not float(num_clusters).is_integer() or num_clusters < 1:
            raise ValueError("MS-E cluster count must be a positive integer")

        record: dict[str, Any] = {
            "model_name": task.model_name,
            "space_name": task.space_name,
        }
        if seed_type is not None:
            record[seed_type] = task.seed
        record.update(
            {
                "pca_dim": task.scale,
                "l2_norm": l2_norm,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": distance_metric,
                "cluster_selection_method": cluster_selection_method,
                "cluster_selection_epsilon": float(cluster_selection_epsilon),
                "num_clusters": int(num_clusters),
                "mean_H": observations[("raw", "mean", "partition")],
                "std_H": observations[("raw", "std", "partition")],
                "mean_H_norm": observations[
                    ("normalized", "mean", "partition")
                ],
                "std_H_norm": observations[
                    ("normalized", "std", "partition")
                ],
            }
        )
        records.append(record)

    output = pd.DataFrame(records)
    sort_columns = ["model_name"]
    if seed_type is not None:
        sort_columns.append(seed_type)
    sort_columns.append("pca_dim")
    output = output.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if seed_type is not None:
        output[seed_type] = pd.array(output[seed_type], dtype="Int64")
    return output


def aggregate_seed_metric_results(
    results: MetricResultTable,
    *,
    input_analysis_id: str,
    input_source: str,
    seed_type: str,
    expected_seeds: tuple[int, ...],
    output_analysis_id: str,
    output_source: str,
    input_aggregation: str = "mean",
    ddof: int = 1,
) -> MetricResultTable:
    """Aggregate seeded metric curves into canonical seed-mean and seed-SD rows."""

    if not isinstance(results, MetricResultTable):
        raise TypeError("results must be a MetricResultTable")
    for field_name, value in (
        ("input_analysis_id", input_analysis_id),
        ("input_source", input_source),
        ("seed_type", seed_type),
        ("output_analysis_id", output_analysis_id),
        ("output_source", output_source),
        ("input_aggregation", input_aggregation),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(expected_seeds, tuple) or not expected_seeds:
        raise ValueError("expected_seeds must be a non-empty tuple")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in expected_seeds
    ):
        raise ValueError("expected_seeds must contain non-negative integers")
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("expected_seeds must be unique")
    if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if ddof >= len(expected_seeds):
        raise ValueError("ddof must be smaller than the number of expected seeds")

    selected = tuple(
        row
        for row in results.rows
        if row.task.analysis_id == input_analysis_id
        and row.task.source == input_source
        and row.task.seed_type == seed_type
        and row.aggregation == input_aggregation
    )
    if not selected:
        raise ValueError("no metric observations matched the seeded aggregation request")

    groups: dict[tuple[Any, ...], list[MetricResultRow]] = {}
    for row in selected:
        key = (
            row.task.model_name,
            row.task.space_name,
            row.task.scale,
            row.partition_method,
            row.metric,
            row.component,
            row.scope,
        )
        groups.setdefault(key, []).append(row)

    output_rows: list[MetricResultRow] = []
    for key in sorted(groups):
        group_rows = groups[key]
        rows_by_seed = {row.task.seed: row for row in group_rows}
        if set(rows_by_seed) != set(expected_seeds):
            raise ValueError(
                "seeded metric group does not match expected seeds: "
                f"group={key}, observed={sorted(rows_by_seed)}, "
                f"expected={sorted(expected_seeds)}"
            )
        ordered_rows = [rows_by_seed[seed] for seed in expected_seeds]
        values = np.asarray([row.value for row in ordered_rows], dtype=np.float64)
        model_name, space_name, scale, method, metric, component, scope = key
        task = ScaleTask(
            analysis_id=output_analysis_id,
            model_name=model_name,
            space_name=space_name,
            scale=scale,
            source=output_source,
        )
        config_hash = stable_config_hash(
            {
                "schema_version": 1,
                "operation": "aggregate_seed_metric_results",
                "input_analysis_id": input_analysis_id,
                "input_source": input_source,
                "seed_type": seed_type,
                "expected_seeds": list(expected_seeds),
                "input_aggregation": input_aggregation,
                "ddof": ddof,
                "input_config_hashes": [row.config_hash for row in ordered_rows],
            }
        )
        for aggregation, value in (
            ("seed_mean", float(np.mean(values))),
            ("seed_std", float(np.std(values, ddof=ddof))),
        ):
            output_rows.append(
                MetricResultRow(
                    task=task,
                    partition_method=method,
                    metric=metric,
                    component=component,
                    aggregation=aggregation,
                    value=value,
                    scope=scope,
                    config_hash=config_hash,
                )
            )
    return MetricResultTable(rows=tuple(output_rows))


@dataclass(frozen=True)
class MetricOutputPaths:
    """Hold files that make one partition-metric output complete."""

    task_dir: Path
    summary_csv: Path
    metadata_json: Path
    details_dir: Path


class MetricOutputLayout:
    """Resolve metric outputs entirely beneath an explicit output root."""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)

    def paths_for(
        self,
        task: ScaleTask,
        metric_name: str,
        partition_method: str,
    ) -> MetricOutputPaths:
        """Resolve output paths for one metric and partition."""

        _validate_path_segment(metric_name, "metric_name")
        _validate_path_segment(partition_method, "partition_method")
        task_dir = (
            self.output_root
            / task.analysis_id
            / "metrics"
            / metric_name
            / task.model_name
            / task.space_name
            / task.source
        )
        if task.seed is not None:
            task_dir = task_dir / str(task.seed_type) / f"seed_{task.seed:05d}"
        task_dir = task_dir / partition_method / f"scale_{task.scale}"
        return MetricOutputPaths(
            task_dir=task_dir,
            summary_csv=task_dir / "summary.csv",
            metadata_json=task_dir / "metadata.json",
            details_dir=task_dir / "details",
        )

    def result_table_path(self, analysis_id: str, metric_name: str) -> Path:
        """Return the aggregate long-form result path for one run."""

        _validate_path_segment(metric_name, "metric_name")
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            raise ValueError("analysis_id must be a non-empty string")
        return (
            self.output_root
            / analysis_id
            / "metric_results"
            / f"{metric_name}.csv"
        )


ModelContextFactory = Callable[
    [str, tuple[PartitionManifestRow, ...]],
    ModelMetricContext,
]


@dataclass(frozen=True)
class PartitionMetricPipelineRun:
    """Return metric results and execution provenance from one run."""

    results: MetricResultTable
    report: StageRunReport
    result_table_path: Path


class PartitionMetricPipeline:
    """Schedule one partition metric over an explicit partition manifest."""

    stage = "partition_metric"

    def __init__(
        self,
        *,
        metric: PartitionMetric,
        output_layout: MetricOutputLayout,
        run_policy: RunPolicy | None = None,
        context_factory: ModelContextFactory | None = None,
        partition_loader: Callable[[str | Path], pd.DataFrame] = pd.read_csv,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(metric, PartitionMetric):
            raise TypeError("metric must satisfy PartitionMetric")
        _validate_path_segment(metric.name, "metric.name")
        self.metric = metric
        self.output_layout = output_layout
        self.run_policy = run_policy or RunPolicy()
        self._context_factory = context_factory or self._default_context_factory
        self._partition_loader = partition_loader
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def config(self) -> dict[str, Any]:
        """Return task-independent metric computation configuration."""

        return {
            "schema_version": 1,
            "stage": self.stage,
            "metric": {
                "name": self.metric.name,
                "parameters": dict(self.metric.config()),
            },
        }

    def run(
        self,
        manifest: PartitionManifest | str | Path,
    ) -> PartitionMetricPipelineRun:
        """Evaluate the configured metric over all manifest rows."""

        partition_manifest = (
            PartitionManifest.read_csv(manifest)
            if isinstance(manifest, (str, Path))
            else manifest
        )
        if not isinstance(partition_manifest, PartitionManifest):
            raise TypeError("manifest must be a PartitionManifest or CSV path")
        if not partition_manifest.rows:
            raise ValueError("partition manifest must contain at least one row")
        task_keys = [row.task.task_key for row in partition_manifest.rows]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("one metric run cannot contain multiple partitions per task")
        analysis_ids = {row.task.analysis_id for row in partition_manifest.rows}
        if len(analysis_ids) != 1:
            raise ValueError("one metric run must use a single analysis_id")

        analysis_id = next(iter(analysis_ids))
        result_table_path = self.output_layout.result_table_path(
            analysis_id,
            self.metric.name,
        )
        base_config = self.config()
        if self.run_policy.existing_output == "error":
            self._preflight_error_mode(partition_manifest.rows, result_table_path)

        model_rows = self._group_rows_by_model(partition_manifest.rows)
        prepared_states: dict[str, Any] = {}
        result_rows: list[MetricResultRow] = []
        records: list[StageRunRecord] = []
        try:
            for partition_row in partition_manifest.rows:
                task = partition_row.task
                paths = self.output_layout.paths_for(
                    task,
                    self.metric.name,
                    partition_row.method,
                )
                config = self._task_config(base_config, partition_row)
                config_hash = stable_config_hash(config)
                started_at = self._timestamp()
                try:
                    existing = paths.summary_csv.exists() or paths.metadata_json.exists()
                    if self.run_policy.existing_output == "resume" and existing:
                        task_rows, detail_paths = validate_metric_output(
                            paths,
                            expected_task=task,
                            expected_partition_method=partition_row.method,
                            expected_metric=self.metric.name,
                            expected_config_hash=config_hash,
                        )
                        status = "reused"
                    else:
                        if task.model_name not in prepared_states:
                            context = self._context_factory(
                                task.model_name,
                                model_rows[task.model_name],
                            )
                            if not isinstance(context, ModelMetricContext):
                                raise TypeError(
                                    "context_factory must return ModelMetricContext"
                                )
                            if context.model_name != task.model_name:
                                raise ValueError(
                                    "model context name does not match partition model"
                                )
                            prepared_states[task.model_name] = self.metric.prepare(context)
                        partition = self._load_partition(partition_row)
                        evaluation = self.metric.evaluate(
                            partition,
                            prepared_states[task.model_name],
                        )
                        if not isinstance(evaluation, MetricEvaluation):
                            raise TypeError("metric.evaluate must return MetricEvaluation")
                        task_rows, detail_paths = write_metric_output(
                            evaluation,
                            paths,
                            task=task,
                            partition_method=partition_row.method,
                            metric_name=self.metric.name,
                            config_hash=config_hash,
                            config=config,
                            atomic_write=self.run_policy.atomic_write,
                            replace_existing=self.run_policy.existing_output == "replace",
                        )
                        status = "computed"
                    result_rows.extend(task_rows)
                    records.append(
                        StageRunRecord(
                            task=task,
                            stage=self.stage,
                            status=status,
                            output_paths=(
                                str(paths.summary_csv),
                                str(paths.metadata_json),
                                *(str(path) for path in detail_paths),
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
                            output_paths=(str(paths.summary_csv), str(paths.metadata_json)),
                            config_hash=config_hash,
                            started_at=started_at,
                            ended_at=self._timestamp(),
                            error_message=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    if self.run_policy.stop_on_error:
                        raise
        finally:
            for state in prepared_states.values():
                self.metric.finalize(state)

        results = MetricResultTable(rows=tuple(result_rows))
        _write_metric_result_table(
            results,
            result_table_path,
            atomic_write=self.run_policy.atomic_write,
        )
        return PartitionMetricPipelineRun(
            results=results,
            report=StageRunReport(stage=self.stage, records=tuple(records)),
            result_table_path=result_table_path,
        )

    def _load_partition(self, row: PartitionManifestRow) -> pd.DataFrame:
        partition = self._partition_loader(row.partition_csv_path)
        if not isinstance(partition, pd.DataFrame):
            raise TypeError("partition_loader must return a pandas DataFrame")
        required = {"token_id", "cluster_id"}
        missing = sorted(required - set(partition.columns))
        if missing:
            raise ValueError(f"partition is missing required columns: {missing}")
        if len(partition) != row.n_tokens:
            raise ValueError("partition row count does not match manifest n_tokens")
        return partition

    def _preflight_error_mode(
        self,
        rows: tuple[PartitionManifestRow, ...],
        result_table_path: Path,
    ) -> None:
        existing_paths = [str(result_table_path)] if result_table_path.exists() else []
        for row in rows:
            paths = self.output_layout.paths_for(
                row.task,
                self.metric.name,
                row.method,
            )
            existing_paths.extend(
                str(path)
                for path in (paths.summary_csv, paths.metadata_json)
                if path.exists()
            )
        if existing_paths:
            raise FileExistsError(
                f"Metric outputs already exist: {sorted(existing_paths)}"
            )

    @staticmethod
    def _group_rows_by_model(
        rows: tuple[PartitionManifestRow, ...],
    ) -> dict[str, tuple[PartitionManifestRow, ...]]:
        grouped: dict[str, list[PartitionManifestRow]] = {}
        for row in rows:
            grouped.setdefault(row.task.model_name, []).append(row)
        return {name: tuple(items) for name, items in grouped.items()}

    @staticmethod
    def _default_context_factory(
        model_name: str,
        rows: tuple[PartitionManifestRow, ...],
    ) -> ModelMetricContext:
        return ModelMetricContext(model_name=model_name)

    @staticmethod
    def _task_config(
        base_config: Mapping[str, Any],
        row: PartitionManifestRow,
    ) -> dict[str, Any]:
        return {
            **dict(base_config),
            "partition": {
                "method": row.method,
                "path": row.partition_csv_path,
                "config_hash": row.config_hash,
            },
        }

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return datetime objects")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


def write_metric_output(
    evaluation: MetricEvaluation,
    paths: MetricOutputPaths,
    *,
    task: ScaleTask,
    partition_method: str,
    metric_name: str,
    config_hash: str,
    config: Mapping[str, Any],
    atomic_write: bool = True,
    replace_existing: bool = False,
) -> tuple[tuple[MetricResultRow, ...], tuple[Path, ...]]:
    """Persist one evaluation, committing metadata after all data files."""

    detail_destinations = {
        name: paths.details_dir / f"{name}.csv"
        for name in evaluation.detail_tables
    }
    for name in detail_destinations:
        _validate_path_segment(name, "detail table name")
    destinations = (paths.summary_csv, paths.metadata_json, *detail_destinations.values())
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not replace_existing:
        raise FileExistsError(f"Metric output already exists: {existing}")

    rows = tuple(
        MetricResultRow(
            task=task,
            partition_method=partition_method,
            metric=metric_name,
            component=observation.component,
            aggregation=observation.aggregation,
            value=float(observation.value),
            scope=observation.scope,
            config_hash=config_hash,
            reused=False,
        )
        for observation in evaluation.observations
    )
    summary = MetricResultTable(rows=rows).to_frame()
    paths.task_dir.mkdir(parents=True, exist_ok=True)
    if detail_destinations:
        paths.details_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "task": task.to_dict(),
        "task_key": task.task_key,
        "partition_method": partition_method,
        "metric": metric_name,
        "config_hash": config_hash,
        "config": dict(config),
        "evaluation_metadata": dict(evaluation.metadata),
        "summary_csv_path": str(paths.summary_csv),
        "detail_csv_paths": {
            name: str(path) for name, path in detail_destinations.items()
        },
    }

    temporary_files: dict[Path, Path] = {}
    try:
        if atomic_write:
            for destination in (paths.summary_csv, *detail_destinations.values()):
                temporary_files[destination] = _metric_temporary_sibling(destination)
            metadata_temp = _metric_temporary_sibling(paths.metadata_json)
            temporary_files[paths.metadata_json] = metadata_temp
            summary.to_csv(temporary_files[paths.summary_csv], index=False, encoding="utf-8")
            for name, table in evaluation.detail_tables.items():
                table.to_csv(
                    temporary_files[detail_destinations[name]],
                    index=False,
                    encoding="utf-8",
                    escapechar="\\",
                )
            _write_metric_json(metadata_temp, metadata)
            for destination in (paths.summary_csv, *detail_destinations.values()):
                os.replace(temporary_files[destination], destination)
            os.replace(metadata_temp, paths.metadata_json)
        else:
            summary.to_csv(paths.summary_csv, index=False, encoding="utf-8")
            for name, table in evaluation.detail_tables.items():
                table.to_csv(
                    detail_destinations[name],
                    index=False,
                    encoding="utf-8",
                    escapechar="\\",
                )
            _write_metric_json(paths.metadata_json, metadata)
    finally:
        for temporary_path in temporary_files.values():
            if temporary_path.exists():
                temporary_path.unlink()
    return rows, tuple(detail_destinations.values())


def validate_metric_output(
    paths: MetricOutputPaths,
    *,
    expected_task: ScaleTask,
    expected_partition_method: str,
    expected_metric: str,
    expected_config_hash: str,
) -> tuple[tuple[MetricResultRow, ...], tuple[Path, ...]]:
    """Validate a complete metric output before resume."""

    missing = [
        str(path)
        for path in (paths.summary_csv, paths.metadata_json)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete metric output; missing files: {missing}")
    with paths.metadata_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_fields = {
        "schema_version": 1,
        "task": expected_task.to_dict(),
        "task_key": expected_task.task_key,
        "partition_method": expected_partition_method,
        "metric": expected_metric,
        "config_hash": expected_config_hash,
        "summary_csv_path": str(paths.summary_csv),
    }
    for field_name, expected_value in expected_fields.items():
        if metadata.get(field_name) != expected_value:
            raise ValueError(f"saved {field_name} does not match expected metric output")

    detail_mapping = metadata.get("detail_csv_paths")
    if not isinstance(detail_mapping, dict):
        raise ValueError("saved detail_csv_paths must be a mapping")
    detail_paths = tuple(Path(path) for path in detail_mapping.values())
    missing_details = [str(path) for path in detail_paths if not path.is_file()]
    if missing_details:
        raise FileNotFoundError(
            f"Incomplete metric output; missing detail files: {missing_details}"
        )

    saved = MetricResultTable.read_csv(paths.summary_csv)
    if not saved.rows:
        raise ValueError("saved metric summary must contain at least one row")
    for row in saved.rows:
        if row.task != expected_task:
            raise ValueError("saved metric task does not match expected task")
        if row.partition_method != expected_partition_method:
            raise ValueError("saved partition method does not match expected method")
        if row.metric != expected_metric:
            raise ValueError("saved metric name does not match expected metric")
        if row.config_hash != expected_config_hash:
            raise ValueError("saved metric config hash does not match expected hash")
    return tuple(replace(row, reused=True) for row in saved.rows), detail_paths


def _write_metric_result_table(
    table: MetricResultTable,
    destination: Path,
    *,
    atomic_write: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if atomic_write:
        temporary_path = _metric_temporary_sibling(destination)
        try:
            table.to_frame().to_csv(temporary_path, index=False, encoding="utf-8")
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    else:
        table.to_frame().to_csv(destination, index=False, encoding="utf-8")


def _metric_temporary_sibling(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _write_metric_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2, allow_nan=False)


def _validate_path_segment(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if Path(value).name != value:
        raise ValueError(f"{field_name} must be a single path-safe segment")


def _required_integer(value: Any, field_name: str) -> int:
    if pd.isna(value) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    integer = int(value)
    if float(value) != integer:
        raise ValueError(f"{field_name} must be an integer")
    return integer


def _optional_integer(value: Any, field_name: str) -> int | None:
    if pd.isna(value):
        return None
    return _required_integer(value, field_name)


def _boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field_name} must be a boolean")
