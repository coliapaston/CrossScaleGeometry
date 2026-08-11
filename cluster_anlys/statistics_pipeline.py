"""Composable curve selection and statistics contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .metric_pipeline import MetricResultTable
from .pipeline_core import stable_config_hash


CURVE_TABLE_COLUMNS = (
    "curve_id",
    "analysis_id",
    "model_name",
    "space_name",
    "source",
    "seed_type",
    "seed",
    "partition_method",
    "metric",
    "component",
    "aggregation",
    "scope",
    "scale",
    "value",
    "config_hash",
    "reused",
)

STATISTIC_RESULT_COLUMNS = (
    "reference_curve_id",
    "target_curve_id",
    "alignment_strategy",
    "statistic",
    "component",
    "value",
    "n_points",
    "config_hash",
    "reused",
)

ALIGNMENT_STRATEGIES = ("exact", "intersection", "reference")
PAIR_MODES = ("all_pairs", "reference_only")


@dataclass(frozen=True, order=True)
class CompositeObservationKey:
    """Identify one feature-scale observation inside a composed vector."""

    feature_slot: str
    scale: int

    def __post_init__(self) -> None:
        if not isinstance(self.feature_slot, str) or not self.feature_slot.strip():
            raise ValueError("feature_slot must be a non-empty string")
        if isinstance(self.scale, bool) or not isinstance(self.scale, int):
            raise TypeError("scale must be an integer")
        if self.scale <= 0:
            raise ValueError("scale must be positive")


@dataclass(frozen=True)
class ComposedCurve:
    """Represent one vector with explicit feature-scale observation keys."""

    curve_id: str
    composer_name: str
    observation_keys: tuple[CompositeObservationKey, ...]
    values: tuple[float, ...]
    source_curve_ids: tuple[str, ...]
    config_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("curve_id", "composer_name", "config_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.observation_keys, tuple) or not self.observation_keys:
            raise ValueError("observation_keys must be a non-empty tuple")
        if any(
            not isinstance(key, CompositeObservationKey)
            for key in self.observation_keys
        ):
            raise TypeError(
                "observation_keys must contain CompositeObservationKey objects"
            )
        if len(self.observation_keys) != len(set(self.observation_keys)):
            raise ValueError("observation_keys must be unique")
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple")
        if len(self.values) != len(self.observation_keys):
            raise ValueError("values must match observation_keys length")
        if not np.all(np.isfinite(np.asarray(self.values, dtype=float))):
            raise ValueError("values must be finite")
        if not isinstance(self.source_curve_ids, tuple) or not self.source_curve_ids:
            raise ValueError("source_curve_ids must be a non-empty tuple")
        if any(
            not isinstance(curve_id, str) or not curve_id.strip()
            for curve_id in self.source_curve_ids
        ):
            raise ValueError("source_curve_ids must contain non-empty strings")
        if len(self.source_curve_ids) != len(set(self.source_curve_ids)):
            raise ValueError("source_curve_ids must be unique")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) or not key for key in self.metadata):
            raise ValueError("metadata keys must be non-empty strings")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def array(self) -> np.ndarray:
        """Return a read-only numeric view of the composed values."""

        array = np.asarray(self.values, dtype=float)
        array.setflags(write=False)
        return array

    def to_frame(self) -> pd.DataFrame:
        """Return feature slots, physical scales, and values in vector order."""

        return pd.DataFrame({
            "feature_slot": [key.feature_slot for key in self.observation_keys],
            "scale": [key.scale for key in self.observation_keys],
            "value": [float(value) for value in self.values],
        })


@dataclass(frozen=True)
class ComposedAlignedPair:
    """Hold reference and target composed vectors with identical keys."""

    reference: ComposedCurve
    target: ComposedCurve

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ComposedCurve):
            raise TypeError("reference must be a ComposedCurve")
        if not isinstance(self.target, ComposedCurve):
            raise TypeError("target must be a ComposedCurve")
        if self.reference.curve_id == self.target.curve_id:
            raise ValueError("reference and target curve ids must differ")
        if self.reference.composer_name != self.target.composer_name:
            raise ValueError("reference and target must share one composer")
        if self.reference.config_hash != self.target.config_hash:
            raise ValueError("reference and target must share one composer config")
        if self.reference.observation_keys != self.target.observation_keys:
            raise ValueError("reference and target must have identical observation keys")

    @property
    def observation_keys(self) -> tuple[CompositeObservationKey, ...]:
        """Return the shared ordered composite observation keys."""

        return self.reference.observation_keys

    @property
    def reference_array(self) -> np.ndarray:
        """Return the read-only reference vector."""

        return self.reference.array

    @property
    def target_array(self) -> np.ndarray:
        """Return the read-only target vector."""

        return self.target.array


@runtime_checkable
class CurveComposer(Protocol):
    """Compose corresponding reference and target curves into paired vectors."""

    name: str

    def config(self) -> Mapping[str, Any]:
        """Return the effective composition configuration."""

    def compose(
        self,
        reference_curves: tuple["CurveTable", ...],
        target_curves: tuple["CurveTable", ...],
    ) -> ComposedAlignedPair:
        """Compose paired curve collections using explicit feature-scale keys."""


class ZScoreConcatenationComposer:
    """Concatenate feature blocks standardized against reference curves."""

    name = "zscore_concatenation"

    def __init__(
        self,
        *,
        feature_slots: tuple[str, ...],
        alignment_strategy: str = "exact",
    ) -> None:
        if not isinstance(feature_slots, tuple) or len(feature_slots) < 2:
            raise ValueError("feature_slots must contain at least two names")
        if any(
            not isinstance(feature_slot, str) or not feature_slot.strip()
            for feature_slot in feature_slots
        ):
            raise ValueError("feature_slots must contain non-empty strings")
        if len(feature_slots) != len(set(feature_slots)):
            raise ValueError("feature_slots must be unique")
        if alignment_strategy not in ("exact", "intersection"):
            raise ValueError(
                "alignment_strategy must be either exact or intersection"
            )
        self.feature_slots = feature_slots
        self.alignment_strategy = alignment_strategy

    def config(self) -> Mapping[str, Any]:
        """Return the reference-owned population z-score configuration."""

        return {
            "schema_version": 1,
            "feature_slots": list(self.feature_slots),
            "alignment_strategy": self.alignment_strategy,
            "normalization": "reference_zscore",
            "standard_deviation_ddof": 0,
            "constant_reference_fallback": "mean_center",
            "concatenation_order": "feature_block_then_scale",
        }

    def compose(
        self,
        reference_curves: tuple["CurveTable", ...],
        target_curves: tuple["CurveTable", ...],
    ) -> ComposedAlignedPair:
        """Align features, fit each transform on reference, and concatenate."""

        self._validate_curve_collections(reference_curves, target_curves)
        aligner = CurveAligner(self.alignment_strategy)
        aligned_features = tuple(
            aligner.align(reference, target)
            for reference, target in zip(reference_curves, target_curves)
        )
        shared_scales = self._shared_scales(aligned_features)

        from .spearman import _safe_standardize_against_baseline

        observation_keys = []
        reference_blocks = []
        target_blocks = []
        normalization = {}
        for feature_slot, aligned in zip(self.feature_slots, aligned_features):
            reference_by_scale = dict(zip(aligned.scales, aligned.reference_values))
            target_by_scale = dict(zip(aligned.scales, aligned.target_values))
            reference_values = np.asarray(
                [reference_by_scale[scale] for scale in shared_scales],
                dtype=float,
            )
            target_values = np.asarray(
                [target_by_scale[scale] for scale in shared_scales],
                dtype=float,
            )
            reference_z, target_z = _safe_standardize_against_baseline(
                reference_values,
                target_values,
            )
            reference_blocks.extend(float(value) for value in reference_z)
            target_blocks.extend(float(value) for value in target_z)
            observation_keys.extend(
                CompositeObservationKey(feature_slot, scale)
                for scale in shared_scales
            )
            normalization[feature_slot] = {
                "reference_mean": float(np.mean(reference_values)),
                "reference_std_ddof0": float(np.std(reference_values, ddof=0)),
            }

        composer_config_hash = stable_config_hash(dict(self.config()))
        reference_source_ids = tuple(
            aligned.reference_curve_id for aligned in aligned_features
        )
        target_source_ids = tuple(
            aligned.target_curve_id for aligned in aligned_features
        )
        shared_metadata = {
            "shared_scales": list(shared_scales),
            "normalization": normalization,
        }
        reference = ComposedCurve(
            curve_id=stable_config_hash({
                "composer_config_hash": composer_config_hash,
                "role": "reference",
                "source_curve_ids": list(reference_source_ids),
            }),
            composer_name=self.name,
            observation_keys=tuple(observation_keys),
            values=tuple(reference_blocks),
            source_curve_ids=reference_source_ids,
            config_hash=composer_config_hash,
            metadata=shared_metadata,
        )
        target = ComposedCurve(
            curve_id=stable_config_hash({
                "composer_config_hash": composer_config_hash,
                "role": "target",
                "source_curve_ids": list(target_source_ids),
            }),
            composer_name=self.name,
            observation_keys=tuple(observation_keys),
            values=tuple(target_blocks),
            source_curve_ids=target_source_ids,
            config_hash=composer_config_hash,
            metadata=shared_metadata,
        )
        return ComposedAlignedPair(reference=reference, target=target)

    def _validate_curve_collections(
        self,
        reference_curves: tuple["CurveTable", ...],
        target_curves: tuple["CurveTable", ...],
    ) -> None:
        for field_name, curves in (
            ("reference_curves", reference_curves),
            ("target_curves", target_curves),
        ):
            if not isinstance(curves, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(not isinstance(curve, CurveTable) for curve in curves):
                raise TypeError(f"{field_name} must contain CurveTable objects")
        if len(reference_curves) != len(self.feature_slots):
            raise ValueError("reference_curves must match feature_slots length")
        if len(target_curves) != len(self.feature_slots):
            raise ValueError("target_curves must match feature_slots length")

    def _shared_scales(
        self,
        aligned_features: tuple["AlignedCurvePair", ...],
    ) -> tuple[int, ...]:
        scale_sets = [set(aligned.scales) for aligned in aligned_features]
        if self.alignment_strategy == "exact":
            if any(scales != scale_sets[0] for scales in scale_sets[1:]):
                raise ValueError(
                    "exact composition requires identical scales across features"
                )
            return tuple(sorted(scale_sets[0]))
        shared = set.intersection(*scale_sets)
        if not shared:
            raise ValueError("intersection composition found no common feature scales")
        return tuple(sorted(shared))


class WeightedMeanComposer:
    """Average commensurable feature curves with explicit fixed weights."""

    name = "weighted_mean"

    def __init__(
        self,
        *,
        feature_slots: tuple[str, ...],
        weights: tuple[float, ...],
        output_feature_slot: str = "weighted_mean",
        alignment_strategy: str = "exact",
    ) -> None:
        if not isinstance(feature_slots, tuple) or len(feature_slots) < 2:
            raise ValueError("feature_slots must contain at least two names")
        if any(
            not isinstance(feature_slot, str) or not feature_slot.strip()
            for feature_slot in feature_slots
        ):
            raise ValueError("feature_slots must contain non-empty strings")
        if len(feature_slots) != len(set(feature_slots)):
            raise ValueError("feature_slots must be unique")
        if not isinstance(weights, tuple) or len(weights) != len(feature_slots):
            raise ValueError("weights must match feature_slots length")
        weight_array = np.asarray(weights, dtype=float)
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must be finite")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")
        if float(np.sum(weight_array)) <= 0.0:
            raise ValueError("weights must contain at least one positive value")
        if not isinstance(output_feature_slot, str) or not output_feature_slot.strip():
            raise ValueError("output_feature_slot must be a non-empty string")
        if alignment_strategy not in ("exact", "intersection"):
            raise ValueError(
                "alignment_strategy must be either exact or intersection"
            )
        self.feature_slots = feature_slots
        self.weights = tuple(float(weight) for weight in weight_array)
        self.normalized_weights = tuple(
            float(weight) for weight in weight_array / np.sum(weight_array)
        )
        self.output_feature_slot = output_feature_slot
        self.alignment_strategy = alignment_strategy

    def config(self) -> Mapping[str, Any]:
        """Return the explicit raw-scale weighted arithmetic mean configuration."""

        return {
            "schema_version": 1,
            "feature_slots": list(self.feature_slots),
            "weights": list(self.weights),
            "normalized_weights": list(self.normalized_weights),
            "output_feature_slot": self.output_feature_slot,
            "alignment_strategy": self.alignment_strategy,
            "normalization": "none",
            "aggregation": "weighted_arithmetic_mean",
        }

    def compose(
        self,
        reference_curves: tuple["CurveTable", ...],
        target_curves: tuple["CurveTable", ...],
    ) -> ComposedAlignedPair:
        """Align features and apply the same normalized weights to both sides."""

        self._validate_curve_collections(reference_curves, target_curves)
        aligner = CurveAligner(self.alignment_strategy)
        aligned_features = tuple(
            aligner.align(reference, target)
            for reference, target in zip(reference_curves, target_curves)
        )
        shared_scales = self._shared_scales(aligned_features)
        reference_matrix = []
        target_matrix = []
        for aligned in aligned_features:
            reference_by_scale = dict(zip(aligned.scales, aligned.reference_values))
            target_by_scale = dict(zip(aligned.scales, aligned.target_values))
            reference_matrix.append(
                [reference_by_scale[scale] for scale in shared_scales]
            )
            target_matrix.append(
                [target_by_scale[scale] for scale in shared_scales]
            )
        reference_values = np.average(
            np.asarray(reference_matrix, dtype=float),
            axis=0,
            weights=np.asarray(self.normalized_weights, dtype=float),
        )
        target_values = np.average(
            np.asarray(target_matrix, dtype=float),
            axis=0,
            weights=np.asarray(self.normalized_weights, dtype=float),
        )

        composer_config_hash = stable_config_hash(dict(self.config()))
        reference_source_ids = tuple(
            aligned.reference_curve_id for aligned in aligned_features
        )
        target_source_ids = tuple(
            aligned.target_curve_id for aligned in aligned_features
        )
        observation_keys = tuple(
            CompositeObservationKey(self.output_feature_slot, scale)
            for scale in shared_scales
        )
        metadata = {
            "shared_scales": list(shared_scales),
            "feature_weights": {
                feature_slot: normalized_weight
                for feature_slot, normalized_weight in zip(
                    self.feature_slots,
                    self.normalized_weights,
                )
            },
            "normalization": "none",
        }
        reference = ComposedCurve(
            curve_id=stable_config_hash({
                "composer_config_hash": composer_config_hash,
                "role": "reference",
                "source_curve_ids": list(reference_source_ids),
            }),
            composer_name=self.name,
            observation_keys=observation_keys,
            values=tuple(float(value) for value in reference_values),
            source_curve_ids=reference_source_ids,
            config_hash=composer_config_hash,
            metadata=metadata,
        )
        target = ComposedCurve(
            curve_id=stable_config_hash({
                "composer_config_hash": composer_config_hash,
                "role": "target",
                "source_curve_ids": list(target_source_ids),
            }),
            composer_name=self.name,
            observation_keys=observation_keys,
            values=tuple(float(value) for value in target_values),
            source_curve_ids=target_source_ids,
            config_hash=composer_config_hash,
            metadata=metadata,
        )
        return ComposedAlignedPair(reference=reference, target=target)

    def _validate_curve_collections(
        self,
        reference_curves: tuple["CurveTable", ...],
        target_curves: tuple["CurveTable", ...],
    ) -> None:
        for field_name, curves in (
            ("reference_curves", reference_curves),
            ("target_curves", target_curves),
        ):
            if not isinstance(curves, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(not isinstance(curve, CurveTable) for curve in curves):
                raise TypeError(f"{field_name} must contain CurveTable objects")
        if len(reference_curves) != len(self.feature_slots):
            raise ValueError("reference_curves must match feature_slots length")
        if len(target_curves) != len(self.feature_slots):
            raise ValueError("target_curves must match feature_slots length")

    def _shared_scales(
        self,
        aligned_features: tuple["AlignedCurvePair", ...],
    ) -> tuple[int, ...]:
        scale_sets = [set(aligned.scales) for aligned in aligned_features]
        if self.alignment_strategy == "exact":
            if any(scales != scale_sets[0] for scales in scale_sets[1:]):
                raise ValueError(
                    "exact composition requires identical scales across features"
                )
            return tuple(sorted(scale_sets[0]))
        shared = set.intersection(*scale_sets)
        if not shared:
            raise ValueError("intersection composition found no common feature scales")
        return tuple(sorted(shared))


@dataclass(frozen=True)
class CorrelationMatrix:
    """Represent one symmetric all-pairs coefficient matrix."""

    statistic: str
    component: str
    alignment_strategy: str
    config_hash: str
    curve_ids: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        for field_name in ("statistic", "component", "config_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.alignment_strategy not in ALIGNMENT_STRATEGIES:
            raise ValueError(
                f"alignment_strategy must be one of {ALIGNMENT_STRATEGIES}"
            )
        if not isinstance(self.curve_ids, tuple) or len(self.curve_ids) < 2:
            raise ValueError("curve_ids must contain at least two curves")
        if any(not isinstance(curve_id, str) or not curve_id for curve_id in self.curve_ids):
            raise ValueError("curve_ids must contain non-empty strings")
        if len(self.curve_ids) != len(set(self.curve_ids)):
            raise ValueError("curve_ids must be unique")
        size = len(self.curve_ids)
        if (
            not isinstance(self.values, tuple)
            or len(self.values) != size
            or any(not isinstance(row, tuple) or len(row) != size for row in self.values)
        ):
            raise ValueError("values must be a square tuple matrix")
        array = np.asarray(self.values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("values must be finite")
        if not np.allclose(array, array.T, rtol=0.0, atol=0.0):
            raise ValueError("values must be exactly symmetric")
        if not np.allclose(np.diag(array), 1.0, rtol=0.0, atol=0.0):
            raise ValueError("correlation matrix diagonal must equal one")

    def to_frame(self) -> pd.DataFrame:
        """Return a labeled square DataFrame in the requested curve order."""

        return pd.DataFrame(
            np.asarray(self.values, dtype=float),
            index=pd.Index(self.curve_ids, name="curve_id"),
            columns=pd.Index(self.curve_ids, name="curve_id"),
        )


class CorrelationMatrixBuilder:
    """Build a complete symmetric view from all-pairs statistic results."""

    def __init__(self, *, statistic: str, component: str) -> None:
        for field_name, value in (("statistic", statistic), ("component", component)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        self.statistic = statistic
        self.component = component

    def build(
        self,
        results: "StatisticResultTable",
        *,
        curve_ids: tuple[str, ...] | None = None,
    ) -> CorrelationMatrix:
        """Select one coefficient component and require complete all-pairs coverage."""

        if not isinstance(results, StatisticResultTable):
            raise TypeError("results must be a StatisticResultTable")
        selected = tuple(
            row
            for row in results.rows
            if row.statistic == self.statistic and row.component == self.component
        )
        if not selected:
            raise ValueError("no statistic results matched the requested coefficient")

        strategies = {row.alignment_strategy for row in selected}
        config_hashes = {row.config_hash for row in selected}
        if len(strategies) != 1 or len(config_hashes) != 1:
            raise ValueError(
                "correlation matrix rows must share one alignment strategy and config hash"
            )
        if any(row.reference_curve_id == row.target_curve_id for row in selected):
            raise ValueError("correlation matrix input must not contain self-pairs")

        observed_order = tuple(dict.fromkeys(
            curve_id
            for row in selected
            for curve_id in (row.reference_curve_id, row.target_curve_id)
        ))
        ordered_curve_ids = (
            observed_order
            if curve_ids is None
            else self._validate_curve_order(curve_ids, observed_order)
        )

        pair_values: dict[frozenset[str], float] = {}
        for row in selected:
            pair_key = frozenset((row.reference_curve_id, row.target_curve_id))
            if pair_key in pair_values:
                raise ValueError("correlation matrix input contains a duplicate unordered pair")
            pair_values[pair_key] = float(row.value)

        expected_pairs = {
            frozenset(pair)
            for pair in combinations(ordered_curve_ids, 2)
        }
        actual_pairs = set(pair_values)
        if actual_pairs != expected_pairs:
            missing = len(expected_pairs - actual_pairs)
            extra = len(actual_pairs - expected_pairs)
            raise ValueError(
                "correlation matrix requires complete all-pairs coverage: "
                f"missing={missing}, extra={extra}"
            )

        size = len(ordered_curve_ids)
        matrix = np.eye(size, dtype=float)
        for row_index, reference_curve_id in enumerate(ordered_curve_ids):
            for column_index in range(row_index + 1, size):
                target_curve_id = ordered_curve_ids[column_index]
                value = pair_values[frozenset((reference_curve_id, target_curve_id))]
                matrix[row_index, column_index] = value
                matrix[column_index, row_index] = value
        return CorrelationMatrix(
            statistic=self.statistic,
            component=self.component,
            alignment_strategy=next(iter(strategies)),
            config_hash=next(iter(config_hashes)),
            curve_ids=ordered_curve_ids,
            values=tuple(tuple(float(value) for value in row) for row in matrix),
        )

    @staticmethod
    def _validate_curve_order(
        curve_ids: tuple[str, ...],
        observed_curve_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not isinstance(curve_ids, tuple) or len(curve_ids) < 2:
            raise ValueError("curve_ids must be a tuple containing at least two curves")
        if any(not isinstance(curve_id, str) or not curve_id for curve_id in curve_ids):
            raise ValueError("curve_ids must contain non-empty strings")
        if len(curve_ids) != len(set(curve_ids)):
            raise ValueError("curve_ids must be unique")
        if set(curve_ids) != set(observed_curve_ids):
            raise ValueError("curve_ids must exactly match the observed curves")
        return curve_ids


@dataclass(frozen=True)
class StatisticObservation:
    """Represent one named scalar emitted by a statistic adapter."""

    component: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("component must be a non-empty string")
        if not np.isscalar(self.value) or not np.isfinite(self.value):
            raise ValueError("value must be a finite scalar")


@dataclass(frozen=True)
class StatisticResult:
    """Hold one statistic adapter result without curve or path concerns."""

    statistic: str
    observations: tuple[StatisticObservation, ...]
    n_points: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.statistic, str) or not self.statistic.strip():
            raise ValueError("statistic must be a non-empty string")
        if not isinstance(self.observations, tuple) or not self.observations:
            raise ValueError("observations must be a non-empty tuple")
        if any(
            not isinstance(observation, StatisticObservation)
            for observation in self.observations
        ):
            raise TypeError("observations must contain StatisticObservation objects")
        components = [observation.component for observation in self.observations]
        if len(components) != len(set(components)):
            raise ValueError("statistic result contains duplicate components")
        if isinstance(self.n_points, bool) or not isinstance(self.n_points, int):
            raise TypeError("n_points must be an integer")
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) or not key for key in self.metadata):
            raise ValueError("metadata keys must be non-empty strings")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def value_for(self, component: str) -> float:
        """Return one scalar by component name."""

        matches = [
            observation.value
            for observation in self.observations
            if observation.component == component
        ]
        if not matches:
            raise KeyError(f"statistic result has no component: {component}")
        return float(matches[0])


@dataclass(frozen=True)
class StatisticResultRow:
    """Represent one long-form component of a curve comparison."""

    reference_curve_id: str
    target_curve_id: str
    alignment_strategy: str
    statistic: str
    component: str
    value: float
    n_points: int
    config_hash: str
    reused: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "reference_curve_id",
            "target_curve_id",
            "statistic",
            "component",
            "config_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.alignment_strategy not in ALIGNMENT_STRATEGIES:
            raise ValueError(
                f"alignment_strategy must be one of {ALIGNMENT_STRATEGIES}"
            )
        if not np.isscalar(self.value) or not np.isfinite(self.value):
            raise ValueError("value must be a finite scalar")
        if isinstance(self.n_points, bool) or not isinstance(self.n_points, int):
            raise TypeError("n_points must be an integer")
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be a boolean")

    @property
    def identity(self) -> tuple[str, ...]:
        """Return the unique long-form result identity."""

        return (
            self.reference_curve_id,
            self.target_curve_id,
            self.alignment_strategy,
            self.statistic,
            self.component,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return one canonical flat result row."""

        return {
            "reference_curve_id": self.reference_curve_id,
            "target_curve_id": self.target_curve_id,
            "alignment_strategy": self.alignment_strategy,
            "statistic": self.statistic,
            "component": self.component,
            "value": float(self.value),
            "n_points": self.n_points,
            "config_hash": self.config_hash,
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StatisticResultRow":
        """Construct one row from CSV-compatible values."""

        expected = set(STATISTIC_RESULT_COLUMNS)
        provided = set(data)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(
                f"Invalid statistic result fields: missing={missing}, extra={extra}"
            )
        return cls(
            reference_curve_id=str(data["reference_curve_id"]),
            target_curve_id=str(data["target_curve_id"]),
            alignment_strategy=str(data["alignment_strategy"]),
            statistic=str(data["statistic"]),
            component=str(data["component"]),
            value=float(data["value"]),
            n_points=_required_integer(data["n_points"], "n_points"),
            config_hash=str(data["config_hash"]),
            reused=_boolean(data["reused"], "reused"),
        )


@dataclass(frozen=True)
class StatisticResultTable:
    """Collect long-form results across curve pairs and statistics."""

    rows: tuple[StatisticResultRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        if any(not isinstance(row, StatisticResultRow) for row in self.rows):
            raise TypeError("rows must contain StatisticResultRow objects")
        identities = [row.identity for row in self.rows]
        if len(identities) != len(set(identities)):
            raise ValueError("statistic result table contains duplicate components")

    @classmethod
    def from_result(
        cls,
        aligned: "AlignedCurvePair",
        result: StatisticResult,
        *,
        config_hash: str,
        reused: bool = False,
    ) -> "StatisticResultTable":
        """Map one adapter result and alignment provenance to long form."""

        if not isinstance(aligned, AlignedCurvePair):
            raise TypeError("aligned must be an AlignedCurvePair")
        if not isinstance(result, StatisticResult):
            raise TypeError("result must be a StatisticResult")
        if result.n_points != len(aligned.scales):
            raise ValueError("statistic n_points does not match aligned scales")
        return cls(rows=tuple(
            StatisticResultRow(
                reference_curve_id=aligned.reference_curve_id,
                target_curve_id=aligned.target_curve_id,
                alignment_strategy=aligned.strategy,
                statistic=result.statistic,
                component=observation.component,
                value=float(observation.value),
                n_points=result.n_points,
                config_hash=config_hash,
                reused=reused,
            )
            for observation in result.observations
        ))

    @classmethod
    def from_composed_pair(
        cls,
        pair: ComposedAlignedPair,
        result: StatisticResult,
        *,
        alignment_strategy: str,
        config_hash: str,
        reused: bool = False,
    ) -> "StatisticResultTable":
        """Map one composed-vector result to the common long-form schema."""

        if not isinstance(pair, ComposedAlignedPair):
            raise TypeError("pair must be a ComposedAlignedPair")
        if not isinstance(result, StatisticResult):
            raise TypeError("result must be a StatisticResult")
        if alignment_strategy not in ALIGNMENT_STRATEGIES:
            raise ValueError(
                f"alignment_strategy must be one of {ALIGNMENT_STRATEGIES}"
            )
        if result.n_points != len(pair.observation_keys):
            raise ValueError(
                "statistic n_points does not match composed observation keys"
            )
        return cls(rows=tuple(
            StatisticResultRow(
                reference_curve_id=pair.reference.curve_id,
                target_curve_id=pair.target.curve_id,
                alignment_strategy=alignment_strategy,
                statistic=result.statistic,
                component=observation.component,
                value=float(observation.value),
                n_points=result.n_points,
                config_hash=config_hash,
                reused=reused,
            )
            for observation in result.observations
        ))

    @classmethod
    def combine(cls, *tables: "StatisticResultTable") -> "StatisticResultTable":
        """Combine result tables while retaining duplicate validation."""

        if any(not isinstance(table, StatisticResultTable) for table in tables):
            raise TypeError("tables must contain StatisticResultTable objects")
        return cls(rows=tuple(row for table in tables for row in table.rows))

    def to_frame(self) -> pd.DataFrame:
        """Return the canonical long-form statistic DataFrame."""

        return pd.DataFrame(
            [row.to_dict() for row in self.rows],
            columns=STATISTIC_RESULT_COLUMNS,
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "StatisticResultTable":
        """Validate and construct a result table from a DataFrame."""

        if tuple(frame.columns) != STATISTIC_RESULT_COLUMNS:
            raise ValueError(
                "statistic result columns must exactly match canonical order"
            )
        return cls(rows=tuple(
            StatisticResultRow.from_dict(record)
            for record in frame.to_dict(orient="records")
        ))

    def write_csv(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write results without silently replacing an existing table."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Statistic result table already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(destination, index=False, encoding="utf-8")
        return destination

    @classmethod
    def read_csv(cls, path: str | Path) -> "StatisticResultTable":
        """Read and validate a canonical statistic result CSV."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Statistic result table not found: {source}")
        return cls.from_frame(pd.read_csv(source, float_precision="round_trip"))


@runtime_checkable
class PairwiseStatistic(Protocol):
    """Compute one statistic over two already aligned value vectors."""

    name: str

    def config(self) -> Mapping[str, Any]:
        """Return the effective numerical configuration."""

    def compute(
        self,
        reference_values: np.ndarray,
        target_values: np.ndarray,
    ) -> StatisticResult:
        """Compute one result from aligned reference and target values."""


class SpearmanStatistic:
    """Adapt the existing Spearman helper to PairwiseStatistic."""

    name = "spearman"

    def config(self) -> Mapping[str, Any]:
        """Return the existing two-sided SciPy Spearman semantics."""

        return {
            "implementation": "cluster_anlys.spearman.compute_spearman_stats",
            "alternative": "two-sided",
        }

    def compute(
        self,
        reference_values: np.ndarray,
        target_values: np.ndarray,
    ) -> StatisticResult:
        """Return rho and its two-sided p-value."""

        reference, target = _coerce_pair(
            reference_values,
            target_values,
            statistic_name=self.name,
            minimum_points=2,
        )
        _require_nonconstant_pair(self.name, reference, target)
        from .spearman import compute_spearman_stats

        legacy = compute_spearman_stats(reference, target)
        _require_finite_result(self.name, rho=legacy.rho, p_value=legacy.pvalue)
        return StatisticResult(
            statistic=self.name,
            observations=(
                StatisticObservation("rho", legacy.rho),
                StatisticObservation("p_value", legacy.pvalue),
            ),
            n_points=legacy.n_obs,
            metadata={"alternative": "two-sided"},
        )


class PearsonStatistic:
    """Adapt the existing Pearson helper to PairwiseStatistic."""

    name = "pearson"

    def config(self) -> Mapping[str, Any]:
        """Return the existing two-sided SciPy Pearson semantics."""

        return {
            "implementation": "cluster_anlys.pearson.compute_pearson_stats",
            "alternative": "two-sided",
        }

    def compute(
        self,
        reference_values: np.ndarray,
        target_values: np.ndarray,
    ) -> StatisticResult:
        """Return r and its two-sided p-value."""

        reference, target = _coerce_pair(
            reference_values,
            target_values,
            statistic_name=self.name,
            minimum_points=2,
        )
        _require_nonconstant_pair(self.name, reference, target)
        from .pearson import compute_pearson_stats

        legacy = compute_pearson_stats(reference, target)
        _require_finite_result(self.name, r=legacy.r, p_value=legacy.pvalue)
        return StatisticResult(
            statistic=self.name,
            observations=(
                StatisticObservation("r", legacy.r),
                StatisticObservation("p_value", legacy.pvalue),
            ),
            n_points=legacy.n_obs,
            metadata={"alternative": "two-sided"},
        )


class MAEStatistic:
    """Compute the mean absolute error between aligned curves."""

    name = "mae"

    def config(self) -> Mapping[str, Any]:
        """Return the direct equal-weight pointwise MAE semantics."""

        return {
            "definition": "mean_absolute_target_minus_reference",
            "point_weighting": "equal",
        }

    def compute(
        self,
        reference_values: np.ndarray,
        target_values: np.ndarray,
    ) -> StatisticResult:
        """Return equal-weight mean absolute error."""

        reference, target = _coerce_pair(
            reference_values,
            target_values,
            statistic_name=self.name,
            minimum_points=1,
        )
        value = float(np.mean(np.abs(target - reference)))
        return StatisticResult(
            statistic=self.name,
            observations=(StatisticObservation("mae", value),),
            n_points=int(reference.size),
            metadata={"point_weighting": "equal"},
        )


class R2Statistic:
    """Compute baseline-as-reference R-squared from existing notebooks."""

    name = "r2"

    def config(self) -> Mapping[str, Any]:
        """Declare the asymmetric reference/target numerical direction."""

        return {
            "definition": "one_minus_target_residual_over_reference_total",
            "reference_role": "true_values",
            "target_role": "predicted_values",
        }

    def compute(
        self,
        reference_values: np.ndarray,
        target_values: np.ndarray,
    ) -> StatisticResult:
        """Return R-squared with total variance defined by the reference."""

        reference, target = _coerce_pair(
            reference_values,
            target_values,
            statistic_name=self.name,
            minimum_points=2,
        )
        residual_sum = float(np.sum((target - reference) ** 2))
        total_sum = float(np.sum((reference - np.mean(reference)) ** 2))
        if total_sum == 0.0:
            raise ValueError("r2 is undefined for a constant reference curve")
        value = float(1.0 - residual_sum / total_sum)
        return StatisticResult(
            statistic=self.name,
            observations=(StatisticObservation("r2", value),),
            n_points=int(reference.size),
            metadata={
                "reference_role": "true_values",
                "target_role": "predicted_values",
            },
        )


@dataclass(frozen=True)
class StatisticsPipelineRun:
    """Return selected curves, alignments, and long-form statistics."""

    curves: "CurveTable"
    alignments: tuple["AlignedCurvePair", ...]
    results: StatisticResultTable


class StatisticsPipeline:
    """Run arbitrary pairwise statistics over explicitly selected curves."""

    def __init__(
        self,
        *,
        statistics: tuple[PairwiseStatistic, ...],
        alignment_strategy: str = "exact",
        pair_mode: str = "all_pairs",
        reference_index: int = 0,
    ) -> None:
        if not isinstance(statistics, tuple) or not statistics:
            raise ValueError("statistics must be a non-empty tuple")
        if any(not isinstance(statistic, PairwiseStatistic) for statistic in statistics):
            raise TypeError("statistics must satisfy PairwiseStatistic")
        statistic_names = [statistic.name for statistic in statistics]
        if len(statistic_names) != len(set(statistic_names)):
            raise ValueError("statistics must have unique names")
        if alignment_strategy not in ALIGNMENT_STRATEGIES:
            raise ValueError(
                f"alignment_strategy must be one of {ALIGNMENT_STRATEGIES}"
            )
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}")
        if isinstance(reference_index, bool) or not isinstance(reference_index, int):
            raise TypeError("reference_index must be an integer")
        if reference_index < 0:
            raise ValueError("reference_index must be non-negative")
        self.statistics = statistics
        self.alignment_strategy = alignment_strategy
        self.pair_mode = pair_mode
        self.reference_index = reference_index

    def config(self) -> dict[str, Any]:
        """Return the deterministic in-memory scheduling configuration."""

        return {
            "schema_version": 1,
            "alignment_strategy": self.alignment_strategy,
            "pair_mode": self.pair_mode,
            "reference_index": self.reference_index,
            "statistics": [
                {"name": statistic.name, "parameters": dict(statistic.config())}
                for statistic in self.statistics
            ],
        }

    def run(
        self,
        metric_results: MetricResultTable,
        queries: tuple["CurveQuery", ...],
    ) -> StatisticsPipelineRun:
        """Select curves, generate explicit pairs, align, and compute."""

        if not isinstance(metric_results, MetricResultTable):
            raise TypeError("metric_results must be a MetricResultTable")
        if not isinstance(queries, tuple) or len(queries) < 2:
            raise ValueError("queries must contain at least two CurveQuery objects")
        if any(not isinstance(query, CurveQuery) for query in queries):
            raise TypeError("queries must contain CurveQuery objects")
        if self.pair_mode == "reference_only" and self.reference_index >= len(queries):
            raise ValueError("reference_index is outside the query tuple")

        selected = tuple(query.select(metric_results) for query in queries)
        curve_ids = [table.curve_ids[0] for table in selected]
        if len(curve_ids) != len(set(curve_ids)):
            raise ValueError("queries must select distinct curves")
        pair_indices = self._pair_indices(len(selected))
        aligner = CurveAligner(self.alignment_strategy)
        alignments = tuple(
            aligner.align(selected[reference_index], selected[target_index])
            for reference_index, target_index in pair_indices
        )

        result_tables = []
        for aligned in alignments:
            for statistic in self.statistics:
                result = statistic.compute(
                    aligned.reference_array,
                    aligned.target_array,
                )
                if result.statistic != statistic.name:
                    raise ValueError(
                        "statistic result name does not match the adapter name"
                    )
                config_hash = stable_config_hash({
                    "schema_version": 1,
                    "alignment_strategy": self.alignment_strategy,
                    "statistic": {
                        "name": statistic.name,
                        "parameters": dict(statistic.config()),
                    },
                })
                result_tables.append(
                    StatisticResultTable.from_result(
                        aligned,
                        result,
                        config_hash=config_hash,
                    )
                )
        return StatisticsPipelineRun(
            curves=CurveTable.combine(*selected),
            alignments=alignments,
            results=StatisticResultTable.combine(*result_tables),
        )

    def _pair_indices(self, n_curves: int) -> tuple[tuple[int, int], ...]:
        if self.pair_mode == "all_pairs":
            return tuple(combinations(range(n_curves), 2))
        return tuple(
            (self.reference_index, target_index)
            for target_index in range(n_curves)
            if target_index != self.reference_index
        )


@dataclass(frozen=True)
class ComposedStatisticsPipelineRun:
    """Return source curves, one composed pair, and its statistics."""

    reference_curves: "CurveTable"
    target_curves: "CurveTable"
    composed_pair: ComposedAlignedPair
    results: StatisticResultTable


class ComposedStatisticsPipeline:
    """Run arbitrary statistics after one explicit curve composition step."""

    def __init__(
        self,
        *,
        composer: CurveComposer,
        statistics: tuple[PairwiseStatistic, ...],
    ) -> None:
        if not isinstance(composer, CurveComposer):
            raise TypeError("composer must satisfy CurveComposer")
        if not isinstance(statistics, tuple) or not statistics:
            raise ValueError("statistics must be a non-empty tuple")
        if any(not isinstance(statistic, PairwiseStatistic) for statistic in statistics):
            raise TypeError("statistics must satisfy PairwiseStatistic")
        statistic_names = [statistic.name for statistic in statistics]
        if len(statistic_names) != len(set(statistic_names)):
            raise ValueError("statistics must have unique names")
        self.composer = composer
        self.statistics = statistics

    def config(self) -> dict[str, Any]:
        """Return deterministic composer and statistic configuration."""

        return {
            "schema_version": 1,
            "composer": {
                "name": self.composer.name,
                "parameters": dict(self.composer.config()),
            },
            "statistics": [
                {"name": statistic.name, "parameters": dict(statistic.config())}
                for statistic in self.statistics
            ],
        }

    def run(
        self,
        metric_results: MetricResultTable,
        *,
        reference_queries: tuple["CurveQuery", ...],
        target_queries: tuple["CurveQuery", ...],
    ) -> ComposedStatisticsPipelineRun:
        """Select paired feature curves, compose once, and compute statistics."""

        if not isinstance(metric_results, MetricResultTable):
            raise TypeError("metric_results must be a MetricResultTable")
        for field_name, queries in (
            ("reference_queries", reference_queries),
            ("target_queries", target_queries),
        ):
            if not isinstance(queries, tuple) or not queries:
                raise ValueError(f"{field_name} must be a non-empty tuple")
            if any(not isinstance(query, CurveQuery) for query in queries):
                raise TypeError(f"{field_name} must contain CurveQuery objects")
        if len(reference_queries) != len(target_queries):
            raise ValueError("reference_queries and target_queries must have equal length")

        reference_tables = tuple(
            query.select(metric_results) for query in reference_queries
        )
        target_tables = tuple(
            query.select(metric_results) for query in target_queries
        )
        pair = self.composer.compose(reference_tables, target_tables)
        composer_config = dict(self.composer.config())
        alignment_strategy = str(composer_config.get("alignment_strategy", ""))
        if alignment_strategy not in ALIGNMENT_STRATEGIES:
            raise ValueError(
                "composer config must declare a supported alignment_strategy"
            )

        result_tables = []
        for statistic in self.statistics:
            result = statistic.compute(pair.reference_array, pair.target_array)
            if result.statistic != statistic.name:
                raise ValueError(
                    "statistic result name does not match the adapter name"
                )
            config_hash = stable_config_hash({
                "schema_version": 1,
                "composer": {
                    "name": self.composer.name,
                    "parameters": composer_config,
                    "config_hash": pair.reference.config_hash,
                },
                "statistic": {
                    "name": statistic.name,
                    "parameters": dict(statistic.config()),
                },
            })
            result_tables.append(StatisticResultTable.from_composed_pair(
                pair,
                result,
                alignment_strategy=alignment_strategy,
                config_hash=config_hash,
            ))
        return ComposedStatisticsPipelineRun(
            reference_curves=CurveTable.combine(*reference_tables),
            target_curves=CurveTable.combine(*target_tables),
            composed_pair=pair,
            results=StatisticResultTable.combine(*result_tables),
        )


@dataclass(frozen=True)
class CurveQuery:
    """Select one unambiguous curve from long-form metric results."""

    analysis_id: str
    model_name: str
    source: str
    metric: str
    component: str
    aggregation: str
    scope: str = "partition"
    seed_type: str | None = None
    seed: int | None = None
    space_name: str | None = None
    partition_method: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "analysis_id",
            "model_name",
            "source",
            "metric",
            "component",
            "aggregation",
            "scope",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (self.seed_type is None) != (self.seed is None):
            raise ValueError("seed_type and seed must either both be set or both be None")
        if self.seed_type is not None:
            if not isinstance(self.seed_type, str) or not self.seed_type.strip():
                raise ValueError("seed_type must be a non-empty string when seed is set")
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer")
            if self.seed < 0:
                raise ValueError("seed must be non-negative")
        for field_name in ("space_name", "partition_method"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string when set")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical query mapping used for stable identity."""

        return {
            "analysis_id": self.analysis_id,
            "model_name": self.model_name,
            "source": self.source,
            "seed_type": self.seed_type,
            "seed": self.seed,
            "metric": self.metric,
            "component": self.component,
            "aggregation": self.aggregation,
            "scope": self.scope,
            "space_name": self.space_name,
            "partition_method": self.partition_method,
        }

    @property
    def curve_id(self) -> str:
        """Return a deterministic identity for the requested curve."""

        return stable_config_hash(self.to_dict())

    def select(self, results: MetricResultTable) -> "CurveTable":
        """Select, validate, and sort one curve from metric results."""

        if not isinstance(results, MetricResultTable):
            raise TypeError("results must be a MetricResultTable")
        selected = tuple(
            row
            for row in results.rows
            if row.task.analysis_id == self.analysis_id
            and row.task.model_name == self.model_name
            and row.task.source == self.source
            and row.task.seed_type == self.seed_type
            and row.task.seed == self.seed
            and row.metric == self.metric
            and row.component == self.component
            and row.aggregation == self.aggregation
            and row.scope == self.scope
            and (self.space_name is None or row.task.space_name == self.space_name)
            and (
                self.partition_method is None
                or row.partition_method == self.partition_method
            )
        )
        if not selected:
            raise ValueError("curve query matched no metric observations")

        spaces = {row.task.space_name for row in selected}
        methods = {row.partition_method for row in selected}
        if len(spaces) != 1 or len(methods) != 1:
            raise ValueError(
                "curve query is ambiguous across representation spaces or partition methods"
            )
        scales = [row.task.scale for row in selected]
        if len(scales) != len(set(scales)):
            raise ValueError("curve query matched duplicate scales")

        resolved_query = replace(
            self,
            space_name=next(iter(spaces)),
            partition_method=next(iter(methods)),
        )
        curve_rows = tuple(
            CurveRow(
                curve_id=resolved_query.curve_id,
                analysis_id=row.task.analysis_id,
                model_name=row.task.model_name,
                space_name=row.task.space_name,
                source=row.task.source,
                seed_type=row.task.seed_type,
                seed=row.task.seed,
                partition_method=row.partition_method,
                metric=row.metric,
                component=row.component,
                aggregation=row.aggregation,
                scope=row.scope,
                scale=row.task.scale,
                value=float(row.value),
                config_hash=row.config_hash,
                reused=row.reused,
            )
            for row in sorted(selected, key=lambda item: item.task.scale)
        )
        return CurveTable(rows=curve_rows)


@dataclass(frozen=True)
class CurveRow:
    """Represent one scale-value observation in a selected curve."""

    curve_id: str
    analysis_id: str
    model_name: str
    space_name: str
    source: str
    seed_type: str | None
    seed: int | None
    partition_method: str
    metric: str
    component: str
    aggregation: str
    scope: str
    scale: int
    value: float
    config_hash: str
    reused: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "curve_id",
            "analysis_id",
            "model_name",
            "space_name",
            "source",
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
        if (self.seed_type is None) != (self.seed is None):
            raise ValueError("seed_type and seed must either both be set or both be None")
        if self.seed_type is not None:
            if not isinstance(self.seed_type, str) or not self.seed_type.strip():
                raise ValueError("seed_type must be a non-empty string when seed is set")
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("seed must be an integer")
            if self.seed < 0:
                raise ValueError("seed must be non-negative")
        if isinstance(self.scale, bool) or not isinstance(self.scale, int):
            raise TypeError("scale must be an integer")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if not np.isscalar(self.value) or not np.isfinite(self.value):
            raise ValueError("value must be a finite scalar")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be a boolean")

    @property
    def identity(self) -> tuple[str, int]:
        """Return the unique point identity within a curve table."""

        return self.curve_id, self.scale

    def to_dict(self) -> dict[str, Any]:
        """Return one canonical flat curve row."""

        return {
            "curve_id": self.curve_id,
            "analysis_id": self.analysis_id,
            "model_name": self.model_name,
            "space_name": self.space_name,
            "source": self.source,
            "seed_type": self.seed_type,
            "seed": self.seed,
            "partition_method": self.partition_method,
            "metric": self.metric,
            "component": self.component,
            "aggregation": self.aggregation,
            "scope": self.scope,
            "scale": self.scale,
            "value": float(self.value),
            "config_hash": self.config_hash,
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CurveRow":
        """Construct one curve row from CSV-compatible values."""

        expected = set(CURVE_TABLE_COLUMNS)
        provided = set(data)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(f"Invalid curve row fields: missing={missing}, extra={extra}")
        seed_type = None if pd.isna(data["seed_type"]) else str(data["seed_type"])
        seed = _optional_integer(data["seed"], "seed")
        return cls(
            curve_id=str(data["curve_id"]),
            analysis_id=str(data["analysis_id"]),
            model_name=str(data["model_name"]),
            space_name=str(data["space_name"]),
            source=str(data["source"]),
            seed_type=seed_type,
            seed=seed,
            partition_method=str(data["partition_method"]),
            metric=str(data["metric"]),
            component=str(data["component"]),
            aggregation=str(data["aggregation"]),
            scope=str(data["scope"]),
            scale=_required_integer(data["scale"], "scale"),
            value=float(data["value"]),
            config_hash=str(data["config_hash"]),
            reused=_boolean(data["reused"], "reused"),
        )


@dataclass(frozen=True)
class CurveTable:
    """Collect one or more selected long-form curves."""

    rows: tuple[CurveRow, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")
        if any(not isinstance(row, CurveRow) for row in self.rows):
            raise TypeError("rows must contain CurveRow objects")
        identities = [row.identity for row in self.rows]
        if len(identities) != len(set(identities)):
            raise ValueError("curve table contains duplicate curve-scale identities")
        semantics_by_curve: dict[str, set[tuple[Any, ...]]] = {}
        for row in self.rows:
            semantics_by_curve.setdefault(row.curve_id, set()).add(
                (
                    row.analysis_id,
                    row.model_name,
                    row.space_name,
                    row.source,
                    row.seed_type,
                    row.seed,
                    row.partition_method,
                    row.metric,
                    row.component,
                    row.aggregation,
                    row.scope,
                )
            )
        if any(len(signatures) != 1 for signatures in semantics_by_curve.values()):
            raise ValueError("one curve_id cannot describe multiple curve semantics")

    @property
    def curve_ids(self) -> tuple[str, ...]:
        """Return curve identities in first-observed order."""

        return tuple(dict.fromkeys(row.curve_id for row in self.rows))

    def to_frame(self) -> pd.DataFrame:
        """Return the canonical long-form curve DataFrame."""

        frame = pd.DataFrame(
            [row.to_dict() for row in self.rows],
            columns=CURVE_TABLE_COLUMNS,
        )
        if not frame.empty:
            frame["seed"] = pd.array(frame["seed"], dtype="Int64")
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "CurveTable":
        """Validate and construct a curve table from a DataFrame."""

        if tuple(frame.columns) != CURVE_TABLE_COLUMNS:
            raise ValueError("curve table columns must exactly match canonical order")
        return cls(
            rows=tuple(
                CurveRow.from_dict(record)
                for record in frame.to_dict(orient="records")
            )
        )

    @classmethod
    def combine(cls, *tables: "CurveTable") -> "CurveTable":
        """Combine curve tables while preserving duplicate validation."""

        if any(not isinstance(table, CurveTable) for table in tables):
            raise TypeError("tables must contain CurveTable objects")
        return cls(rows=tuple(row for table in tables for row in table.rows))

    def write_csv(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write curves without silently replacing an existing table."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Curve table already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(destination, index=False, encoding="utf-8")
        return destination

    @classmethod
    def read_csv(cls, path: str | Path) -> "CurveTable":
        """Read and validate a canonical curve table CSV."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Curve table not found: {source}")
        return cls.from_frame(pd.read_csv(source))


@dataclass(frozen=True)
class AlignedCurvePair:
    """Hold two curves evaluated at the same ordered scales."""

    reference_curve_id: str
    target_curve_id: str
    scales: tuple[int, ...]
    reference_values: tuple[float, ...]
    target_values: tuple[float, ...]
    strategy: str

    def __post_init__(self) -> None:
        for field_name in ("reference_curve_id", "target_curve_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.strategy not in ALIGNMENT_STRATEGIES:
            raise ValueError(f"strategy must be one of {ALIGNMENT_STRATEGIES}")
        if not self.scales:
            raise ValueError("aligned curves must contain at least one scale")
        if len(self.scales) != len(set(self.scales)):
            raise ValueError("aligned scales must be unique")
        if tuple(sorted(self.scales)) != self.scales:
            raise ValueError("aligned scales must be sorted")
        if not (
            len(self.scales)
            == len(self.reference_values)
            == len(self.target_values)
        ):
            raise ValueError("aligned scales and values must have equal lengths")
        for values in (self.reference_values, self.target_values):
            if any(not np.isfinite(value) for value in values):
                raise ValueError("aligned values must be finite")

    @property
    def reference_array(self) -> np.ndarray:
        """Return reference values as a floating NumPy array."""

        return np.asarray(self.reference_values, dtype=float)

    @property
    def target_array(self) -> np.ndarray:
        """Return target values as a floating NumPy array."""

        return np.asarray(self.target_values, dtype=float)

    def to_frame(self) -> pd.DataFrame:
        """Return a readable wide-form aligned table."""

        return pd.DataFrame(
            {
                "scale": self.scales,
                "reference_value": self.reference_values,
                "target_value": self.target_values,
            }
        )


class CurveAligner:
    """Align two single-curve tables using an explicit scale policy."""

    def __init__(self, strategy: str = "exact") -> None:
        if strategy not in ALIGNMENT_STRATEGIES:
            raise ValueError(f"strategy must be one of {ALIGNMENT_STRATEGIES}")
        self.strategy = strategy

    def align(
        self,
        reference: CurveTable,
        target: CurveTable,
    ) -> AlignedCurvePair:
        """Return two value vectors evaluated at the selected common scales."""

        reference_rows = self._single_curve_rows(reference, "reference")
        target_rows = self._single_curve_rows(target, "target")
        reference_by_scale = {row.scale: float(row.value) for row in reference_rows}
        target_by_scale = {row.scale: float(row.value) for row in target_rows}
        reference_scales = set(reference_by_scale)
        target_scales = set(target_by_scale)

        if self.strategy == "exact":
            if reference_scales != target_scales:
                missing_from_target = sorted(reference_scales - target_scales)
                extra_in_target = sorted(target_scales - reference_scales)
                raise ValueError(
                    "exact alignment requires identical scales: "
                    f"missing_from_target={missing_from_target}, "
                    f"extra_in_target={extra_in_target}"
                )
            aligned_scales = sorted(reference_scales)
        elif self.strategy == "intersection":
            aligned_scales = sorted(reference_scales & target_scales)
            if not aligned_scales:
                raise ValueError("intersection alignment found no common scales")
        else:
            missing_from_target = sorted(reference_scales - target_scales)
            if missing_from_target:
                raise ValueError(
                    "reference alignment requires target to cover reference scales: "
                    f"missing_from_target={missing_from_target}"
                )
            aligned_scales = sorted(reference_scales)

        return AlignedCurvePair(
            reference_curve_id=reference_rows[0].curve_id,
            target_curve_id=target_rows[0].curve_id,
            scales=tuple(aligned_scales),
            reference_values=tuple(reference_by_scale[scale] for scale in aligned_scales),
            target_values=tuple(target_by_scale[scale] for scale in aligned_scales),
            strategy=self.strategy,
        )

    @staticmethod
    def _single_curve_rows(
        table: CurveTable,
        role: str,
    ) -> tuple[CurveRow, ...]:
        if not isinstance(table, CurveTable):
            raise TypeError(f"{role} must be a CurveTable")
        if not table.rows:
            raise ValueError(f"{role} curve table must not be empty")
        if len(table.curve_ids) != 1:
            raise ValueError(f"{role} curve table must contain exactly one curve")
        return table.rows


def _required_integer(value: Any, field_name: str) -> int:
    if pd.isna(value) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    integer = int(value)
    if float(value) != integer:
        raise ValueError(f"{field_name} must be an integer")
    return integer


def _coerce_pair(
    reference_values: Any,
    target_values: Any,
    *,
    statistic_name: str,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    if reference.ndim != 1 or target.ndim != 1:
        raise ValueError(f"{statistic_name} inputs must be one-dimensional")
    if reference.shape != target.shape:
        raise ValueError(
            f"{statistic_name} inputs must share a shape, "
            f"got {reference.shape} and {target.shape}"
        )
    if reference.size < minimum_points:
        raise ValueError(
            f"{statistic_name} requires at least {minimum_points} points, "
            f"got {reference.size}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError(f"{statistic_name} inputs must be finite")
    return reference, target


def _require_finite_result(statistic_name: str, **values: float) -> None:
    undefined = [name for name, value in values.items() if not np.isfinite(value)]
    if undefined:
        raise ValueError(
            f"{statistic_name} is undefined for these inputs: "
            f"non_finite_components={undefined}"
        )


def _require_nonconstant_pair(
    statistic_name: str,
    reference: np.ndarray,
    target: np.ndarray,
) -> None:
    constant_roles = []
    if np.ptp(reference) == 0.0:
        constant_roles.append("reference")
    if np.ptp(target) == 0.0:
        constant_roles.append("target")
    if constant_roles:
        raise ValueError(
            f"{statistic_name} is undefined for constant inputs: "
            f"constant_roles={constant_roles}"
        )


def _optional_integer(value: Any, field_name: str) -> int | None:
    if pd.isna(value):
        return None
    return _required_integer(value, field_name)


def _boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field_name} must be a boolean")
