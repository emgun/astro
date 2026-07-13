from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, field_validator, model_validator

from astro_core.models import (
    AstroModel,
    _integer_input_must_be_int,
    _numeric_scalar_input_must_be_number,
)

SchemaVersion = Literal["1.0"]
MetricValue = float | bool | str | None


class UncertaintyKind(StrEnum):
    ALEATORY = "aleatory"
    EPISTEMIC = "epistemic"


class DistributionKind(StrEnum):
    CONSTANT = "constant"
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"
    EMPIRICAL = "empirical"
    CATEGORICAL = "categorical"


class SamplerKind(StrEnum):
    PSEUDORANDOM = "pseudorandom"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"
    SWEEP = "sweep"
    ENSEMBLE = "ensemble"


class EvaluatorKind(StrEnum):
    AUTHORITATIVE = "authoritative"
    SURROGATE = "surrogate"
    PROGRESSIVE_FIDELITY = "progressive_fidelity"


class OutcomeStatus(StrEnum):
    SUCCESS = "success"
    INVALID_REALIZATION = "invalid_realization"
    EXECUTION_FAILURE = "execution_failure"
    NUMERICAL_FAILURE = "numerical_failure"
    OUT_OF_DOMAIN = "out_of_domain"
    POLICY_REJECTION = "policy_rejection"


class MetricValueKind(StrEnum):
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    EVENT_TIME = "event_time"
    TIME_SERIES_SUMMARY = "time_series_summary"


class RequirementOperator(StrEnum):
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    BETWEEN = "between"
    WITHIN_TOLERANCE = "within_tolerance"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class RetentionPolicy(StrEnum):
    ALL = "all"
    NONE = "none"
    FAILURES = "failures"
    FAILURES_AND_BOUNDARIES = "failures_and_boundaries"
    AUDIT_SAMPLE = "audit_sample"


class CampaignState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    COMPLETED = "completed"


class DistributionSpec(AstroModel):
    kind: DistributionKind
    value: FiniteFloat | str | None = None
    low: FiniteFloat | None = None
    high: FiniteFloat | None = None
    mean: FiniteFloat | None = None
    sigma: FiniteFloat | None = None
    mode: FiniteFloat | None = None
    values: tuple[FiniteFloat, ...] = ()
    labels: tuple[str, ...] = ()
    probabilities: tuple[FiniteFloat, ...] = ()

    @field_validator("value", "low", "high", "mean", "sigma", "mode", mode="before")
    @classmethod
    def numeric_inputs_must_not_be_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("distribution numeric inputs must be numbers, not booleans")
        return value

    @field_validator("values", "probabilities", mode="before")
    @classmethod
    def sequences_must_not_contain_booleans(cls, value: Any) -> Any:
        if isinstance(value, list | tuple) and any(isinstance(item, bool) for item in value):
            raise ValueError("distribution sequences must contain numbers, not booleans")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> DistributionSpec:
        if self.kind is DistributionKind.CONSTANT:
            if self.value is None or isinstance(self.value, str):
                raise ValueError("constant distributions require a numeric value")
        elif self.kind is DistributionKind.UNIFORM:
            if self.low is None or self.high is None or self.low >= self.high:
                raise ValueError("uniform distributions require low < high")
        elif self.kind in {DistributionKind.NORMAL, DistributionKind.LOGNORMAL}:
            if self.mean is None or self.sigma is None or self.sigma <= 0.0:
                raise ValueError("normal distributions require mean and sigma > 0")
        elif self.kind is DistributionKind.TRIANGULAR:
            if (
                self.low is None
                or self.mode is None
                or self.high is None
                or not self.low <= self.mode <= self.high
                or self.low == self.high
            ):
                raise ValueError("triangular distributions require low <= mode <= high")
        elif self.kind is DistributionKind.EMPIRICAL:
            self._validate_weighted_values(len(self.values), "empirical values")
        elif self.kind is DistributionKind.CATEGORICAL:
            if any(not label for label in self.labels) or len(set(self.labels)) != len(self.labels):
                raise ValueError("categorical labels must be non-empty and unique")
            self._validate_weighted_values(len(self.labels), "categorical labels")
        return self

    def _validate_weighted_values(self, count: int, label: str) -> None:
        if count == 0:
            raise ValueError(f"{label} must not be empty")
        if self.probabilities and len(self.probabilities) != count:
            raise ValueError("probabilities must match the number of values")
        if self.probabilities:
            if any(probability < 0.0 for probability in self.probabilities):
                raise ValueError("probabilities must be nonnegative")
            if not np.isclose(sum(self.probabilities), 1.0, rtol=0.0, atol=1.0e-12):
                raise ValueError("probabilities must sum to one")


class UncertainParameter(AstroModel):
    parameter_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    uncertainty_kind: UncertaintyKind
    distribution: DistributionSpec


class CorrelationModel(AstroModel):
    parameter_ids: tuple[str, ...] = Field(min_length=2)
    matrix: tuple[tuple[FiniteFloat, ...], ...]

    @model_validator(mode="after")
    def validate_matrix_shape(self) -> CorrelationModel:
        dimension = len(self.parameter_ids)
        if len(set(self.parameter_ids)) != dimension:
            raise ValueError("correlation parameter ids must be unique")
        if len(self.matrix) != dimension or any(len(row) != dimension for row in self.matrix):
            raise ValueError("correlation matrix dimensions must match parameter ids")
        return self


class ModelVariant(AstroModel):
    variant_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target: str = Field(min_length=1)
    value: str = Field(min_length=1)
    weight: FiniteFloat = Field(gt=0.0, default=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)


class UncertaintyModel(AstroModel):
    parameters: tuple[UncertainParameter, ...] = Field(default_factory=tuple)
    correlations: tuple[CorrelationModel, ...] = Field(default_factory=tuple)
    model_variants: tuple[ModelVariant, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def ids_and_correlations_must_be_valid(self) -> UncertaintyModel:
        parameter_ids = [parameter.parameter_id for parameter in self.parameters]
        variant_ids = [variant.variant_id for variant in self.model_variants]
        if len(set(parameter_ids)) != len(parameter_ids):
            raise ValueError("uncertain parameter ids must be unique")
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("model variant ids must be unique")
        known = set(parameter_ids)
        correlated: set[str] = set()
        for correlation in self.correlations:
            ids = set(correlation.parameter_ids)
            if not ids <= known:
                raise ValueError("correlation references an unknown parameter")
            if correlated & ids:
                raise ValueError("a parameter may appear in only one correlation model")
            correlated.update(ids)
        if not self.parameters and not self.model_variants:
            raise ValueError("uncertainty model must define parameters or model variants")
        return self


class SamplerSpec(AstroModel):
    kind: SamplerKind
    samples: int = Field(gt=0)
    seed: int = 0
    scramble: bool = True
    skip: int = Field(ge=0, default=0)
    sweep_values: dict[str, tuple[FiniteFloat, ...]] = Field(default_factory=dict)

    @field_validator("samples", "seed", "skip", mode="before")
    @classmethod
    def integer_inputs_must_be_int(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "sampler integer")


class SamplePlan(AstroModel):
    sampler: SamplerSpec
    campaign_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ParameterRealization(AstroModel):
    sample_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    normalized_values: dict[str, FiniteFloat] = Field(default_factory=dict)
    physical_values: dict[str, FiniteFloat | str] = Field(default_factory=dict)
    model_variants: dict[str, str] = Field(default_factory=dict)
    weight: FiniteFloat = Field(gt=0.0, default=1.0)

    @field_validator("sample_index", mode="before")
    @classmethod
    def sample_index_must_be_int(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "sample index")


class AppliedBinding(AstroModel):
    parameter_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    value: FiniteFloat | str


class ScenarioRealization(AstroModel):
    sample_id: str = Field(min_length=1)
    base_scenario_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_scenario_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    bindings: tuple[AppliedBinding, ...] = Field(default_factory=tuple)
    valid: bool = True
    validation_errors: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validity_must_match_errors(self) -> ScenarioRealization:
        if self.valid == bool(self.validation_errors):
            raise ValueError(
                "valid realizations must have no errors and invalid ones must have errors"
            )
        return self


class EvaluatorSpec(AstroModel):
    evaluator_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    kind: EvaluatorKind
    workflow: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    backend: str | None = None
    model_artifact: str | None = None
    fallback_evaluator_id: str | None = None
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def kind_specific_fields_must_be_present(self) -> EvaluatorSpec:
        if self.kind is EvaluatorKind.SURROGATE and self.model_artifact is None:
            raise ValueError("surrogate evaluators require a model artifact")
        return self


class EvaluationTiming(AstroModel):
    setup_s: FiniteFloat = Field(ge=0.0, default=0.0)
    evaluation_s: FiniteFloat = Field(ge=0.0, default=0.0)
    serialization_s: FiniteFloat = Field(ge=0.0, default=0.0)
    total_s: FiniteFloat = Field(ge=0.0, default=0.0)

    @model_validator(mode="after")
    def total_must_cover_components(self) -> EvaluationTiming:
        if self.total_s + 1.0e-12 < self.setup_s + self.evaluation_s + self.serialization_s:
            raise ValueError("total evaluation time must cover component times")
        return self


class EvaluationOutcome(AstroModel):
    sample_id: str = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    status: OutcomeStatus
    timing: EvaluationTiming
    artifact_refs: tuple[str, ...] = Field(default_factory=tuple)
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def status_must_match_error(self) -> EvaluationOutcome:
        has_error = self.error_type is not None or self.error_message is not None
        if self.status is OutcomeStatus.SUCCESS and has_error:
            raise ValueError("successful outcomes cannot contain errors")
        if self.status is not OutcomeStatus.SUCCESS and not has_error:
            raise ValueError("unsuccessful outcomes require error details")
        return self


class MetricSpec(AstroModel):
    metric_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    extractor: str = Field(min_length=1)
    value_kind: MetricValueKind
    unit: str | None = None

    @model_validator(mode="after")
    def numeric_metrics_require_units(self) -> MetricSpec:
        if (
            self.value_kind
            in {
                MetricValueKind.NUMERIC,
                MetricValueKind.EVENT_TIME,
                MetricValueKind.TIME_SERIES_SUMMARY,
            }
            and not self.unit
        ):
            raise ValueError("numeric metrics require units")
        return self


class RequirementSpec(AstroModel):
    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    metric_id: str = Field(min_length=1)
    operator: RequirementOperator
    value: FiniteFloat | bool | None = None
    lower: FiniteFloat | None = None
    upper: FiniteFloat | None = None
    tolerance: FiniteFloat | None = Field(ge=0.0, default=None)

    @field_validator("value", "lower", "upper", "tolerance", mode="before")
    @classmethod
    def numeric_requirement_inputs_must_be_numbers(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        return _numeric_scalar_input_must_be_number(value, "requirement operand")

    @model_validator(mode="after")
    def validate_operands(self) -> RequirementSpec:
        if self.operator is RequirementOperator.BETWEEN:
            if self.lower is None or self.upper is None or self.lower > self.upper:
                raise ValueError("between requirements require lower <= upper")
        elif self.operator is RequirementOperator.WITHIN_TOLERANCE:
            if self.value is None or isinstance(self.value, bool) or self.tolerance is None:
                raise ValueError("within_tolerance requires numeric value and tolerance")
        elif self.operator in {RequirementOperator.IS_TRUE, RequirementOperator.IS_FALSE}:
            operands = (self.value, self.lower, self.upper, self.tolerance)
            if any(item is not None for item in operands):
                raise ValueError("boolean requirements do not accept operands")
        elif self.value is None or isinstance(self.value, bool):
            raise ValueError("comparison requirements require a numeric value")
        return self


class RequirementOutcome(AstroModel):
    requirement_id: str = Field(min_length=1)
    passed: bool | None
    margin: FiniteFloat | None = None
    reason: str | None = None


class CaseObservation(AstroModel):
    sample_id: str = Field(min_length=1)
    outcome_status: OutcomeStatus
    metric_values: dict[str, MetricValue] = Field(default_factory=dict)
    requirements: tuple[RequirementOutcome, ...] = Field(default_factory=tuple)
    evaluator_id: str = Field(min_length=1)
    evaluation_timing: EvaluationTiming | None = None
    artifact_refs: tuple[str, ...] = Field(default_factory=tuple)
    claim_boundary: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatisticSummary(AstroModel):
    metric_id: str = Field(min_length=1)
    count: int = Field(ge=0)
    effective_sample_size: FiniteFloat = Field(ge=0.0)
    mean: FiniteFloat | None = None
    variance: FiniteFloat | None = Field(ge=0.0, default=None)
    standard_error: FiniteFloat | None = Field(ge=0.0, default=None)
    quantiles: dict[str, FiniteFloat] = Field(default_factory=dict)


class CampaignStatistics(AstroModel):
    requested_samples: int = Field(gt=0)
    completed_samples: int = Field(ge=0)
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    metrics: tuple[StatisticSummary, ...] = Field(default_factory=tuple)
    requirement_probabilities: dict[str, FiniteFloat] = Field(default_factory=dict)
    requirement_denominator_policy: str = "all_completed_cases"
    convergence_history: tuple[dict[str, Any], ...] = Field(default_factory=tuple)


class WorkflowSpec(AstroModel):
    kind: str = Field(min_length=1)
    scenario: str = Field(min_length=1)


class FixedCountStopping(AstroModel):
    kind: Literal["fixed_count"] = "fixed_count"


class ConfidenceIntervalStopping(AstroModel):
    kind: Literal["ci_half_width"] = "ci_half_width"
    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_half_width: FiniteFloat = Field(gt=0.0, le=0.5)
    minimum_samples: int = Field(gt=0)
    maximum_samples: int = Field(gt=0)
    minimum_effective_sample_size: FiniteFloat | None = Field(gt=0.0, default=None)
    confidence: FiniteFloat = Field(gt=0.0, lt=1.0, default=0.95)
    batch_size: int = Field(gt=0, default=1)

    @field_validator("minimum_samples", "maximum_samples", "batch_size", mode="before")
    @classmethod
    def integer_inputs_must_be_int(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "stopping-rule integer")

    @model_validator(mode="after")
    def bounds_must_be_achievable(self) -> ConfidenceIntervalStopping:
        _validate_stopping_bounds(
            self.minimum_samples,
            self.maximum_samples,
            self.minimum_effective_sample_size,
        )
        return self


class MetricStabilityStopping(AstroModel):
    kind: Literal["metric_stability"] = "metric_stability"
    metric_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    absolute_tolerance: FiniteFloat = Field(ge=0.0)
    minimum_samples: int = Field(gt=0)
    maximum_samples: int = Field(gt=0)
    minimum_effective_sample_size: FiniteFloat | None = Field(gt=0.0, default=None)
    window: int = Field(ge=2, default=3)
    batch_size: int = Field(gt=0, default=1)

    @field_validator("minimum_samples", "maximum_samples", "window", "batch_size", mode="before")
    @classmethod
    def integer_inputs_must_be_int(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "stopping-rule integer")

    @model_validator(mode="after")
    def bounds_must_be_achievable(self) -> MetricStabilityStopping:
        _validate_stopping_bounds(
            self.minimum_samples,
            self.maximum_samples,
            self.minimum_effective_sample_size,
        )
        return self


StoppingRule = Annotated[
    FixedCountStopping | ConfidenceIntervalStopping | MetricStabilityStopping,
    Field(discriminator="kind"),
]


def _validate_stopping_bounds(
    minimum_samples: int,
    maximum_samples: int,
    minimum_effective_sample_size: float | None,
) -> None:
    if maximum_samples < minimum_samples:
        raise ValueError("stopping rules require minimum_samples <= maximum_samples")
    if (
        minimum_effective_sample_size is not None
        and minimum_effective_sample_size > maximum_samples
    ):
        raise ValueError("minimum effective sample size cannot exceed maximum_samples")


class RetentionSpec(AstroModel):
    policy: RetentionPolicy = RetentionPolicy.FAILURES_AND_BOUNDARIES
    boundary_tolerance: FiniteFloat = Field(ge=0.0, default=0.0)
    audit_fraction: FiniteFloat = Field(ge=0.0, le=1.0, default=0.0)


class CampaignDefinition(AstroModel):
    schema_version: SchemaVersion = "1.0"
    campaign_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    workflow: WorkflowSpec
    uncertainty: UncertaintyModel
    sampler: SamplerSpec
    evaluator: EvaluatorSpec
    metrics: tuple[MetricSpec, ...] = Field(default_factory=tuple)
    requirements: tuple[RequirementSpec, ...] = Field(default_factory=tuple)
    stopping: StoppingRule = Field(default_factory=FixedCountStopping)
    retention: RetentionSpec = Field(default_factory=RetentionSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ids_and_references_must_be_valid(self) -> CampaignDefinition:
        metric_ids = [metric.metric_id for metric in self.metrics]
        requirement_ids = [requirement.requirement_id for requirement in self.requirements]
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("metric ids must be unique")
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("requirement ids must be unique")
        known_metrics = set(metric_ids)
        if any(requirement.metric_id not in known_metrics for requirement in self.requirements):
            raise ValueError("requirements must reference a configured metric")
        if isinstance(
            self.stopping, ConfidenceIntervalStopping
        ) and self.stopping.requirement_id not in set(requirement_ids):
            raise ValueError("CI half-width stopping must reference a configured requirement")
        if (
            isinstance(self.stopping, MetricStabilityStopping)
            and self.stopping.metric_id not in known_metrics
        ):
            raise ValueError("metric-stability stopping must reference a configured metric")
        if (
            not isinstance(self.stopping, FixedCountStopping)
            and self.sampler.samples < self.stopping.maximum_samples
        ):
            raise ValueError("sampler samples must reach adaptive stopping maximum_samples")
        return self


class CampaignResult(AstroModel):
    schema_version: SchemaVersion = "1.0"
    campaign_id: str = Field(min_length=1)
    definition_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: CampaignState
    statistics: CampaignStatistics
    case_index_path: str = Field(min_length=1)
    sample_index_path: str = Field(min_length=1)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    claim_boundary: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_finite_mapping(values: dict[str, float], label: str) -> None:
    if not all(isfinite(value) for value in values.values()):
        raise ValueError(f"{label} values must be finite")
