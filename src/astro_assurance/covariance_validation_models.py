from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, FiniteFloat, model_validator

from astro_core.models import AstroModel, ForceModelConfig

StateOrder = tuple[
    Literal["x"],
    Literal["y"],
    Literal["z"],
    Literal["vx"],
    Literal["vy"],
    Literal["vz"],
]
StateUnits = tuple[
    Literal["km"],
    Literal["km"],
    Literal["km"],
    Literal["km/s"],
    Literal["km/s"],
    Literal["km/s"],
]
Vector6 = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
Matrix6 = tuple[Vector6, Vector6, Vector6, Vector6, Vector6, Vector6]
ForceFeature = Literal[
    "two_body",
    "j2",
    "high_order_gravity",
    "atmospheric_drag",
    "solar_radiation_pressure",
    "third_body_gravity",
]


class StrictCovarianceModel(AstroModel):
    """Closed schema base for covariance-validation products."""

    model_config = ConfigDict(extra="forbid")


class CovarianceUnitsPolicy(StrictCovarianceModel):
    frame: Literal["EME2000"] = "EME2000"
    representation: Literal["cartesian"] = "cartesian"
    time_scale: Literal["UTC"] = "UTC"
    state_order: StateOrder = ("x", "y", "z", "vx", "vy", "vz")
    state_units: StateUnits = ("km", "km", "km", "km/s", "km/s", "km/s")
    covariance_units_policy: Literal["outer_product_of_state_units"] = (
        "outer_product_of_state_units"
    )


class CovarianceValidationThresholds(StrictCovarianceModel):
    minimum_epochs: int = Field(ge=2)
    symmetry_tolerance: FiniteFloat = Field(ge=0.0)
    minimum_eigenvalue: FiniteFloat = Field(gt=0.0)
    maximum_condition_number: FiniteFloat = Field(ge=1.0)
    maximum_relative_covariance_frobenius_error: FiniteFloat = Field(ge=0.0)
    covariance_trace_ratio_minimum: FiniteFloat = Field(gt=0.0)
    covariance_trace_ratio_maximum: FiniteFloat = Field(gt=0.0)
    maximum_accumulated_state_transition_frobenius_error: FiniteFloat = Field(ge=0.0)
    generalized_eigenvalue_minimum: FiniteFloat = Field(gt=0.0)
    generalized_eigenvalue_maximum: FiniteFloat = Field(gt=0.0)
    maximum_state_position_delta_km: FiniteFloat = Field(ge=0.0)
    maximum_state_velocity_delta_km_s: FiniteFloat = Field(ge=0.0)
    confidence_level: FiniteFloat = Field(gt=0.0, lt=1.0)
    minimum_empirical_samples: int = Field(ge=2)
    minimum_coverage: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def bounds_are_not_contradictory(self) -> CovarianceValidationThresholds:
        if self.generalized_eigenvalue_minimum > self.generalized_eigenvalue_maximum:
            raise ValueError("generalized eigenvalue minimum cannot exceed maximum")
        if not (self.generalized_eigenvalue_minimum <= 1.0 <= self.generalized_eigenvalue_maximum):
            raise ValueError("generalized eigenvalue interval must include unity")
        if self.covariance_trace_ratio_minimum > self.covariance_trace_ratio_maximum:
            raise ValueError("covariance trace-ratio minimum cannot exceed maximum")
        if not (
            self.covariance_trace_ratio_minimum <= 1.0 <= self.covariance_trace_ratio_maximum
        ):
            raise ValueError("covariance trace-ratio interval must include unity")
        return self


class CovarianceIndependenceDeclaration(StrictCovarianceModel):
    candidate_implementation: str = Field(min_length=1)
    reference_implementation: str = Field(min_length=1)
    independent_implementations: bool
    shared_code_paths: tuple[str, ...] = ()
    shared_data_sources: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def declarations_are_distinct(self) -> CovarianceIndependenceDeclaration:
        if (
            self.independent_implementations
            and self.candidate_implementation == self.reference_implementation
        ):
            raise ValueError("candidate and reference implementations must be distinct")
        for label, values in (
            ("shared code paths", self.shared_code_paths),
            ("shared data sources", self.shared_data_sources),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must not contain duplicates")
        return self


class CovarianceValidationProtocol(StrictCovarianceModel):
    protocol_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    workflow: Literal["covariance_validation_v1"] = "covariance_validation_v1"
    candidate_trajectory_path: str = Field(min_length=1)
    reference_trajectory_path: str = Field(min_length=1)
    empirical_evidence_path: str | None = Field(default=None, min_length=1)
    empirical_scenario_path: str | None = Field(default=None, min_length=1)
    independence_review_path: str | None = Field(default=None, min_length=1)
    units_policy: CovarianceUnitsPolicy
    thresholds: CovarianceValidationThresholds
    independence: CovarianceIndependenceDeclaration
    required_force_features: tuple[ForceFeature, ...] = Field(min_length=1)
    claim_boundary: Literal[
        "covariance_comparison_evidence_not_flight_certification_or_operational_authority"
    ] = "covariance_comparison_evidence_not_flight_certification_or_operational_authority"
    source_path: str | None = Field(default=None, exclude=True)
    source_digest: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def protocol_inputs_are_unambiguous(self) -> CovarianceValidationProtocol:
        paths = [self.candidate_trajectory_path, self.reference_trajectory_path]
        if self.empirical_evidence_path is not None:
            paths.append(self.empirical_evidence_path)
        if self.empirical_scenario_path is not None:
            paths.append(self.empirical_scenario_path)
        if self.independence_review_path is not None:
            paths.append(self.independence_review_path)
        if len(paths) != len(set(paths)):
            raise ValueError("protocol input paths must be distinct")
        if (self.empirical_evidence_path is None) != (
            self.empirical_scenario_path is None
        ):
            raise ValueError(
                "empirical evidence and empirical scenario paths must be provided together"
            )
        if len(self.required_force_features) != len(set(self.required_force_features)):
            raise ValueError("required force features must not contain duplicates")
        return self


class EmpiricalCovarianceRawSample(StrictCovarianceModel):
    sample_id: str = Field(min_length=1)
    epoch: datetime
    state_error: Vector6
    predicted_covariance: Matrix6
    independent_truth: bool
    initial_state_perturbation: Vector6
    nominal_truth_state: Vector6
    realized_truth_state: Vector6

    @model_validator(mode="after")
    def epoch_is_aware(self) -> EmpiricalCovarianceRawSample:
        if self.epoch.tzinfo is None or self.epoch.utcoffset() is None:
            raise ValueError("empirical sample epoch must include timezone information")
        return self

    @model_validator(mode="after")
    def state_error_matches_raw_truth_states(self) -> EmpiricalCovarianceRawSample:
        expected = tuple(
            float(realized) - float(nominal)
            for realized, nominal in zip(
                self.realized_truth_state, self.nominal_truth_state, strict=True
            )
        )
        if any(
            abs(float(actual) - expected_component) > 1e-12
            for actual, expected_component in zip(self.state_error, expected, strict=True)
        ):
            raise ValueError("empirical state_error must equal realized minus nominal truth")
        return self


class EmpiricalCovarianceCampaignProvenance(StrictCovarianceModel):
    scenario_id: str = Field(min_length=1)
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictor_trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictor_backend: Literal["orekit", "tudat", "local"]
    predictor_implementation: str = Field(min_length=1)
    truth_backend: Literal["orekit", "tudat", "local"]
    seed: int = Field(ge=0)
    sampling_engine: Literal["numpy.random.PCG64"] = "numpy.random.PCG64"
    perturbation_distribution: Literal["zero_mean_gaussian_initial_state"] = (
        "zero_mean_gaussian_initial_state"
    )
    process_noise_realization: Literal["none"] = "none"
    evaluation_epoch: datetime
    initial_covariance: Matrix6
    sample_count: int = Field(ge=2)
    force_model: ForceModelConfig

    @model_validator(mode="after")
    def campaign_is_independent_and_time_bounded(
        self,
    ) -> EmpiricalCovarianceCampaignProvenance:
        if self.predictor_backend == self.truth_backend:
            raise ValueError("empirical predictor and truth backends must be independent")
        if self.evaluation_epoch.tzinfo is None or self.evaluation_epoch.utcoffset() is None:
            raise ValueError("empirical evaluation epoch must include timezone information")
        return self


class EmpiricalCovarianceArtifact(StrictCovarianceModel):
    artifact_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    units_policy: CovarianceUnitsPolicy
    population_definition: str = Field(min_length=1)
    independent_realizations: bool
    independence_basis: str = Field(min_length=1)
    campaign_provenance: EmpiricalCovarianceCampaignProvenance
    samples: tuple[EmpiricalCovarianceRawSample, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def samples_are_unique(self) -> EmpiricalCovarianceArtifact:
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("empirical sample ids must be unique")
        perturbations = [sample.initial_state_perturbation for sample in self.samples]
        if len(perturbations) != len(set(perturbations)):
            raise ValueError("empirical initial-state perturbations must not be duplicated")
        if len(self.samples) != self.campaign_provenance.sample_count:
            raise ValueError("empirical sample count must match campaign provenance")
        if any(
            sample.epoch != self.campaign_provenance.evaluation_epoch
            for sample in self.samples
        ):
            raise ValueError("empirical samples must use the campaign evaluation epoch")
        return self


class CovarianceIndependenceReview(StrictCovarianceModel):
    review_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    candidate_implementation: str = Field(min_length=1)
    reference_implementation: str = Field(min_length=1)
    evidence_reviewed: tuple[str, ...] = Field(min_length=1)
    conclusion: Literal["independent_implementations"]
    rationale: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def review_is_complete(self) -> CovarianceIndependenceReview:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("independence review timestamp must include timezone information")
        if self.candidate_implementation == self.reference_implementation:
            raise ValueError("independence review implementations must be distinct")
        if len(self.evidence_reviewed) != len(set(self.evidence_reviewed)):
            raise ValueError("reviewed evidence entries must be unique")
        return self


class CovarianceEpochComparisonDiagnostic(StrictCovarianceModel):
    epoch: datetime
    symmetry_error: FiniteFloat = Field(ge=0.0)
    candidate_minimum_eigenvalue: FiniteFloat
    reference_minimum_eigenvalue: FiniteFloat
    candidate_condition_number: FiniteFloat = Field(ge=1.0)
    reference_condition_number: FiniteFloat = Field(ge=1.0)
    relative_covariance_frobenius_error: FiniteFloat = Field(ge=0.0)
    covariance_trace_ratio: FiniteFloat = Field(gt=0.0)
    accumulated_state_transition_frobenius_error: FiniteFloat = Field(ge=0.0)
    generalized_eigenvalue_minimum: FiniteFloat = Field(gt=0.0)
    generalized_eigenvalue_maximum: FiniteFloat = Field(gt=0.0)
    state_position_delta_km: FiniteFloat = Field(ge=0.0)
    state_velocity_delta_km_s: FiniteFloat = Field(ge=0.0)
    criteria_satisfied: bool
    failed_criteria: tuple[str, ...] = ()

    @model_validator(mode="after")
    def diagnostic_is_consistent(self) -> CovarianceEpochComparisonDiagnostic:
        if self.epoch.tzinfo is None or self.epoch.utcoffset() is None:
            raise ValueError("comparison epoch must include timezone information")
        if self.generalized_eigenvalue_minimum > self.generalized_eigenvalue_maximum:
            raise ValueError("diagnostic generalized eigenvalue bounds are reversed")
        if len(self.failed_criteria) != len(set(self.failed_criteria)):
            raise ValueError("failed criteria must not contain duplicates")
        if self.criteria_satisfied == bool(self.failed_criteria):
            raise ValueError("criteria_satisfied must be equivalent to no failed criteria")
        return self


class CovarianceComparisonSummary(StrictCovarianceModel):
    requested_epochs: int = Field(ge=0)
    compared_epochs: int = Field(ge=0)
    passed_epochs: int = Field(ge=0)
    maximum_symmetry_error: FiniteFloat | None = Field(default=None, ge=0.0)
    minimum_candidate_eigenvalue: FiniteFloat | None = None
    maximum_candidate_condition_number: FiniteFloat | None = Field(default=None, ge=1.0)
    maximum_relative_covariance_frobenius_error: FiniteFloat | None = Field(default=None, ge=0.0)
    covariance_trace_ratio_minimum: FiniteFloat | None = Field(default=None, gt=0.0)
    covariance_trace_ratio_maximum: FiniteFloat | None = Field(default=None, gt=0.0)
    maximum_accumulated_state_transition_frobenius_error: FiniteFloat | None = Field(
        default=None, ge=0.0
    )
    generalized_eigenvalue_minimum: FiniteFloat | None = Field(default=None, gt=0.0)
    generalized_eigenvalue_maximum: FiniteFloat | None = Field(default=None, gt=0.0)
    maximum_state_position_delta_km: FiniteFloat | None = Field(default=None, ge=0.0)
    maximum_state_velocity_delta_km_s: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def counts_and_optional_metrics_are_consistent(self) -> CovarianceComparisonSummary:
        if not self.passed_epochs <= self.compared_epochs <= self.requested_epochs:
            raise ValueError(
                "comparison summary counts must satisfy passed <= compared <= requested"
            )
        metrics = (
            self.maximum_symmetry_error,
            self.minimum_candidate_eigenvalue,
            self.maximum_candidate_condition_number,
            self.maximum_relative_covariance_frobenius_error,
            self.covariance_trace_ratio_minimum,
            self.covariance_trace_ratio_maximum,
            self.maximum_accumulated_state_transition_frobenius_error,
            self.generalized_eigenvalue_minimum,
            self.generalized_eigenvalue_maximum,
            self.maximum_state_position_delta_km,
            self.maximum_state_velocity_delta_km_s,
        )
        if (self.compared_epochs == 0) != all(metric is None for metric in metrics):
            raise ValueError("summary metrics must be absent exactly when no epochs were compared")
        if (
            self.generalized_eigenvalue_minimum is not None
            and self.generalized_eigenvalue_maximum is not None
            and self.generalized_eigenvalue_minimum > self.generalized_eigenvalue_maximum
        ):
            raise ValueError("summary generalized eigenvalue bounds are reversed")
        return self


class EmpiricalNEESSummary(StrictCovarianceModel):
    sample_count: int = Field(ge=0)
    confidence_level: FiniteFloat = Field(gt=0.0, lt=1.0)
    individual_lower_bound: FiniteFloat = Field(ge=0.0)
    individual_upper_bound: FiniteFloat = Field(gt=0.0)
    mean_lower_bound: FiniteFloat = Field(ge=0.0)
    mean_upper_bound: FiniteFloat = Field(gt=0.0)
    mean_nees: FiniteFloat = Field(ge=0.0)
    samples_within_bounds: int = Field(ge=0)
    coverage: FiniteFloat = Field(ge=0.0, le=1.0)
    coverage_lower_confidence_bound: FiniteFloat = Field(ge=0.0, le=1.0)
    criteria_satisfied: bool

    @model_validator(mode="after")
    def nees_summary_is_consistent(self) -> EmpiricalNEESSummary:
        if self.individual_lower_bound >= self.individual_upper_bound:
            raise ValueError("individual NEES lower bound must be below upper bound")
        if self.mean_lower_bound >= self.mean_upper_bound:
            raise ValueError("mean NEES lower bound must be below upper bound")
        if self.samples_within_bounds > self.sample_count:
            raise ValueError("NEES in-bound sample count cannot exceed sample count")
        expected = 0.0 if self.sample_count == 0 else self.samples_within_bounds / self.sample_count
        if abs(float(self.coverage) - expected) > 1e-12:
            raise ValueError("NEES coverage must equal in-bound samples divided by sample count")
        if self.coverage_lower_confidence_bound > self.coverage:
            raise ValueError("NEES coverage lower confidence bound cannot exceed coverage")
        return self


class CovarianceSourceBinding(StrictCovarianceModel):
    role: Literal[
        "protocol",
        "candidate_trajectory",
        "reference_trajectory",
        "empirical_evidence",
        "empirical_scenario",
        "independence_review",
    ]
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CovarianceValidationDisposition(StrEnum):
    CRITERIA_SATISFIED = "criteria_satisfied"
    CRITERIA_FAILED = "criteria_failed"
    ADDITIONAL_EVIDENCE_REQUIRED = "additional_evidence_required"


class CovarianceValidationBlocker(StrictCovarianceModel):
    blocker_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: Literal["input", "independence", "force_feature", "comparison", "empirical"]
    statement: str = Field(min_length=1)
    required_evidence: str = Field(min_length=1)


class CovarianceValidationResult(StrictCovarianceModel):
    protocol_id: str = Field(min_length=1)
    workflow: Literal["covariance_validation_v1"] = "covariance_validation_v1"
    source_bindings: tuple[CovarianceSourceBinding, ...] = Field(min_length=3, max_length=6)
    diagnostics: tuple[CovarianceEpochComparisonDiagnostic, ...]
    comparison_summary: CovarianceComparisonSummary
    empirical_nees_summary: EmpiricalNEESSummary | None = None
    blockers: tuple[CovarianceValidationBlocker, ...] = ()
    disposition: CovarianceValidationDisposition
    claim_boundary: Literal[
        "covariance_comparison_evidence_not_flight_certification_or_operational_authority"
    ] = "covariance_comparison_evidence_not_flight_certification_or_operational_authority"
    certification_claim: Literal["no_certification_claim"] = "no_certification_claim"

    @model_validator(mode="after")
    def result_is_fail_closed_and_internally_consistent(self) -> CovarianceValidationResult:
        roles = [binding.role for binding in self.source_bindings]
        required = {"protocol", "candidate_trajectory", "reference_trajectory"}
        if len(roles) != len(set(roles)) or not required.issubset(roles):
            raise ValueError("source bindings must bind each required role exactly once")
        has_empirical_binding = "empirical_evidence" in roles
        if has_empirical_binding != (self.empirical_nees_summary is not None):
            raise ValueError("empirical binding and NEES summary must be present together")
        if ("empirical_scenario" in roles) != has_empirical_binding:
            raise ValueError(
                "empirical scenario and evidence bindings must be present together"
            )
        epochs = [diagnostic.epoch for diagnostic in self.diagnostics]
        if len(epochs) != len(set(epochs)):
            raise ValueError("comparison diagnostic epochs must be unique")
        blocker_ids = [blocker.blocker_id for blocker in self.blockers]
        if len(blocker_ids) != len(set(blocker_ids)):
            raise ValueError("blocker ids must be unique")
        summary = self.comparison_summary
        if summary.compared_epochs != len(self.diagnostics):
            raise ValueError("compared epoch count must match diagnostics")
        if summary.passed_epochs != sum(item.criteria_satisfied for item in self.diagnostics):
            raise ValueError("passed epoch count must match diagnostics")
        if self.diagnostics:
            expected = (
                max(float(item.symmetry_error) for item in self.diagnostics),
                min(float(item.candidate_minimum_eigenvalue) for item in self.diagnostics),
                max(float(item.candidate_condition_number) for item in self.diagnostics),
                max(float(item.relative_covariance_frobenius_error) for item in self.diagnostics),
                min(float(item.covariance_trace_ratio) for item in self.diagnostics),
                max(float(item.covariance_trace_ratio) for item in self.diagnostics),
                max(
                    float(item.accumulated_state_transition_frobenius_error)
                    for item in self.diagnostics
                ),
                min(float(item.generalized_eigenvalue_minimum) for item in self.diagnostics),
                max(float(item.generalized_eigenvalue_maximum) for item in self.diagnostics),
                max(float(item.state_position_delta_km) for item in self.diagnostics),
                max(float(item.state_velocity_delta_km_s) for item in self.diagnostics),
            )
            actual = (
                summary.maximum_symmetry_error,
                summary.minimum_candidate_eigenvalue,
                summary.maximum_candidate_condition_number,
                summary.maximum_relative_covariance_frobenius_error,
                summary.covariance_trace_ratio_minimum,
                summary.covariance_trace_ratio_maximum,
                summary.maximum_accumulated_state_transition_frobenius_error,
                summary.generalized_eigenvalue_minimum,
                summary.generalized_eigenvalue_maximum,
                summary.maximum_state_position_delta_km,
                summary.maximum_state_velocity_delta_km_s,
            )
            if actual != expected:
                raise ValueError("comparison summary extrema must match diagnostics")
        all_comparisons_pass = bool(self.diagnostics) and all(
            item.criteria_satisfied for item in self.diagnostics
        )
        empirical_pass = (
            self.empirical_nees_summary is None or self.empirical_nees_summary.criteria_satisfied
        )
        numerical_failure = bool(self.diagnostics) and (
            not all_comparisons_pass
            or (
                self.empirical_nees_summary is not None
                and not self.empirical_nees_summary.criteria_satisfied
            )
        )
        expected_disposition = (
            CovarianceValidationDisposition.CRITERIA_FAILED
            if numerical_failure
            else CovarianceValidationDisposition.ADDITIONAL_EVIDENCE_REQUIRED
            if self.blockers
            else CovarianceValidationDisposition.CRITERIA_SATISFIED
            if all_comparisons_pass and empirical_pass
            else CovarianceValidationDisposition.ADDITIONAL_EVIDENCE_REQUIRED
        )
        if self.disposition is not expected_disposition:
            raise ValueError("disposition must match blockers and validation outcomes")
        return self
