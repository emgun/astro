from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from statistics import median
from typing import Any, Literal

from pydantic import Field, FiniteFloat, model_validator

from astro_assurance.models import (
    InsertionDispersion,
    MissionAssuranceCase,
    MissionAssuranceInputOverrides,
)
from astro_core.models import AstroModel, ForceModelName


class AssuranceValidationProfile(StrEnum):
    MATCHED_TWO_BODY = "matched_two_body"
    TRUTH_J2_ESTIMATOR_TWO_BODY = "truth_j2_estimator_two_body"


class AssuranceValidationStatus(StrEnum):
    SUCCESS = "success"
    EXECUTION_FAILURE = "execution_failure"


class AssuranceCalibrationAuthority(StrEnum):
    ILLUSTRATIVE = "illustrative"
    PROJECT_DERIVED = "project_derived"
    EXTERNAL_REFERENCE_INFORMED = "external_reference_informed"
    MISSION_TEST_CALIBRATED = "mission_test_calibrated"
    FLIGHT_CALIBRATED = "flight_calibrated"


class AssuranceCalibrationPromotionStatus(StrEnum):
    ILLUSTRATIVE = "illustrative"
    REFERENCE_INFORMED = "reference_informed"
    MISSION_CALIBRATED = "mission_calibrated"


class AssuranceCalibrationSourceKind(StrEnum):
    OFFICIAL_TECHNICAL_REFERENCE = "official_technical_reference"
    PROJECT_CONFIGURATION = "project_configuration"
    MISSION_TEST_DATA = "mission_test_data"
    FLIGHT_DATA = "flight_data"


class AssuranceCalibrationSource(AstroModel):
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    revision_or_date: str = Field(min_length=1)
    location: str = Field(min_length=1)
    source_kind: AssuranceCalibrationSourceKind
    applicability: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


class AssuranceCalibrationBound(AstroModel):
    parameter: str = Field(min_length=1)
    minimum: FiniteFloat
    maximum: FiniteFloat
    unit: str = Field(min_length=1)
    authority: AssuranceCalibrationAuthority
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def range_and_sources_must_be_valid(self) -> AssuranceCalibrationBound:
        if self.minimum > self.maximum:
            raise ValueError("calibration bound minimum must not exceed maximum")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("calibration bound source ids must be unique")
        if self.authority is not AssuranceCalibrationAuthority.ILLUSTRATIVE and not self.source_ids:
            raise ValueError("non-illustrative calibration bounds require a source")
        return self


class AssuranceValidationCalibrationManifest(AstroModel):
    calibration_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    protocol_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    sources: tuple[AssuranceCalibrationSource, ...] = Field(min_length=1)
    parameter_bounds: tuple[AssuranceCalibrationBound, ...] = Field(min_length=1)
    promotion_status: AssuranceCalibrationPromotionStatus
    coverage_policy: Literal["all_configured_values_within_declared_envelopes"] = (
        "all_configured_values_within_declared_envelopes"
    )
    claim_boundary: Literal[
        "parameter_envelope_traceability_not_operational_calibration_or_probability"
    ] = "parameter_envelope_traceability_not_operational_calibration_or_probability"
    source_path: str | None = Field(default=None, exclude=True)
    source_digest: str | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_and_promotion_must_be_consistent(
        self,
    ) -> AssuranceValidationCalibrationManifest:
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("calibration source ids must be unique")
        parameters = [bound.parameter for bound in self.parameter_bounds]
        if len(set(parameters)) != len(parameters):
            raise ValueError("calibration parameters must be unique")
        sources = {source.source_id: source for source in self.sources}
        required_source_kind = {
            AssuranceCalibrationAuthority.PROJECT_DERIVED: (
                AssuranceCalibrationSourceKind.PROJECT_CONFIGURATION
            ),
            AssuranceCalibrationAuthority.EXTERNAL_REFERENCE_INFORMED: (
                AssuranceCalibrationSourceKind.OFFICIAL_TECHNICAL_REFERENCE
            ),
            AssuranceCalibrationAuthority.MISSION_TEST_CALIBRATED: (
                AssuranceCalibrationSourceKind.MISSION_TEST_DATA
            ),
            AssuranceCalibrationAuthority.FLIGHT_CALIBRATED: (
                AssuranceCalibrationSourceKind.FLIGHT_DATA
            ),
        }
        for bound in self.parameter_bounds:
            unknown = set(bound.source_ids) - set(sources)
            if unknown:
                raise ValueError(
                    f"calibration bound {bound.parameter} references unknown sources: "
                    f"{sorted(unknown)}"
                )
            required_kind = required_source_kind.get(bound.authority)
            if required_kind is not None and not any(
                sources[source_id].source_kind is required_kind
                for source_id in bound.source_ids
            ):
                raise ValueError(
                    f"calibration bound {bound.parameter} lacks a source for its authority"
                )
        expected = derive_calibration_promotion_status(self.parameter_bounds)
        if self.promotion_status is not expected:
            raise ValueError("calibration promotion status must match bound authority")
        return self


class AssuranceValidationRealization(AstroModel):
    case_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    dispersion: InsertionDispersion
    input_overrides: MissionAssuranceInputOverrides

    @model_validator(mode="after")
    def profile_force_models_are_protocol_owned(self) -> AssuranceValidationRealization:
        if self.input_overrides.truth_force_model is not None:
            raise ValueError("validation realization must not set truth_force_model")
        if self.input_overrides.estimation_force_model is not None:
            raise ValueError("validation realization must not set estimation_force_model")
        if self.input_overrides.tracking_noise_seed is None:
            raise ValueError("validation realization must set tracking_noise_seed")
        return self


class PairedAssuranceValidationProtocol(AstroModel):
    protocol_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    assurance_scenario: str = Field(min_length=1)
    calibration_evidence: str = Field(min_length=1)
    tracking_duration_s: FiniteFloat = Field(gt=0.0)
    correction_elapsed_s: FiniteFloat = Field(gt=0.0)
    verification_elapsed_s: FiniteFloat = Field(gt=0.0)
    diagnostic_maximum_component_delta_v_km_s: FiniteFloat = Field(gt=0.0)
    diagnostic_maximum_total_delta_v_km_s: FiniteFloat = Field(gt=0.0)
    realizations: tuple[AssuranceValidationRealization, ...] = Field(min_length=1)
    claim_boundary: Literal[
        "paired_simulation_design_space_validation_not_operational_probability_or_flight_authority"
    ] = "paired_simulation_design_space_validation_not_operational_probability_or_flight_authority"
    source_path: str | None = Field(default=None, exclude=True)
    source_digest: str | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def case_ids_must_be_unique(self) -> PairedAssuranceValidationProtocol:
        case_ids = [realization.case_id for realization in self.realizations]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("validation realization case ids must be unique")
        noise_seeds = [
            realization.input_overrides.tracking_noise_seed
            for realization in self.realizations
        ]
        if len(set(noise_seeds)) != len(noise_seeds):
            raise ValueError("validation realization tracking noise seeds must be unique")
        if not (
            self.correction_elapsed_s < self.verification_elapsed_s <= self.tracking_duration_s
        ):
            raise ValueError(
                "validation protocol requires correction < verification <= tracking duration"
            )
        if (
            self.diagnostic_maximum_total_delta_v_km_s
            < self.diagnostic_maximum_component_delta_v_km_s
        ):
            raise ValueError("diagnostic total delta-v limit must be at least the component limit")
        return self


class AssuranceValidationProfileResult(AstroModel):
    profile: AssuranceValidationProfile
    truth_force_model: ForceModelName
    estimation_force_model: ForceModelName
    status: AssuranceValidationStatus
    passed: bool | None = None
    assurance_case_passed: bool | None = None
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    assurance_result_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    assurance_case: MissionAssuranceCase | None = None
    error_type: str | None = None
    error_message: str | None = None
    failure_phase: str | None = None

    @model_validator(mode="after")
    def status_fields_must_be_consistent(self) -> AssuranceValidationProfileResult:
        if self.status is AssuranceValidationStatus.SUCCESS:
            if self.assurance_case is None or self.passed is None:
                raise ValueError("successful validation profile requires an assurance case")
            if self.assurance_case_passed != self.assurance_case.passed:
                raise ValueError("assurance_case_passed must match the embedded assurance case")
            expected_digest = sha256(
                self.assurance_case.model_dump_json().encode("utf-8")
            ).hexdigest()
            if self.assurance_result_digest != expected_digest:
                raise ValueError("assurance result digest must match the embedded case")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful validation profile cannot contain an error")
        else:
            if self.assurance_case is not None or self.assurance_result_digest is not None:
                raise ValueError("failed validation profile cannot contain an assurance case")
            if (
                self.passed is not None
                or self.assurance_case_passed is not None
                or not self.error_type
                or not self.error_message
            ):
                raise ValueError("failed validation profile requires typed error evidence")
        return self


class AssuranceValidationPairResult(AstroModel):
    case_id: str = Field(min_length=1)
    realization: AssuranceValidationRealization
    matched: AssuranceValidationProfileResult
    mismatched: AssuranceValidationProfileResult
    paired_complete: bool
    delta_mismatched_minus_matched: dict[str, FiniteFloat] = Field(default_factory=dict)
    pass_reversal: Literal["unchanged", "regression", "improvement", "not_comparable"]

    @model_validator(mode="after")
    def pair_fields_must_match_profiles(self) -> AssuranceValidationPairResult:
        if self.matched.profile is not AssuranceValidationProfile.MATCHED_TWO_BODY:
            raise ValueError("matched slot must contain the matched_two_body profile")
        if (
            self.mismatched.profile
            is not AssuranceValidationProfile.TRUTH_J2_ESTIMATOR_TWO_BODY
        ):
            raise ValueError(
                "mismatched slot must contain the truth_j2_estimator_two_body profile"
            )
        expected_complete = (
            self.matched.status is AssuranceValidationStatus.SUCCESS
            and self.mismatched.status is AssuranceValidationStatus.SUCCESS
        )
        if self.paired_complete != expected_complete:
            raise ValueError("paired_complete must match profile outcomes")
        if self.case_id != self.realization.case_id:
            raise ValueError("pair case id must match realization")
        if not expected_complete and self.delta_mismatched_minus_matched:
            raise ValueError("incomplete pair cannot contain metric deltas")
        return self


class PairedMetricDeltaSummary(AstroModel):
    count: int = Field(gt=0)
    minimum: FiniteFloat
    median: FiniteFloat
    maximum: FiniteFloat


class AssuranceValidationSummary(AstroModel):
    requested_pairs: int = Field(gt=0)
    matched_completed: int = Field(ge=0)
    mismatched_completed: int = Field(ge=0)
    paired_complete: int = Field(ge=0)
    matched_passed: int = Field(ge=0)
    mismatched_passed: int = Field(ge=0)
    pass_regressions: int = Field(ge=0)
    pass_improvements: int = Field(ge=0)
    unchanged_pass_disposition: int = Field(ge=0)
    paired_metric_deltas: dict[str, PairedMetricDeltaSummary] = Field(default_factory=dict)
    denominator_policy: Literal["counts_only_profiles_unpooled_no_probability_estimate"] = (
        "counts_only_profiles_unpooled_no_probability_estimate"
    )


class PairedAssuranceValidationResult(AstroModel):
    protocol_id: str = Field(min_length=1)
    workflow: str = "paired_mission_assurance_validation_v1"
    protocol_source_path: str = Field(min_length=1)
    protocol_source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    assurance_source_path: str = Field(min_length=1)
    assurance_source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_id: str = Field(min_length=1)
    calibration_source_path: str = Field(min_length=1)
    calibration_source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_promotion_status: AssuranceCalibrationPromotionStatus
    calibration_claim_boundary: Literal[
        "parameter_envelope_traceability_not_operational_calibration_or_probability"
    ]
    pairs: tuple[AssuranceValidationPairResult, ...] = Field(min_length=1)
    summary: AssuranceValidationSummary
    claim_boundary: Literal[
        "paired_simulation_design_space_validation_not_operational_probability_or_flight_authority"
    ]
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def summary_must_match_pairs(self) -> PairedAssuranceValidationResult:
        expected = summarize_validation_pairs(self.pairs)
        if self.summary != expected:
            raise ValueError("validation summary must match paired outcomes")
        return self


def summarize_validation_pairs(
    pairs: tuple[AssuranceValidationPairResult, ...],
) -> AssuranceValidationSummary:
    metric_names = sorted(
        {
            metric
            for pair in pairs
            if pair.paired_complete
            for metric in pair.delta_mismatched_minus_matched
        }
    )
    metric_deltas = {}
    for metric in metric_names:
        values = [
            float(pair.delta_mismatched_minus_matched[metric])
            for pair in pairs
            if pair.paired_complete and metric in pair.delta_mismatched_minus_matched
        ]
        metric_deltas[metric] = PairedMetricDeltaSummary(
            count=len(values),
            minimum=min(values),
            median=median(values),
            maximum=max(values),
        )
    return AssuranceValidationSummary(
        requested_pairs=len(pairs),
        matched_completed=sum(
            pair.matched.status is AssuranceValidationStatus.SUCCESS for pair in pairs
        ),
        mismatched_completed=sum(
            pair.mismatched.status is AssuranceValidationStatus.SUCCESS for pair in pairs
        ),
        paired_complete=sum(pair.paired_complete for pair in pairs),
        matched_passed=sum(pair.matched.passed is True for pair in pairs),
        mismatched_passed=sum(pair.mismatched.passed is True for pair in pairs),
        pass_regressions=sum(pair.pass_reversal == "regression" for pair in pairs),
        pass_improvements=sum(pair.pass_reversal == "improvement" for pair in pairs),
        unchanged_pass_disposition=sum(pair.pass_reversal == "unchanged" for pair in pairs),
        paired_metric_deltas=metric_deltas,
    )


def derive_calibration_promotion_status(
    bounds: tuple[AssuranceCalibrationBound, ...],
) -> AssuranceCalibrationPromotionStatus:
    authorities = {bound.authority for bound in bounds}
    if AssuranceCalibrationAuthority.ILLUSTRATIVE in authorities:
        return AssuranceCalibrationPromotionStatus.ILLUSTRATIVE
    calibrated = {
        AssuranceCalibrationAuthority.MISSION_TEST_CALIBRATED,
        AssuranceCalibrationAuthority.FLIGHT_CALIBRATED,
    }
    if authorities and authorities <= calibrated:
        return AssuranceCalibrationPromotionStatus.MISSION_CALIBRATED
    return AssuranceCalibrationPromotionStatus.REFERENCE_INFORMED
