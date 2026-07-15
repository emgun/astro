from __future__ import annotations

from statistics import median
from typing import Any, Literal

from pydantic import Field, FiniteFloat, model_validator

from astro_assurance.validation_models import (
    AssuranceCalibrationPromotionStatus,
    AssuranceValidationProfile,
    AssuranceValidationProfileResult,
    AssuranceValidationRealization,
    AssuranceValidationStatus,
)
from astro_core.models import AstroModel, ForceModelName

MODEL_FORM_FACTORIAL_PROFILE_VALUES = (
    "matched_two_body",
    "truth_two_body_estimator_j2",
    "truth_j2_estimator_two_body",
    "matched_j2",
)

ModelFormContrastId = Literal[
    "estimator_j2_minus_two_body_under_truth_two_body",
    "estimator_j2_minus_two_body_under_truth_j2",
    "difference_in_differences_interaction",
]

MODEL_FORM_FACTORIAL_CONTRAST_IDS: tuple[ModelFormContrastId, ...] = (
    "estimator_j2_minus_two_body_under_truth_two_body",
    "estimator_j2_minus_two_body_under_truth_j2",
    "difference_in_differences_interaction",
)


def _expected_profiles() -> tuple[AssuranceValidationProfile, ...]:
    return tuple(AssuranceValidationProfile(value) for value in MODEL_FORM_FACTORIAL_PROFILE_VALUES)


class ModelFormFactorialProtocol(AstroModel):
    protocol_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    calibration_protocol_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    assurance_scenario: str = Field(min_length=1)
    calibration_evidence: str = Field(min_length=1)
    tracking_duration_s: FiniteFloat = Field(gt=0.0)
    correction_elapsed_s: FiniteFloat = Field(gt=0.0)
    verification_elapsed_s: FiniteFloat = Field(gt=0.0)
    diagnostic_maximum_component_delta_v_km_s: FiniteFloat = Field(gt=0.0)
    diagnostic_maximum_total_delta_v_km_s: FiniteFloat = Field(gt=0.0)
    realizations: tuple[AssuranceValidationRealization, ...] = Field(min_length=1)
    profiles: tuple[AssuranceValidationProfile, ...] = Field(default_factory=_expected_profiles)
    claim_boundary: Literal[
        "model_form_factorial_simulation_sensitivity_not_physical_truth_or_flight_authority"
    ] = "model_form_factorial_simulation_sensitivity_not_physical_truth_or_flight_authority"
    source_path: str | None = Field(default=None, exclude=True)
    source_digest: str | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def protocol_is_complete_and_factorial(self) -> ModelFormFactorialProtocol:
        if self.profiles != _expected_profiles():
            raise ValueError("model-form factorial profiles must use the exact required order")
        case_ids = [realization.case_id for realization in self.realizations]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("validation realization case ids must be unique")
        noise_seeds = [
            realization.input_overrides.tracking_noise_seed for realization in self.realizations
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


class ModelFormFactorialCellResult(AstroModel):
    case_id: str = Field(min_length=1)
    profile_result: AssuranceValidationProfileResult

    @model_validator(mode="after")
    def force_roles_match_profile(self) -> ModelFormFactorialCellResult:
        expected = {
            "matched_two_body": (ForceModelName.TWO_BODY, ForceModelName.TWO_BODY),
            "truth_two_body_estimator_j2": (ForceModelName.TWO_BODY, ForceModelName.J2),
            "truth_j2_estimator_two_body": (ForceModelName.J2, ForceModelName.TWO_BODY),
            "matched_j2": (ForceModelName.J2, ForceModelName.J2),
        }[self.profile_result.profile.value]
        actual = (
            self.profile_result.truth_force_model,
            self.profile_result.estimation_force_model,
        )
        if actual != expected:
            raise ValueError("factorial cell force roles must match its profile")
        return self


class ModelFormFactorialContrastResult(AstroModel):
    contrast_id: ModelFormContrastId
    complete: bool
    metric_deltas: dict[str, FiniteFloat] = Field(default_factory=dict)

    @model_validator(mode="after")
    def incomplete_contrast_has_no_values(self) -> ModelFormFactorialContrastResult:
        if not self.complete and self.metric_deltas:
            raise ValueError("incomplete contrast cannot contain metric deltas")
        return self


class ModelFormFactorialRealizationResult(AstroModel):
    case_id: str = Field(min_length=1)
    realization: AssuranceValidationRealization
    cells: tuple[ModelFormFactorialCellResult, ...] = Field(min_length=4, max_length=4)
    contrasts: tuple[ModelFormFactorialContrastResult, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def cells_and_contrasts_are_exact(self) -> ModelFormFactorialRealizationResult:
        if self.case_id != self.realization.case_id:
            raise ValueError("factorial case id must match realization")
        if any(cell.case_id != self.case_id for cell in self.cells):
            raise ValueError("factorial cell case ids must match realization")
        if tuple(cell.profile_result.profile for cell in self.cells) != _expected_profiles():
            raise ValueError("factorial cells must use the exact required profile order")
        if tuple(contrast.contrast_id for contrast in self.contrasts) != (
            MODEL_FORM_FACTORIAL_CONTRAST_IDS
        ):
            raise ValueError("factorial contrasts must use the exact required order")
        return self


class ModelFormMetricContrastSummary(AstroModel):
    count: int = Field(gt=0)
    minimum: FiniteFloat
    median: FiniteFloat
    maximum: FiniteFloat


class ModelFormProfileCount(AstroModel):
    requested: int = Field(ge=0)
    completed: int = Field(ge=0)
    passed: int = Field(ge=0)


class ModelFormContrastSummary(AstroModel):
    requested: int = Field(ge=0)
    complete: int = Field(ge=0)
    metric_deltas: dict[str, ModelFormMetricContrastSummary] = Field(default_factory=dict)


class ModelFormFactorialSummary(AstroModel):
    requested_realizations: int = Field(gt=0)
    profile_counts: dict[AssuranceValidationProfile, ModelFormProfileCount]
    contrast_summaries: dict[ModelFormContrastId, ModelFormContrastSummary]
    denominator_policy: Literal[
        "counts_only_profiles_and_contrasts_unpooled_no_probability_estimate"
    ] = "counts_only_profiles_and_contrasts_unpooled_no_probability_estimate"


class ModelFormFactorialResult(AstroModel):
    protocol_id: str = Field(min_length=1)
    calibration_protocol_id: str = Field(min_length=1)
    workflow: Literal["model_form_factorial_assurance_validation_v1"] = (
        "model_form_factorial_assurance_validation_v1"
    )
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
    realizations: tuple[ModelFormFactorialRealizationResult, ...] = Field(min_length=1)
    summary: ModelFormFactorialSummary
    claim_boundary: Literal[
        "model_form_factorial_simulation_sensitivity_not_physical_truth_or_flight_authority"
    ] = "model_form_factorial_simulation_sensitivity_not_physical_truth_or_flight_authority"
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def summary_matches_realizations(self) -> ModelFormFactorialResult:
        if self.summary != summarize_model_form_factorial(self.realizations):
            raise ValueError("factorial summary must match realization outcomes")
        return self


def summarize_model_form_factorial(
    realizations: tuple[ModelFormFactorialRealizationResult, ...],
) -> ModelFormFactorialSummary:
    profile_counts: dict[AssuranceValidationProfile, ModelFormProfileCount] = {}
    for profile in _expected_profiles():
        results = [
            cell.profile_result
            for realization in realizations
            for cell in realization.cells
            if cell.profile_result.profile is profile
        ]
        profile_counts[profile] = ModelFormProfileCount(
            requested=len(realizations),
            completed=sum(result.status is AssuranceValidationStatus.SUCCESS for result in results),
            passed=sum(result.passed is True for result in results),
        )

    contrast_summaries: dict[ModelFormContrastId, ModelFormContrastSummary] = {}
    for contrast_id in MODEL_FORM_FACTORIAL_CONTRAST_IDS:
        contrasts = [
            contrast
            for realization in realizations
            for contrast in realization.contrasts
            if contrast.contrast_id == contrast_id
        ]
        metric_names = sorted(
            {
                metric
                for contrast in contrasts
                if contrast.complete
                for metric in contrast.metric_deltas
            }
        )
        metric_deltas: dict[str, ModelFormMetricContrastSummary] = {}
        for metric in metric_names:
            values = [
                float(contrast.metric_deltas[metric])
                for contrast in contrasts
                if contrast.complete and metric in contrast.metric_deltas
            ]
            metric_deltas[metric] = ModelFormMetricContrastSummary(
                count=len(values),
                minimum=min(values),
                median=median(values),
                maximum=max(values),
            )
        contrast_summaries[contrast_id] = ModelFormContrastSummary(
            requested=len(realizations),
            complete=sum(contrast.complete for contrast in contrasts),
            metric_deltas=metric_deltas,
        )
    return ModelFormFactorialSummary(
        requested_realizations=len(realizations),
        profile_counts=profile_counts,
        contrast_summaries=contrast_summaries,
    )
