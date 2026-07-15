from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from math import atan2, degrees, isclose, sqrt
from statistics import median
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from astro_assurance.models import (
    InsertionDispersion,
    MissionAssuranceCase,
    MissionAssuranceInputOverrides,
)
from astro_core.models import AstroModel, Body, ForceModelName, Frame, MeasurementType, TimeScale


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


class AssuranceCalibrationDerivation(AstroModel):
    method: Literal[
        "residual_summary_envelope",
        "execution_residual_envelope",
        "symmetric_covariance_sigma_envelope",
    ]
    sigma_multiplier: FiniteFloat | None = Field(default=None, gt=0.0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def multiplier_matches_method(self) -> AssuranceCalibrationDerivation:
        covariance_method = self.method == "symmetric_covariance_sigma_envelope"
        if covariance_method != (self.sigma_multiplier is not None):
            raise ValueError("only covariance sigma derivations require sigma_multiplier")
        return self


class StationResidualEvidence(AstroModel):
    kind: Literal["station_residuals"] = "station_residuals"
    evidence_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    source_ids: tuple[str, ...] = Field(min_length=1)
    authority: AssuranceCalibrationAuthority
    assurance_scenario_id: str = Field(min_length=1)
    tracking_scenario_id: str = Field(min_length=1)
    station_id: str = Field(min_length=1)
    measurement_type: Literal[MeasurementType.RANGE, MeasurementType.RANGE_RATE]
    unit: Literal["km", "km/s"]
    band: str = Field(min_length=1)
    tracking_mode: str = Field(min_length=1)
    integration_time_s: FiniteFloat = Field(gt=0.0)
    arc_start: datetime
    arc_end: datetime
    sample_count: int = Field(ge=2)
    mean_residual: FiniteFloat
    sample_standard_deviation: FiniteFloat = Field(ge=0.0)
    rms: FiniteFloat = Field(ge=0.0)
    data_selection_policy: str = Field(min_length=1)
    outlier_policy: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def residual_semantics_are_consistent(self) -> StationResidualEvidence:
        expected_unit = (
            "km" if self.measurement_type is MeasurementType.RANGE else "km/s"
        )
        if self.unit != expected_unit:
            raise ValueError("station residual unit does not match measurement type")
        if self.arc_start.tzinfo is None or self.arc_end.tzinfo is None:
            raise ValueError("station residual arc epochs must include timezone information")
        if self.arc_end <= self.arc_start:
            raise ValueError("station residual arc end must follow its start")
        if float(self.rms) + 1e-15 < abs(float(self.mean_residual)):
            raise ValueError("station residual RMS must be at least the absolute mean")
        return self


class PropulsionExecutionResidualEvidence(AstroModel):
    kind: Literal["propulsion_execution_residuals"] = "propulsion_execution_residuals"
    evidence_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    source_ids: tuple[str, ...] = Field(min_length=1)
    authority: AssuranceCalibrationAuthority
    assurance_scenario_id: str = Field(min_length=1)
    maneuver_id: str = Field(min_length=1)
    propulsion_class: str = Field(min_length=1)
    vector_frame: Frame
    commanded_epoch: datetime
    achieved_epoch: datetime
    commanded_delta_v_km_s: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    achieved_delta_v_km_s: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    timing_residual_s: FiniteFloat
    magnitude_scale: FiniteFloat = Field(gt=0.0)
    pointing_basis: Literal[
        "axis_1_cross_command_with_least_aligned_inertial_axis_then_axis_2"
    ]
    pointing_residual_1_deg: FiniteFloat
    pointing_residual_2_deg: FiniteFloat
    reconstruction_method: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def execution_residuals_match_bound_vectors(
        self,
    ) -> PropulsionExecutionResidualEvidence:
        if self.commanded_epoch.tzinfo is None or self.achieved_epoch.tzinfo is None:
            raise ValueError("propulsion epochs must include timezone information")
        expected_timing = (self.achieved_epoch - self.commanded_epoch).total_seconds()
        if not isclose(float(self.timing_residual_s), expected_timing, abs_tol=1e-9):
            raise ValueError("timing residual must equal achieved minus commanded epoch")
        commanded_norm = sqrt(sum(float(value) ** 2 for value in self.commanded_delta_v_km_s))
        achieved_norm = sqrt(sum(float(value) ** 2 for value in self.achieved_delta_v_km_s))
        if commanded_norm <= 0.0:
            raise ValueError("commanded delta-v must have non-zero magnitude")
        if not isclose(
            float(self.magnitude_scale), achieved_norm / commanded_norm, rel_tol=1e-9
        ):
            raise ValueError("magnitude scale does not match the bound delta-v vectors")
        commanded = np.asarray(self.commanded_delta_v_km_s, dtype=np.float64)
        achieved = np.asarray(self.achieved_delta_v_km_s, dtype=np.float64)
        direction = commanded / commanded_norm
        achieved_direction = achieved / achieved_norm
        reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(direction)))]
        axis_1 = np.cross(direction, reference)
        axis_1 /= np.linalg.norm(axis_1)
        axis_2 = np.cross(direction, axis_1)
        denominator = float(np.dot(achieved_direction, direction))
        expected_pointing_1 = degrees(
            atan2(float(np.dot(achieved_direction, axis_1)), denominator)
        )
        expected_pointing_2 = degrees(
            atan2(float(np.dot(achieved_direction, axis_2)), denominator)
        )
        if not isclose(
            float(self.pointing_residual_1_deg), expected_pointing_1, abs_tol=1e-9
        ) or not isclose(
            float(self.pointing_residual_2_deg), expected_pointing_2, abs_tol=1e-9
        ):
            raise ValueError("pointing residuals do not match the bound delta-v vectors")
        return self


class InsertionCovarianceEvidence(AstroModel):
    kind: Literal["insertion_covariance"] = "insertion_covariance"
    evidence_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    source_ids: tuple[str, ...] = Field(min_length=1)
    authority: AssuranceCalibrationAuthority
    assurance_scenario_id: str = Field(min_length=1)
    tracking_scenario_id: str = Field(min_length=1)
    epoch: datetime
    time_scale: TimeScale
    central_body: Body
    frame: Frame
    state_order: tuple[Literal["x", "y", "z", "vx", "vy", "vz"], ...]
    state_units: tuple[Literal["km", "km/s"], ...]
    covariance: tuple[tuple[FiniteFloat, ...], ...]
    confidence_convention: Literal["one_sigma_covariance"]
    population_definition: str = Field(min_length=1)
    launcher_configuration: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def covariance_semantics_are_complete(self) -> InsertionCovarianceEvidence:
        if self.epoch.tzinfo is None:
            raise ValueError("insertion covariance epoch must include timezone information")
        if self.state_order != ("x", "y", "z", "vx", "vy", "vz"):
            raise ValueError("insertion covariance state order must be x,y,z,vx,vy,vz")
        if self.state_units != ("km", "km", "km", "km/s", "km/s", "km/s"):
            raise ValueError("insertion covariance state units are invalid")
        if len(self.covariance) != 6 or any(len(row) != 6 for row in self.covariance):
            raise ValueError("insertion covariance must be 6x6")
        matrix = np.asarray(self.covariance, dtype=np.float64)
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
            raise ValueError("insertion covariance must be symmetric")
        if np.any(np.diag(matrix) <= 0.0):
            raise ValueError("insertion covariance diagonal must be positive")
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(float(np.max(np.abs(eigenvalues))), float(np.finfo(np.float64).tiny))
        tolerance = 100.0 * float(np.finfo(np.float64).eps) * scale
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError("insertion covariance must be positive semidefinite")
        correlations = np.asarray(self.correlation_matrix, dtype=np.float64)
        if not np.all(np.isfinite(correlations)) or np.any(np.abs(correlations) > 1.0 + 1e-12):
            raise ValueError("insertion covariance implies invalid correlations")
        return self

    @property
    def standard_deviations(self) -> tuple[float, ...]:
        return tuple(sqrt(float(self.covariance[index][index])) for index in range(6))

    @property
    def correlation_matrix(self) -> tuple[tuple[float, ...], ...]:
        deviations = self.standard_deviations
        return tuple(
            tuple(
                float(self.covariance[row][column])
                / (deviations[row] * deviations[column])
                for column in range(6)
            )
            for row in range(6)
        )


AssuranceCalibrationEvidence = Annotated[
    StationResidualEvidence
    | PropulsionExecutionResidualEvidence
    | InsertionCovarianceEvidence,
    Field(discriminator="kind"),
]


class AssuranceCalibrationBound(AstroModel):
    parameter: str = Field(min_length=1)
    minimum: FiniteFloat
    maximum: FiniteFloat
    unit: str = Field(min_length=1)
    authority: AssuranceCalibrationAuthority
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    derivation: AssuranceCalibrationDerivation | None = None
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
        calibrated = self.authority in {
            AssuranceCalibrationAuthority.MISSION_TEST_CALIBRATED,
            AssuranceCalibrationAuthority.FLIGHT_CALIBRATED,
        }
        if calibrated != bool(self.evidence_ids and self.derivation is not None):
            raise ValueError(
                "mission or flight calibrated bounds require evidence ids and derivation"
            )
        return self


class AssuranceValidationCalibrationManifest(AstroModel):
    calibration_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    protocol_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    sources: tuple[AssuranceCalibrationSource, ...] = Field(min_length=1)
    parameter_bounds: tuple[AssuranceCalibrationBound, ...] = Field(min_length=1)
    evidence_products: tuple[AssuranceCalibrationEvidence, ...] = Field(default_factory=tuple)
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
        evidence_ids = [evidence.evidence_id for evidence in self.evidence_products]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("calibration evidence ids must be unique")
        evidence_by_id = {
            evidence.evidence_id: evidence for evidence in self.evidence_products
        }
        for evidence in self.evidence_products:
            unknown = set(evidence.source_ids) - set(sources)
            if unknown:
                raise ValueError(
                    f"calibration evidence {evidence.evidence_id} references unknown sources: "
                    f"{sorted(unknown)}"
                )
            required_kind = {
                AssuranceCalibrationAuthority.MISSION_TEST_CALIBRATED: (
                    AssuranceCalibrationSourceKind.MISSION_TEST_DATA
                ),
                AssuranceCalibrationAuthority.FLIGHT_CALIBRATED: (
                    AssuranceCalibrationSourceKind.FLIGHT_DATA
                ),
            }.get(evidence.authority)
            if required_kind is not None and not any(
                sources[source_id].source_kind is required_kind
                for source_id in evidence.source_ids
            ):
                raise ValueError(
                    f"calibration evidence {evidence.evidence_id} lacks a source for its authority"
                )
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
            unknown_evidence = set(bound.evidence_ids) - set(evidence_by_id)
            if unknown_evidence:
                raise ValueError(
                    f"calibration bound {bound.parameter} references unknown evidence: "
                    f"{sorted(unknown_evidence)}"
                )
            for evidence_id in bound.evidence_ids:
                evidence = evidence_by_id[evidence_id]
                if evidence.authority is not bound.authority:
                    raise ValueError(
                        f"calibration bound {bound.parameter} evidence authority does not match"
                    )
                if not _evidence_applies_to_parameter(evidence, bound.parameter):
                    raise ValueError(
                        f"calibration evidence {evidence_id} does not apply to {bound.parameter}"
                    )
                if not set(evidence.source_ids).issubset(bound.source_ids):
                    raise ValueError(
                        f"calibration bound {bound.parameter} does not cite its evidence sources"
                    )
            if bound.evidence_ids:
                _validate_bound_derivation(bound, evidence_by_id)
        expected = derive_calibration_promotion_status(self.parameter_bounds)
        if self.promotion_status is not expected:
            raise ValueError("calibration promotion status must match bound authority")
        return self


def _evidence_applies_to_parameter(
    evidence: AssuranceCalibrationEvidence, parameter: str
) -> bool:
    if isinstance(evidence, StationResidualEvidence):
        expected = {
            MeasurementType.RANGE: {
                "input_overrides.tracking_range_sigma_km",
                "input_overrides.tracking_range_bias_km",
                "input_overrides.estimation_range_sigma_km",
                "input_overrides.estimation_range_bias_km",
            },
            MeasurementType.RANGE_RATE: {
                "input_overrides.tracking_range_rate_sigma_km_s",
                "input_overrides.tracking_range_rate_bias_km_s",
                "input_overrides.estimation_range_rate_sigma_km_s",
                "input_overrides.estimation_range_rate_bias_km_s",
            },
        }
        return parameter in expected[evidence.measurement_type]
    if isinstance(evidence, PropulsionExecutionResidualEvidence):
        return parameter in {
            "input_overrides.correction_execution_scale",
            "input_overrides.correction_execution_epoch_offset_s",
            "input_overrides.correction_execution_pointing_1_deg",
            "input_overrides.correction_execution_pointing_2_deg",
        }
    return parameter in {
        *(f"dispersion.position_delta_km[{index}]" for index in range(3)),
        *(f"dispersion.velocity_delta_km_s[{index}]" for index in range(3)),
    }


def _validate_bound_derivation(
    bound: AssuranceCalibrationBound,
    evidence_by_id: dict[str, AssuranceCalibrationEvidence],
) -> None:
    if bound.derivation is None:
        raise ValueError(f"calibration bound {bound.parameter} lacks a derivation")
    evidence = [evidence_by_id[evidence_id] for evidence_id in bound.evidence_ids]
    first = evidence[0]
    expected_unit: str
    if isinstance(first, StationResidualEvidence):
        if bound.derivation.method != "residual_summary_envelope":
            raise ValueError(f"calibration bound {bound.parameter} has wrong derivation method")
        expected_unit = first.unit
        values = [
            float(item.sample_standard_deviation)
            if "sigma" in bound.parameter
            else float(item.mean_residual)
            for item in evidence
            if isinstance(item, StationResidualEvidence)
        ]
    elif isinstance(first, PropulsionExecutionResidualEvidence):
        if bound.derivation.method != "execution_residual_envelope":
            raise ValueError(f"calibration bound {bound.parameter} has wrong derivation method")
        field, expected_unit = {
            "input_overrides.correction_execution_scale": ("magnitude_scale", "ratio"),
            "input_overrides.correction_execution_epoch_offset_s": ("timing_residual_s", "s"),
            "input_overrides.correction_execution_pointing_1_deg": (
                "pointing_residual_1_deg",
                "deg",
            ),
            "input_overrides.correction_execution_pointing_2_deg": (
                "pointing_residual_2_deg",
                "deg",
            ),
        }[bound.parameter]
        values = [
            float(getattr(item, field))
            for item in evidence
            if isinstance(item, PropulsionExecutionResidualEvidence)
        ]
    else:
        if bound.derivation.method != "symmetric_covariance_sigma_envelope":
            raise ValueError(f"calibration bound {bound.parameter} has wrong derivation method")
        index = int(bound.parameter.rsplit("[", 1)[1][:-1])
        expected_unit = "km" if "position_delta" in bound.parameter else "km/s"
        multiplier = float(bound.derivation.sigma_multiplier or 0.0)
        magnitudes = [
            item.standard_deviations[index] * multiplier
            for item in evidence
            if isinstance(item, InsertionCovarianceEvidence)
        ]
        values = [-max(magnitudes), max(magnitudes)]
    if bound.unit != expected_unit:
        raise ValueError(f"calibration bound {bound.parameter} unit does not match its evidence")
    if not isclose(float(bound.minimum), min(values), rel_tol=1e-9, abs_tol=1e-15) or not isclose(
        float(bound.maximum), max(values), rel_tol=1e-9, abs_tol=1e-15
    ):
        raise ValueError(
            f"calibration bound {bound.parameter} does not equal its evidence-derived envelope"
        )


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
