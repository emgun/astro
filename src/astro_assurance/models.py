from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from astro_core.models import (
    AstroModel,
    EstimateResult,
    ForceModelName,
    Maneuver,
    MeasurementRecord,
    Scenario,
    Trajectory,
    Vector3,
    _integer_input_must_be_int,
)
from astro_launch.models import LaunchTrajectory
from astro_twin.models import DigitalTwinResult

_ASSURANCE_INPUT_ROLES = {
    "assurance_scenario",
    "launch_scenario",
    "tracking_scenario",
    "twin_scenario",
}
_ASSURANCE_ARTIFACT_NAMES = {
    "launch.json",
    "measurements.json",
    "estimate.json",
    "candidate-maneuver.json",
    "nominal-trajectory.json",
    "truth-trajectory.json",
    "estimated-corrected-trajectory.json",
    "truth-corrected-trajectory.json",
    "corrected-digital-twin.json",
    "nominal-scenario.yaml",
    "truth-scenario.yaml",
    "estimated-corrected-scenario.yaml",
    "truth-corrected-scenario.yaml",
    "continuity-report.json",
    "margin-report.json",
    "decision.json",
}


class AssuranceStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class InsertionDispersion(AstroModel):
    position_delta_km: Vector3
    velocity_delta_km_s: Vector3

    @model_validator(mode="after")
    def dispersion_must_be_nonzero(self) -> InsertionDispersion:
        if not any(self.position_delta_km) and not any(self.velocity_delta_km_s):
            raise ValueError("insertion dispersion must be nonzero")
        return self


class CorrectionTargetingConfig(AstroModel):
    correction_elapsed_s: FiniteFloat = Field(gt=0.0)
    verification_elapsed_s: FiniteFloat = Field(gt=0.0)
    maximum_component_delta_v_km_s: FiniteFloat = Field(gt=0.0)
    maximum_total_delta_v_km_s: FiniteFloat = Field(gt=0.0)
    position_scale_km: FiniteFloat = Field(gt=0.0, default=1.0)
    velocity_scale_km_s: FiniteFloat = Field(gt=0.0, default=0.001)
    specific_impulse_s: FiniteFloat = Field(gt=0.0, default=300.0)

    @model_validator(mode="after")
    def verification_must_follow_correction(self) -> CorrectionTargetingConfig:
        if self.verification_elapsed_s <= self.correction_elapsed_s:
            raise ValueError("verification_elapsed_s must follow correction_elapsed_s")
        return self


class AssuranceRequirements(AstroModel):
    maximum_od_position_error_km: FiniteFloat = Field(gt=0.0)
    maximum_od_velocity_error_km_s: FiniteFloat = Field(gt=0.0)
    maximum_truth_recovery_position_error_km: FiniteFloat = Field(gt=0.0)
    maximum_truth_recovery_velocity_error_km_s: FiniteFloat = Field(gt=0.0)
    minimum_position_error_reduction_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    minimum_propellant_reserve_kg: FiniteFloat = Field(ge=0.0)
    minimum_battery_soc_fraction: FiniteFloat = Field(ge=0.0, le=1.0)


class AssuranceThermalNodeInputOverride(AstroModel):
    node_name: str = Field(min_length=1)
    emissivity: FiniteFloat | None = Field(default=None, gt=0.0, le=1.0)
    internal_heat_fraction: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def at_least_one_field_must_be_set(self) -> AssuranceThermalNodeInputOverride:
        if self.emissivity is None and self.internal_heat_fraction is None:
            raise ValueError("thermal node override must set at least one field")
        return self


class MissionAssuranceInputOverrides(AstroModel):
    tracking_duration_s: FiniteFloat | None = Field(default=None, gt=0.0)
    tracking_range_sigma_km: FiniteFloat | None = Field(default=None, gt=0.0)
    tracking_range_rate_sigma_km_s: FiniteFloat | None = Field(default=None, gt=0.0)
    tracking_noise_seed: int | None = None
    tracking_range_bias_km: FiniteFloat | None = None
    tracking_range_rate_bias_km_s: FiniteFloat | None = None
    estimation_range_sigma_km: FiniteFloat | None = Field(default=None, gt=0.0)
    estimation_range_rate_sigma_km_s: FiniteFloat | None = Field(default=None, gt=0.0)
    estimation_range_bias_km: FiniteFloat | None = None
    estimation_range_rate_bias_km_s: FiniteFloat | None = None
    truth_force_model: ForceModelName | None = None
    estimation_force_model: ForceModelName | None = None
    correction_execution_scale: FiniteFloat | None = Field(default=None, ge=0.0, le=2.0)
    correction_execution_epoch_offset_s: FiniteFloat | None = Field(
        default=None, ge=-300.0, le=300.0
    )
    correction_execution_pointing_1_deg: FiniteFloat | None = Field(default=None, ge=-10.0, le=10.0)
    correction_execution_pointing_2_deg: FiniteFloat | None = Field(default=None, ge=-10.0, le=10.0)
    twin_solar_array_efficiency: FiniteFloat | None = Field(default=None, gt=0.0, le=1.0)
    twin_battery_capacity_wh: FiniteFloat | None = Field(default=None, gt=0.0)
    twin_thermal_node_overrides: tuple[AssuranceThermalNodeInputOverride, ...] = Field(
        default_factory=tuple
    )

    @field_validator("tracking_noise_seed", mode="before")
    @classmethod
    def tracking_seed_must_be_exact_integer(cls, value: Any) -> Any:
        if value is None:
            return value
        return _integer_input_must_be_int(value, "Mission assurance tracking noise seed")

    @model_validator(mode="after")
    def thermal_node_overrides_must_be_unique(self) -> MissionAssuranceInputOverrides:
        names = [override.node_name for override in self.twin_thermal_node_overrides]
        if len(set(names)) != len(names):
            raise ValueError("thermal node overrides must use unique node names")
        local_models = {ForceModelName.TWO_BODY, ForceModelName.J2, None}
        if self.truth_force_model not in local_models:
            raise ValueError("truth force model override must be two_body or j2")
        if self.estimation_force_model not in local_models:
            raise ValueError("estimation force model override must be two_body or j2")
        return self


class PostLaunchAssuranceScenario(AstroModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    description: str = ""
    launch_scenario: str = Field(min_length=1)
    launch_backend: str = Field(min_length=1, default="local")
    tracking_scenario: str = Field(min_length=1)
    tracking_backend: Literal["local"] = "local"
    twin_scenario: str = Field(min_length=1)
    dispersion: InsertionDispersion
    correction: CorrectionTargetingConfig
    requirements: AssuranceRequirements
    input_overrides: MissionAssuranceInputOverrides | None = None
    source_path: str | None = Field(default=None, exclude=True)
    source_digest: str | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssuranceContinuityCheck(AstroModel):
    name: str = Field(min_length=1)
    upstream_phase: str = Field(min_length=1)
    downstream_phase: str = Field(min_length=1)
    error: FiniteFloat = Field(ge=0.0)
    tolerance: FiniteFloat = Field(ge=0.0)
    unit: str = Field(min_length=1)
    passed: bool

    @model_validator(mode="after")
    def passed_must_match_error(self) -> AssuranceContinuityCheck:
        if self.passed != (self.error <= self.tolerance):
            raise ValueError("continuity passed must match error <= tolerance")
        return self


class AssuranceContinuityReport(AstroModel):
    checks: tuple[AssuranceContinuityCheck, ...] = Field(min_length=1)
    all_passed: bool

    @model_validator(mode="after")
    def all_passed_must_match_checks(self) -> AssuranceContinuityReport:
        if self.all_passed != all(check.passed for check in self.checks):
            raise ValueError("continuity all_passed must match checks")
        return self


class AssuranceMargin(AstroModel):
    name: str = Field(min_length=1)
    value: FiniteFloat
    threshold: FiniteFloat
    margin: FiniteFloat
    normalized_margin: FiniteFloat
    unit: str = Field(min_length=1)
    status: AssuranceStatus
    evidence_scope: str = Field(min_length=1)


class AssuranceMarginReport(AstroModel):
    margins: tuple[AssuranceMargin, ...] = Field(min_length=1)
    limiting_margin: AssuranceMargin
    overall_status: AssuranceStatus

    @model_validator(mode="after")
    def report_must_match_margins(self) -> AssuranceMarginReport:
        severity = {
            AssuranceStatus.PASS: 0,
            AssuranceStatus.WARN: 1,
            AssuranceStatus.FAIL: 2,
        }
        expected_status = max(self.margins, key=lambda item: severity[item.status]).status
        if self.overall_status != expected_status:
            raise ValueError("assurance overall_status must match margin severity")
        if self.limiting_margin not in self.margins:
            raise ValueError("assurance limiting_margin must be present in margins")
        return self


class AssuranceManifestEntry(AstroModel):
    sequence: int = Field(ge=1)
    phase: str = Field(min_length=1)
    product_type: str = Field(min_length=1)
    artifact_name: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: str = Field(min_length=1)
    digest_scope: Literal["canonical_json_payload"] = "canonical_json_payload"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssuranceInputReference(AstroModel):
    role: Literal[
        "assurance_scenario",
        "launch_scenario",
        "tracking_scenario",
        "twin_scenario",
    ]
    path: str = Field(min_length=1)
    file_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    digest_scope: Literal["file_bytes"] = "file_bytes"


class AssuranceManifest(AstroModel):
    scenario_id: str = Field(min_length=1)
    workflow: str = "post_launch_mission_assurance_v1"
    inputs: tuple[AssuranceInputReference, ...] = Field(min_length=4, max_length=4)
    entries: tuple[AssuranceManifestEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def entries_must_be_ordered(self) -> AssuranceManifest:
        if [entry.sequence for entry in self.entries] != list(range(1, len(self.entries) + 1)):
            raise ValueError("assurance manifest entries must use contiguous one-based sequence")
        input_roles = [reference.role for reference in self.inputs]
        if len(input_roles) != len(set(input_roles)) or set(input_roles) != _ASSURANCE_INPUT_ROLES:
            raise ValueError("assurance manifest must contain the fixed unique input roles")
        artifact_names = [entry.artifact_name for entry in self.entries]
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("assurance manifest artifact names must be unique")
        if set(artifact_names) != _ASSURANCE_ARTIFACT_NAMES:
            raise ValueError("assurance manifest must contain the fixed workflow artifact set")
        return self


class MissionAssuranceCase(AstroModel):
    scenario_id: str = Field(min_length=1)
    workflow: str = "post_launch_mission_assurance_v1"
    launch_trajectory: LaunchTrajectory
    nominal_scenario: Scenario
    truth_scenario: Scenario
    nominal_trajectory: Trajectory
    truth_trajectory: Trajectory
    measurements: tuple[MeasurementRecord, ...]
    estimate: EstimateResult
    correction_maneuver: Maneuver
    estimated_corrected_scenario: Scenario
    truth_corrected_scenario: Scenario
    estimated_corrected_trajectory: Trajectory
    truth_corrected_trajectory: Trajectory
    corrected_digital_twin: DigitalTwinResult
    continuity_report: AssuranceContinuityReport
    margin_report: AssuranceMarginReport
    manifest: AssuranceManifest
    passed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def passed_must_match_reports(self) -> MissionAssuranceCase:
        expected = (
            self.continuity_report.all_passed
            and self.margin_report.overall_status is not AssuranceStatus.FAIL
        )
        if self.passed != expected:
            raise ValueError("assurance passed must match continuity and margin reports")
        return self
