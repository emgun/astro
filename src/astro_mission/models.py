from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, FiniteFloat, model_validator

from astro_core.models import AstroModel, ForceModelName, Maneuver, Scenario, Trajectory
from astro_launch.models import LaunchTrajectory
from astro_reentry.models import ReentryResult, ReentryScenario
from astro_twin.models import DigitalTwinResult

LifecyclePhaseName = Literal["launch", "operations", "digital_twin", "deorbit", "reentry"]


class LifecycleStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class OrbitPhaseConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)
    spacecraft_name: str = Field(min_length=1)
    reference_area_m2: FiniteFloat = Field(gt=0.0)
    drag_coefficient: FiniteFloat = Field(ge=0.0, le=10.0)
    reflectivity_coefficient: FiniteFloat = Field(ge=0.0, le=5.0)
    gravity: ForceModelName = ForceModelName.TWO_BODY

    @model_validator(mode="after")
    def duration_must_align_with_step(self) -> OrbitPhaseConfig:
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1.0e-9:
            raise ValueError("orbit duration_s must be an integer multiple of step_s")
        return self


class DeorbitPhaseConfig(AstroModel):
    delta_v_km_s: FiniteFloat = Field(gt=0.0)
    specific_impulse_s: FiniteFloat = Field(gt=0.0)
    coast_duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)
    entry_interface_altitude_km: FiniteFloat = Field(gt=0.0)
    interface_tolerance_km: FiniteFloat = Field(gt=0.0)
    minimum_propellant_reserve_kg: FiniteFloat = Field(ge=0.0, default=0.0)

    @model_validator(mode="after")
    def duration_must_align_with_step(self) -> DeorbitPhaseConfig:
        steps = self.coast_duration_s / self.step_s
        if abs(steps - round(steps)) > 1.0e-9:
            raise ValueError("deorbit coast_duration_s must be an integer multiple of step_s")
        return self


class MissionLifecycleInputOverrides(AstroModel):
    launch_upper_stage_thrust_n: FiniteFloat | None = Field(default=None, gt=0.0)
    spacecraft_wet_mass_kg: FiniteFloat | None = Field(default=None, gt=0.0)
    twin_solar_array_efficiency: FiniteFloat | None = Field(default=None, gt=0.0, le=1.0)
    reentry_atmosphere_density_scale_factor: FiniteFloat | None = Field(
        default=None,
        gt=0.0,
    )
    reentry_vehicle_drag_coefficient: FiniteFloat | None = Field(
        default=None,
        gt=0.0,
        le=10.0,
    )


class MissionLifecycleScenario(AstroModel):
    scenario_id: str = Field(min_length=1)
    description: str = ""
    launch_scenario: str = Field(min_length=1)
    launch_backend: str = Field(min_length=1, default="local")
    orbit: OrbitPhaseConfig
    twin_scenario: str = Field(min_length=1)
    deorbit: DeorbitPhaseConfig
    reentry_scenario: str = Field(min_length=1)
    reentry_backend: str = Field(min_length=1, default="local")
    input_overrides: MissionLifecycleInputOverrides | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LifecycleContinuityCheck(AstroModel):
    name: str = Field(min_length=1)
    upstream_phase: LifecyclePhaseName
    downstream_phase: LifecyclePhaseName
    error: FiniteFloat = Field(ge=0.0)
    tolerance: FiniteFloat = Field(ge=0.0)
    unit: str = Field(min_length=1)
    passed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def passed_must_match_error(self) -> LifecycleContinuityCheck:
        if self.passed != (self.error <= self.tolerance):
            raise ValueError("continuity passed must match error <= tolerance")
        return self


class LifecycleContinuityReport(AstroModel):
    checks: tuple[LifecycleContinuityCheck, ...] = Field(min_length=1)
    all_passed: bool

    @model_validator(mode="after")
    def all_passed_must_match_checks(self) -> LifecycleContinuityReport:
        if self.all_passed != all(check.passed for check in self.checks):
            raise ValueError("continuity all_passed must match checks")
        return self


class MissionLifecycleMargin(AstroModel):
    phase: LifecyclePhaseName
    name: str = Field(min_length=1)
    value: FiniteFloat
    threshold: FiniteFloat
    margin: FiniteFloat
    unit: str = Field(min_length=1)
    status: LifecycleStatus


class MissionLifecycleMarginReport(AstroModel):
    margins: tuple[MissionLifecycleMargin, ...] = Field(min_length=1)
    limiting_margin: MissionLifecycleMargin
    overall_status: LifecycleStatus

    @model_validator(mode="after")
    def status_must_match_margins(self) -> MissionLifecycleMarginReport:
        severity = {
            LifecycleStatus.PASS: 0,
            LifecycleStatus.WARN: 1,
            LifecycleStatus.FAIL: 2,
        }
        expected = max(self.margins, key=lambda margin: severity[margin.status]).status
        if self.overall_status != expected:
            raise ValueError("lifecycle overall_status must match margin severity")
        if self.limiting_margin not in self.margins:
            raise ValueError("lifecycle limiting_margin must be present in margins")
        return self


class LifecyclePhaseManifestEntry(AstroModel):
    sequence: int = Field(ge=1)
    phase: LifecyclePhaseName
    product_type: str = Field(min_length=1)
    artifact_name: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    start_epoch: datetime
    end_epoch: datetime
    sample_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> LifecyclePhaseManifestEntry:
        if self.end_epoch < self.start_epoch:
            raise ValueError("manifest end_epoch must not precede start_epoch")
        return self


class MissionLifecycleManifest(AstroModel):
    scenario_id: str = Field(min_length=1)
    workflow: str = "mission_lifecycle_v1"
    entries: tuple[LifecyclePhaseManifestEntry, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def entries_must_be_ordered(self) -> MissionLifecycleManifest:
        sequences = [entry.sequence for entry in self.entries]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("manifest entries must use contiguous one-based sequence values")
        return self


class MissionLifecycleResult(AstroModel):
    scenario_id: str = Field(min_length=1)
    workflow: str = "mission_lifecycle_v1"
    launch_trajectory: LaunchTrajectory
    orbit_scenario: Scenario
    operations_trajectory: Trajectory
    digital_twin: DigitalTwinResult
    deorbit_maneuver: Maneuver
    deorbit_scenario: Scenario
    deorbit_trajectory: Trajectory
    reentry_scenario: ReentryScenario
    reentry_result: ReentryResult
    continuity_report: LifecycleContinuityReport
    margin_report: MissionLifecycleMarginReport
    manifest: MissionLifecycleManifest
    passed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def passed_must_match_reports(self) -> MissionLifecycleResult:
        expected = (
            self.continuity_report.all_passed
            and self.margin_report.overall_status != LifecycleStatus.FAIL
        )
        if self.passed != expected:
            raise ValueError("lifecycle passed must match continuity and margin reports")
        return self
