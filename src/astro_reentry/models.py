from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from astro_core.models import (
    AstroModel,
    Body,
    TimeScale,
    _datetime_input_must_be_datetime_or_string,
    _datetime_must_be_aware,
    _integer_input_must_be_int,
    _numeric_scalar_input_must_be_number,
)

ReentryGuidanceMode = Literal[
    "ballistic",
    "constant_bank",
    "bank_schedule",
    "target_tracking",
]
ReentryEventType = Literal[
    "entry_interface",
    "guidance_bank_reversal",
    "peak_heating",
    "peak_dynamic_pressure",
    "peak_deceleration",
    "terminal",
]


class ReentryMarginStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ReentryVehicle(AstroModel):
    name: str = Field(min_length=1)
    mass_kg: FiniteFloat = Field(gt=0.0)
    reference_area_m2: FiniteFloat = Field(gt=0.0)
    drag_coefficient: FiniteFloat = Field(gt=0.0, le=10.0)
    lift_to_drag_ratio: FiniteFloat = Field(ge=0.0, le=3.0, default=0.0)
    nose_radius_m: FiniteFloat = Field(gt=0.0)

    @field_validator(
        "mass_kg",
        "reference_area_m2",
        "drag_coefficient",
        "lift_to_drag_ratio",
        "nose_radius_m",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry vehicle scalar")

    @property
    def ballistic_coefficient_kg_m2(self) -> float:
        return float(self.mass_kg / (self.drag_coefficient * self.reference_area_m2))


class ReentryInitialState(AstroModel):
    epoch: datetime
    altitude_km: FiniteFloat = Field(ge=0.0)
    velocity_km_s: FiniteFloat = Field(gt=0.0)
    flight_path_angle_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    heading_deg: FiniteFloat = Field(ge=0.0, lt=360.0)
    latitude_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat = Field(ge=-180.0, le=180.0)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "Reentry initial epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "Reentry initial epoch")

    @field_validator(
        "altitude_km",
        "velocity_km_s",
        "flight_path_angle_deg",
        "heading_deg",
        "latitude_deg",
        "longitude_deg",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry initial-state scalar")


class ReentryAtmosphereConfig(AstroModel):
    model: Literal["none", "exponential"] = "exponential"
    reference_density_kg_m3: FiniteFloat = Field(gt=0.0, default=1.225)
    reference_altitude_m: FiniteFloat = Field(ge=0.0, default=0.0)
    scale_height_m: FiniteFloat = Field(gt=0.0, default=7200.0)
    density_scale_factor: FiniteFloat = Field(gt=0.0, default=1.0)

    @field_validator(
        "reference_density_kg_m3",
        "reference_altitude_m",
        "scale_height_m",
        "density_scale_factor",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry atmosphere scalar")


class AerothermalConfig(AstroModel):
    model: Literal["none", "sutton_graves"] = "sutton_graves"
    sutton_graves_coefficient: FiniteFloat = Field(gt=0.0, default=1.7415e-4)
    wall_emissivity: FiniteFloat = Field(gt=0.0, le=1.0, default=0.85)

    @field_validator("sutton_graves_coefficient", "wall_emissivity", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Aerothermal scalar")


class BankSchedulePoint(AstroModel):
    velocity_km_s: FiniteFloat = Field(gt=0.0)
    bank_angle_deg: FiniteFloat = Field(ge=-90.0, le=90.0)

    @field_validator("velocity_km_s", "bank_angle_deg", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Bank schedule scalar")


class ReentryGuidanceConfig(AstroModel):
    mode: ReentryGuidanceMode = "ballistic"
    bank_angle_deg: FiniteFloat = Field(ge=-90.0, le=90.0, default=0.0)
    bank_schedule: tuple[BankSchedulePoint, ...] = ()
    heading_deadband_deg: FiniteFloat = Field(ge=0.0, le=45.0, default=1.0)
    minimum_control_velocity_km_s: FiniteFloat = Field(gt=0.0, default=0.5)
    minimum_bank_reversal_interval_s: FiniteFloat = Field(ge=0.0, default=20.0)

    @field_validator(
        "bank_angle_deg",
        "heading_deadband_deg",
        "minimum_control_velocity_km_s",
        "minimum_bank_reversal_interval_s",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry guidance scalar")

    @model_validator(mode="after")
    def validate_schedule(self) -> ReentryGuidanceConfig:
        if self.mode in {"ballistic", "constant_bank"} and self.bank_schedule:
            raise ValueError(f"{self.mode} guidance does not accept bank_schedule points")
        if self.mode == "ballistic" and self.bank_angle_deg != 0.0:
            raise ValueError("ballistic guidance requires bank_angle_deg = 0")
        if self.mode in {"bank_schedule", "target_tracking"} and self.bank_angle_deg != 0.0:
            raise ValueError(f"{self.mode} guidance requires bank_angle_deg = 0")
        if self.mode in {"bank_schedule", "target_tracking"} and len(self.bank_schedule) < 2:
            raise ValueError(f"{self.mode} guidance requires at least two bank_schedule points")
        velocities = [point.velocity_km_s for point in self.bank_schedule]
        if not all(
            previous > next_velocity
            for previous, next_velocity in zip(velocities, velocities[1:], strict=False)
        ):
            raise ValueError("bank_schedule velocity_km_s values must be strictly decreasing")
        if self.mode == "target_tracking" and any(
            point.bank_angle_deg < 0.0 for point in self.bank_schedule
        ):
            raise ValueError("target_tracking bank_schedule angles must be non-negative magnitudes")
        return self


class ReentryTarget(AstroModel):
    name: str = Field(min_length=1)
    latitude_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat = Field(ge=-180.0, le=180.0)
    allowable_miss_km: FiniteFloat = Field(gt=0.0)

    @field_validator("latitude_deg", "longitude_deg", "allowable_miss_km", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry target scalar")


class ReentryLimits(AstroModel):
    maximum_dynamic_pressure_pa: FiniteFloat = Field(gt=0.0)
    maximum_deceleration_g: FiniteFloat = Field(gt=0.0)
    maximum_heat_rate_w_m2: FiniteFloat = Field(gt=0.0)
    maximum_heat_load_j_m2: FiniteFloat = Field(gt=0.0)

    @field_validator(
        "maximum_dynamic_pressure_pa",
        "maximum_deceleration_g",
        "maximum_heat_rate_w_m2",
        "maximum_heat_load_j_m2",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry limit scalar")


class ReentryPropagationConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)
    internal_step_s: FiniteFloat = Field(gt=0.0, default=0.25)
    termination_altitude_km: FiniteFloat = Field(ge=0.0, default=0.0)
    minimum_velocity_km_s: FiniteFloat = Field(gt=0.0, default=0.05)
    entry_interface_altitude_km: FiniteFloat = Field(gt=0.0, default=120.0)

    @field_validator(
        "duration_s",
        "step_s",
        "internal_step_s",
        "termination_altitude_km",
        "minimum_velocity_km_s",
        "entry_interface_altitude_km",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry propagation scalar")

    @model_validator(mode="after")
    def validate_steps_and_termination(self) -> ReentryPropagationConfig:
        if self.internal_step_s > self.step_s:
            raise ValueError("Reentry internal_step_s must be <= step_s")
        output_steps = self.duration_s / self.step_s
        if abs(output_steps - round(output_steps)) > 1.0e-9:
            raise ValueError("Reentry duration_s must be an integer multiple of step_s")
        internal_steps = self.step_s / self.internal_step_s
        if abs(internal_steps - round(internal_steps)) > 1.0e-9:
            raise ValueError("Reentry step_s must be an integer multiple of internal_step_s")
        if self.termination_altitude_km >= self.entry_interface_altitude_km:
            raise ValueError("termination_altitude_km must be below entry_interface_altitude_km")
        return self


class ReentryScenario(AstroModel):
    scenario_id: str = Field(min_length=1)
    description: str = ""
    body: Body = Body.EARTH
    time_scale: TimeScale = TimeScale.UTC
    initial_state: ReentryInitialState
    vehicle: ReentryVehicle
    atmosphere: ReentryAtmosphereConfig = Field(default_factory=ReentryAtmosphereConfig)
    aerothermal: AerothermalConfig = Field(default_factory=AerothermalConfig)
    guidance: ReentryGuidanceConfig = Field(default_factory=ReentryGuidanceConfig)
    target: ReentryTarget | None = None
    limits: ReentryLimits
    propagation: ReentryPropagationConfig
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_guidance_vehicle_and_target(self) -> ReentryScenario:
        if self.guidance.mode == "ballistic" and self.vehicle.lift_to_drag_ratio != 0.0:
            raise ValueError("ballistic guidance requires vehicle lift_to_drag_ratio = 0")
        if self.guidance.mode != "ballistic" and self.vehicle.lift_to_drag_ratio <= 0.0:
            raise ValueError("lifting guidance requires vehicle lift_to_drag_ratio > 0")
        if self.guidance.mode == "target_tracking" and self.target is None:
            raise ValueError("target_tracking guidance requires a target")
        if self.initial_state.altitude_km <= self.propagation.termination_altitude_km:
            raise ValueError("initial altitude must be above termination_altitude_km")
        if self.initial_state.velocity_km_s <= self.propagation.minimum_velocity_km_s:
            raise ValueError("initial velocity must be above minimum_velocity_km_s")
        return self


class ReentryEvent(AstroModel):
    event_type: ReentryEventType
    epoch: datetime
    time_s: FiniteFloat = Field(ge=0.0)
    altitude_km: FiniteFloat
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "Reentry event epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "Reentry event epoch")

    @field_validator("time_s", "altitude_km", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry event scalar")


class ReentrySample(AstroModel):
    epoch: datetime
    time_s: FiniteFloat = Field(ge=0.0)
    altitude_km: FiniteFloat
    latitude_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat = Field(ge=-180.0, le=180.0)
    downrange_km: FiniteFloat = Field(ge=0.0)
    velocity_km_s: FiniteFloat = Field(ge=0.0)
    flight_path_angle_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    heading_deg: FiniteFloat = Field(ge=0.0, lt=360.0)
    bank_angle_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    atmospheric_density_kg_m3: FiniteFloat = Field(ge=0.0)
    dynamic_pressure_pa: FiniteFloat = Field(ge=0.0)
    drag_acceleration_m_s2: FiniteFloat = Field(ge=0.0)
    lift_acceleration_m_s2: FiniteFloat = Field(ge=0.0)
    deceleration_g: FiniteFloat = Field(ge=0.0)
    convective_heat_rate_w_m2: FiniteFloat = Field(ge=0.0)
    heat_load_j_m2: FiniteFloat = Field(ge=0.0)
    radiative_equilibrium_temperature_k: FiniteFloat = Field(ge=0.0)
    range_to_target_km: FiniteFloat | None = Field(default=None, ge=0.0)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "Reentry sample epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "Reentry sample epoch")


class ReentryPeakMetric(AstroModel):
    value: FiniteFloat = Field(ge=0.0)
    unit: str = Field(min_length=1)
    time_s: FiniteFloat = Field(ge=0.0)
    altitude_km: FiniteFloat


class ReentryPeakSummary(AstroModel):
    dynamic_pressure: ReentryPeakMetric
    deceleration: ReentryPeakMetric
    heat_rate: ReentryPeakMetric
    total_heat_load_j_m2: FiniteFloat = Field(ge=0.0)


class ReentryTargetMiss(AstroModel):
    distance_km: FiniteFloat = Field(ge=0.0)
    latitude_error_deg: FiniteFloat
    longitude_error_deg: FiniteFloat


class ReentryMargin(AstroModel):
    name: str = Field(min_length=1)
    value: FiniteFloat
    threshold: FiniteFloat
    margin: FiniteFloat
    unit: str = Field(min_length=1)
    status: ReentryMarginStatus


class ReentryMarginReport(AstroModel):
    margins: tuple[ReentryMargin, ...]
    limiting_margin: ReentryMargin


class ReentryResult(AstroModel):
    scenario_id: str = Field(min_length=1)
    workflow: str = "reentry_3dof_v1"
    backend: str = Field(min_length=1)
    samples: tuple[ReentrySample, ...] = Field(min_length=1)
    events: tuple[ReentryEvent, ...]
    peaks: ReentryPeakSummary
    target_miss: ReentryTargetMiss | None = None
    margin_report: ReentryMarginReport
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def samples_and_events_must_be_monotonic(self) -> ReentryResult:
        times = [sample.time_s for sample in self.samples]
        if not all(
            previous < next_time
            for previous, next_time in zip(times, times[1:], strict=False)
        ):
            raise ValueError("Reentry sample times must be strictly increasing")
        event_times = [event.time_s for event in self.events]
        if not all(
            previous <= next_time
            for previous, next_time in zip(event_times, event_times[1:], strict=False)
        ):
            raise ValueError("Reentry event times must be monotonic")
        return self


class ReentryOptimizationConfig(AstroModel):
    maximum_iterations: int = Field(ge=1, le=500, default=80)
    bank_angle_lower_deg: FiniteFloat = Field(ge=0.0, le=90.0, default=0.0)
    bank_angle_upper_deg: FiniteFloat = Field(ge=0.0, le=90.0, default=80.0)
    load_penalty_scale: FiniteFloat = Field(gt=0.0, default=1000.0)

    @field_validator("maximum_iterations", mode="before")
    @classmethod
    def maximum_iterations_must_be_int(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "Reentry optimization maximum_iterations")

    @field_validator(
        "bank_angle_lower_deg", "bank_angle_upper_deg", "load_penalty_scale", mode="before"
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Reentry optimization scalar")

    @model_validator(mode="after")
    def bank_bounds_must_be_ordered(self) -> ReentryOptimizationConfig:
        if self.bank_angle_lower_deg >= self.bank_angle_upper_deg:
            raise ValueError("bank_angle_lower_deg must be below bank_angle_upper_deg")
        return self


class ReentryOptimizationResult(AstroModel):
    scenario_id: str = Field(min_length=1)
    success: bool
    message: str
    iterations: int = Field(ge=0)
    initial_objective: FiniteFloat = Field(ge=0.0)
    final_objective: FiniteFloat = Field(ge=0.0)
    tuned_scenario: ReentryScenario
    reentry_result: ReentryResult
    metadata: dict[str, Any] = Field(default_factory=dict)
