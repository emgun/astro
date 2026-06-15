from __future__ import annotations

from datetime import datetime
from math import cos, isfinite, radians, sin
from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from astro_core.constants import R_EARTH_KM
from astro_core.models import (
    AstroModel,
    Body,
    CartesianState,
    Frame,
    OrbitRepresentation,
    OrbitState,
    Scenario,
    TimeScale,
    Trajectory,
    Vector3,
    _datetime_input_must_be_datetime_or_string,
    _datetime_must_be_aware,
    _numeric_scalar_input_must_be_number,
)

STANDARD_GRAVITY_M_S2 = 9.80665
LaunchEventType = Literal["stage_ignition", "stage_burnout", "stage_separation", "insertion"]


class LaunchSite(AstroModel):
    name: str = Field(min_length=1)
    latitude_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat = Field(ge=-180.0, le=180.0)
    altitude_m: FiniteFloat
    body: Body = Body.EARTH

    @field_validator("latitude_deg", "longitude_deg", "altitude_m", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch site scalar")


class LaunchEngine(AstroModel):
    name: str = Field(min_length=1)
    thrust_n: FiniteFloat = Field(gt=0.0)
    specific_impulse_s: FiniteFloat = Field(gt=0.0)
    throttle_min: FiniteFloat = Field(ge=0.0, le=1.0, default=1.0)
    throttle_max: FiniteFloat = Field(ge=0.0, le=1.0, default=1.0)

    @field_validator(
        "thrust_n",
        "specific_impulse_s",
        "throttle_min",
        "throttle_max",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch engine scalar")

    @model_validator(mode="after")
    def throttle_bounds_must_be_ordered(self) -> LaunchEngine:
        if self.throttle_min > self.throttle_max:
            raise ValueError("Launch engine throttle_min must be <= throttle_max")
        return self

    @property
    def mass_flow_rate_kg_s(self) -> float:
        return self.thrust_n / (self.specific_impulse_s * STANDARD_GRAVITY_M_S2)


class LaunchStage(AstroModel):
    name: str = Field(min_length=1)
    dry_mass_kg: FiniteFloat = Field(ge=0.0)
    propellant_mass_kg: FiniteFloat = Field(ge=0.0)
    engine: LaunchEngine
    burn_duration_s: FiniteFloat = Field(gt=0.0)
    reference_area_m2: FiniteFloat = Field(gt=0.0)
    drag_coefficient: FiniteFloat = Field(ge=0.0, le=10.0)

    @field_validator(
        "dry_mass_kg",
        "propellant_mass_kg",
        "burn_duration_s",
        "reference_area_m2",
        "drag_coefficient",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch stage scalar")

    @model_validator(mode="after")
    def propellant_must_cover_configured_burn(self) -> LaunchStage:
        required_propellant_kg = self.engine.mass_flow_rate_kg_s * self.burn_duration_s
        if self.propellant_mass_kg + 1.0e-9 < required_propellant_kg:
            raise ValueError(
                "LaunchStage propellant_mass_kg must cover configured burn_duration_s"
            )
        return self


class LaunchVehicle(AstroModel):
    name: str = Field(min_length=1)
    payload_mass_kg: FiniteFloat = Field(ge=0.0)
    stages: list[LaunchStage] = Field(min_length=1)

    @field_validator("payload_mass_kg", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch vehicle scalar")

    @property
    def initial_mass_kg(self) -> float:
        return self.payload_mass_kg + sum(
            stage.dry_mass_kg + stage.propellant_mass_kg for stage in self.stages
        )


class AtmosphereConfig(AstroModel):
    model: Literal["none", "exponential"] = "exponential"
    sea_level_density_kg_m3: FiniteFloat = Field(gt=0.0, default=1.225)
    scale_height_m: FiniteFloat = Field(gt=0.0, default=8500.0)

    @field_validator("sea_level_density_kg_m3", "scale_height_m", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Atmosphere scalar")


class PitchProgramPoint(AstroModel):
    time_s: FiniteFloat = Field(ge=0.0)
    pitch_deg: FiniteFloat = Field(ge=0.0, le=90.0)

    @field_validator("time_s", "pitch_deg", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Pitch program scalar")


class GuidanceConfig(AstroModel):
    mode: Literal["vertical", "pitch_program"] = "vertical"
    pitch_program: list[PitchProgramPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pitch_program(self) -> GuidanceConfig:
        if self.mode != "pitch_program":
            return self
        if len(self.pitch_program) < 2:
            raise ValueError("pitch_program guidance requires at least two pitch_program points")
        if self.pitch_program[0].time_s != 0.0:
            raise ValueError("first pitch_program point must start at t=0")
        times_s = [point.time_s for point in self.pitch_program]
        if not all(
            previous_time < next_time
            for previous_time, next_time in zip(times_s, times_s[1:], strict=False)
        ):
            raise ValueError("pitch_program time_s values must be strictly increasing")
        return self


class TargetOrbit(AstroModel):
    altitude_km: FiniteFloat = Field(gt=0.0)
    inclination_deg: FiniteFloat = Field(ge=0.0, le=180.0)
    altitude_tolerance_km: FiniteFloat = Field(gt=0.0, default=10.0)
    velocity_tolerance_km_s: FiniteFloat = Field(gt=0.0, default=0.1)

    @field_validator(
        "altitude_km",
        "inclination_deg",
        "altitude_tolerance_km",
        "velocity_tolerance_km_s",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Target orbit scalar")


class LaunchPropagationConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)

    @field_validator("duration_s", "step_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch propagation scalar")

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s / self.step_s)) + 1

    @model_validator(mode="after")
    def validate_steps(self) -> LaunchPropagationConfig:
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1.0e-9:
            raise ValueError("Launch propagation duration_s must be an integer multiple of step_s")
        return self


class LaunchScenario(AstroModel):
    scenario_id: str = Field(min_length=1)
    description: str = ""
    epoch: datetime
    time_scale: TimeScale = TimeScale.UTC
    frame: Frame = Frame.EME2000
    launch_site: LaunchSite
    vehicle: LaunchVehicle
    atmosphere: AtmosphereConfig = Field(default_factory=AtmosphereConfig)
    guidance: GuidanceConfig = Field(default_factory=GuidanceConfig)
    target_orbit: TargetOrbit
    propagation: LaunchPropagationConfig

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "LaunchScenario epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "LaunchScenario epoch")

    def insertion_state_from_vertical_state(
        self,
        *,
        epoch: datetime,
        altitude_km: float,
        velocity_km_s: float,
    ) -> OrbitState:
        return self.insertion_state_from_local_state(
            epoch=epoch,
            altitude_km=altitude_km,
            radial_velocity_km_s=velocity_km_s,
            horizontal_velocity_km_s=0.0,
        )

    def insertion_state_from_local_state(
        self,
        *,
        epoch: datetime,
        altitude_km: float,
        radial_velocity_km_s: float,
        horizontal_velocity_km_s: float,
    ) -> OrbitState:
        if (
            not isfinite(altitude_km)
            or not isfinite(radial_velocity_km_s)
            or not isfinite(horizontal_velocity_km_s)
        ):
            raise ValueError("Launch insertion altitude and velocity must be finite")

        lat_rad = radians(self.launch_site.latitude_deg)
        lon_rad = radians(self.launch_site.longitude_deg)
        radial_unit = (
            cos(lat_rad) * cos(lon_rad),
            cos(lat_rad) * sin(lon_rad),
            sin(lat_rad),
        )
        east_unit = (-sin(lon_rad), cos(lon_rad), 0.0)
        radius_km = R_EARTH_KM + altitude_km
        position_km: Vector3 = (
            radius_km * radial_unit[0],
            radius_km * radial_unit[1],
            radius_km * radial_unit[2],
        )
        velocity_vector_km_s: Vector3 = (
            radial_velocity_km_s * radial_unit[0] + horizontal_velocity_km_s * east_unit[0],
            radial_velocity_km_s * radial_unit[1] + horizontal_velocity_km_s * east_unit[1],
            radial_velocity_km_s * radial_unit[2] + horizontal_velocity_km_s * east_unit[2],
        )
        return OrbitState(
            epoch=epoch,
            time_scale=self.time_scale,
            frame=self.frame,
            central_body=self.launch_site.body,
            representation=OrbitRepresentation.CARTESIAN,
            cartesian=CartesianState(
                position_km=position_km,
                velocity_km_s=velocity_vector_km_s,
            ),
        )


class LaunchEvent(AstroModel):
    event_type: LaunchEventType
    epoch: datetime
    time_s: FiniteFloat = Field(ge=0.0)
    stage_name: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "LaunchEvent epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "LaunchEvent epoch")

    @field_validator("time_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "LaunchEvent time")


class LaunchTrajectorySample(AstroModel):
    epoch: datetime
    time_s: FiniteFloat = Field(ge=0.0)
    altitude_km: FiniteFloat
    downrange_km: FiniteFloat = 0.0
    velocity_km_s: FiniteFloat
    radial_velocity_km_s: FiniteFloat = 0.0
    horizontal_velocity_km_s: FiniteFloat = 0.0
    mass_kg: FiniteFloat = Field(gt=0.0)
    stage_name: str = Field(min_length=1)
    dynamic_pressure_pa: FiniteFloat = Field(ge=0.0)
    acceleration_m_s2: FiniteFloat
    flight_path_angle_deg: FiniteFloat = 90.0
    state: CartesianState | None = None

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "LaunchTrajectorySample epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "LaunchTrajectorySample epoch")

    @field_validator(
        "time_s",
        "altitude_km",
        "downrange_km",
        "velocity_km_s",
        "radial_velocity_km_s",
        "horizontal_velocity_km_s",
        "mass_kg",
        "dynamic_pressure_pa",
        "acceleration_m_s2",
        "flight_path_angle_deg",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch trajectory scalar")


class LaunchTrajectory(AstroModel):
    scenario_id: str = Field(min_length=1)
    samples: list[LaunchTrajectorySample] = Field(min_length=1)
    events: list[LaunchEvent] = Field(default_factory=list)
    insertion_state: OrbitState
    target_miss: dict[str, FiniteFloat]
    backend: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def samples_and_events_must_be_monotonic(self) -> LaunchTrajectory:
        sample_epochs = [sample.epoch for sample in self.samples]
        sample_times = [sample.time_s for sample in self.samples]
        if not all(
            previous_epoch < next_epoch
            for previous_epoch, next_epoch in zip(sample_epochs, sample_epochs[1:], strict=False)
        ):
            raise ValueError("Launch trajectory sample epochs must be strictly increasing")
        if not all(
            previous_time < next_time
            for previous_time, next_time in zip(sample_times, sample_times[1:], strict=False)
        ):
            raise ValueError("Launch trajectory sample times must be strictly increasing")

        event_epochs = [event.epoch for event in self.events]
        event_times = [event.time_s for event in self.events]
        if not all(
            previous_epoch <= next_epoch
            for previous_epoch, next_epoch in zip(event_epochs, event_epochs[1:], strict=False)
        ):
            raise ValueError("Launch event epochs must be monotonic")
        if not all(
            previous_time <= next_time
            for previous_time, next_time in zip(event_times, event_times[1:], strict=False)
        ):
            raise ValueError("Launch event times must be monotonic")

        return self


class LaunchPitchSweepCase(AstroModel):
    pitch_deg: FiniteFloat = Field(ge=0.0, le=90.0)
    score: FiniteFloat = Field(ge=0.0)
    altitude_miss_km: FiniteFloat
    velocity_miss_km_s: FiniteFloat
    final_altitude_km: FiniteFloat
    final_velocity_km_s: FiniteFloat
    final_radial_velocity_km_s: FiniteFloat
    final_horizontal_velocity_km_s: FiniteFloat
    final_downrange_km: FiniteFloat
    target_miss: dict[str, FiniteFloat]

    @field_validator(
        "pitch_deg",
        "score",
        "altitude_miss_km",
        "velocity_miss_km_s",
        "final_altitude_km",
        "final_velocity_km_s",
        "final_radial_velocity_km_s",
        "final_horizontal_velocity_km_s",
        "final_downrange_km",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch pitch sweep scalar")


class LaunchPitchSweepResult(AstroModel):
    scenario_id: str = Field(min_length=1)
    point_index: int = Field(ge=0)
    point_time_s: FiniteFloat = Field(ge=0.0)
    baseline_pitch_deg: FiniteFloat = Field(ge=0.0, le=90.0)
    altitude_weight: FiniteFloat = Field(ge=0.0)
    velocity_weight: FiniteFloat = Field(ge=0.0)
    cases: list[LaunchPitchSweepCase] = Field(min_length=1)
    best_case: LaunchPitchSweepCase
    backend: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "point_time_s",
        "baseline_pitch_deg",
        "altitude_weight",
        "velocity_weight",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch pitch sweep scalar")


class LaunchPitchTuningPoint(AstroModel):
    point_index: int = Field(ge=0)
    time_s: FiniteFloat = Field(ge=0.0)
    baseline_pitch_deg: FiniteFloat = Field(ge=0.0, le=90.0)
    tuned_pitch_deg: FiniteFloat = Field(ge=0.0, le=90.0)

    @field_validator(
        "time_s",
        "baseline_pitch_deg",
        "tuned_pitch_deg",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch pitch tuning scalar")


class LaunchPitchTuningCase(AstroModel):
    iteration: int = Field(ge=1)
    pitch_deg_by_point_index: dict[str, FiniteFloat]
    score: FiniteFloat = Field(ge=0.0)
    altitude_miss_km: FiniteFloat
    velocity_miss_km_s: FiniteFloat
    final_altitude_km: FiniteFloat
    final_velocity_km_s: FiniteFloat
    final_radial_velocity_km_s: FiniteFloat
    final_horizontal_velocity_km_s: FiniteFloat
    final_downrange_km: FiniteFloat
    target_miss: dict[str, FiniteFloat]

    @field_validator(
        "score",
        "altitude_miss_km",
        "velocity_miss_km_s",
        "final_altitude_km",
        "final_velocity_km_s",
        "final_radial_velocity_km_s",
        "final_horizontal_velocity_km_s",
        "final_downrange_km",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch pitch tuning scalar")


class LaunchPitchTuningIteration(AstroModel):
    iteration: int = Field(ge=1)
    span_deg: FiniteFloat = Field(gt=0.0)
    cases: list[LaunchPitchTuningCase] = Field(min_length=1)
    best_case: LaunchPitchTuningCase

    @field_validator("span_deg", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch pitch tuning scalar")


class LaunchPitchTuningResult(AstroModel):
    scenario_id: str = Field(min_length=1)
    point_indices: list[int] = Field(min_length=2, max_length=2)
    tuned_points: list[LaunchPitchTuningPoint] = Field(min_length=2, max_length=2)
    initial_span_deg: FiniteFloat = Field(gt=0.0)
    refinement_factor: FiniteFloat = Field(gt=0.0, lt=1.0)
    altitude_weight: FiniteFloat = Field(ge=0.0)
    velocity_weight: FiniteFloat = Field(ge=0.0)
    iterations: list[LaunchPitchTuningIteration] = Field(min_length=1)
    best_case: LaunchPitchTuningCase
    tuned_scenario: LaunchScenario
    backend: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "initial_span_deg",
        "refinement_factor",
        "altitude_weight",
        "velocity_weight",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch pitch tuning scalar")


class LaunchReportInsertionMetrics(AstroModel):
    target_altitude_km: FiniteFloat = Field(gt=0.0)
    target_circular_velocity_km_s: FiniteFloat = Field(gt=0.0)
    altitude_km: FiniteFloat
    velocity_km_s: FiniteFloat
    radial_velocity_km_s: FiniteFloat
    horizontal_velocity_km_s: FiniteFloat
    altitude_miss_km: FiniteFloat
    velocity_miss_km_s: FiniteFloat

    @field_validator(
        "target_altitude_km",
        "target_circular_velocity_km_s",
        "altitude_km",
        "velocity_km_s",
        "radial_velocity_km_s",
        "horizontal_velocity_km_s",
        "altitude_miss_km",
        "velocity_miss_km_s",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch report scalar")


class LaunchReportShortArcMetrics(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)
    sample_count: int = Field(ge=1)
    target_altitude_km: FiniteFloat = Field(gt=0.0)
    target_circular_velocity_km_s: FiniteFloat = Field(gt=0.0)
    initial_altitude_km: FiniteFloat
    final_altitude_km: FiniteFloat
    min_altitude_km: FiniteFloat
    max_altitude_km: FiniteFloat
    final_velocity_km_s: FiniteFloat
    final_altitude_miss_km: FiniteFloat
    final_velocity_miss_km_s: FiniteFloat
    altitudes_km: list[FiniteFloat] = Field(min_length=1)

    @field_validator(
        "duration_s",
        "step_s",
        "target_altitude_km",
        "target_circular_velocity_km_s",
        "initial_altitude_km",
        "final_altitude_km",
        "min_altitude_km",
        "max_altitude_km",
        "final_velocity_km_s",
        "final_altitude_miss_km",
        "final_velocity_miss_km_s",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Launch report scalar")


class TunedLaunchReport(AstroModel):
    scenario_id: str = Field(min_length=1)
    tuning_result: LaunchPitchTuningResult
    launch_trajectory: LaunchTrajectory
    orbit_scenario: Scenario
    orbit_trajectory: Trajectory
    insertion_metrics: LaunchReportInsertionMetrics
    short_arc_metrics: LaunchReportShortArcMetrics
    backend: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
