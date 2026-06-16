from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
from math import cos, isfinite, radians, sin, sqrt
from numbers import Number
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

Vector3 = tuple[float, float, float]
WGS84_EQUATORIAL_RADIUS_KM = 6378.137
WGS84_FLATTENING = 1.0 / 298.257223563
SECONDS_PER_DAY = 86400.0
UNIX_EPOCH_JULIAN_DATE = 2440587.5
J2000_JULIAN_DATE = 2451545.0


class AstroModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _datetime_must_be_aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include timezone information")
    return value


def _datetime_input_must_be_datetime_or_string(value: Any, label: str) -> Any:
    if isinstance(value, bool | np.bool_ | Number | np.number | Decimal | Fraction):
        raise ValueError(f"{label} must be a datetime or ISO datetime string")
    if isinstance(value, str):
        try:
            float(value.strip())
        except ValueError:
            return value
        else:
            raise ValueError(f"{label} must be a datetime or ISO datetime string")
    return value


def _numeric_scalar_input_must_be_number(value: Any, label: str) -> Any:
    if isinstance(value, bool | np.bool_ | str):
        raise ValueError(f"{label} must be a numeric scalar")
    return value


def _integer_input_must_be_int(value: Any, label: str) -> Any:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _numeric_sequence_input_must_be_numbers(value: Any, label: str) -> Any:
    if isinstance(value, np.ndarray):
        raise ValueError(f"{label} does not accept NumPy arrays")
    if isinstance(value, str | bytes):
        raise ValueError(f"{label} must contain numeric scalar values")
    if not isinstance(value, list | tuple):
        return value

    for component in value:
        _numeric_scalar_input_must_be_number(component, label)
    return value


def _numeric_matrix_input_must_be_numbers(value: Any, label: str) -> Any:
    if isinstance(value, np.ndarray):
        raise ValueError(f"{label} does not accept NumPy arrays")
    if isinstance(value, str | bytes):
        raise ValueError(f"{label} must contain numeric scalar values")
    if not isinstance(value, list | tuple):
        return value

    for row in value:
        _numeric_sequence_input_must_be_numbers(row, label)
    return value


class Body(StrEnum):
    EARTH = "earth"


class Frame(StrEnum):
    EME2000 = "EME2000"


class TimeScale(StrEnum):
    UTC = "UTC"


class OrbitRepresentation(StrEnum):
    CARTESIAN = "cartesian"


class ForceModelName(StrEnum):
    TWO_BODY = "two_body"
    J2 = "j2"
    OREKIT_HIGH_FIDELITY = "orekit_high_fidelity"


class MeasurementType(StrEnum):
    RANGE = "range"
    RANGE_RATE = "range_rate"
    DOPPLER = "doppler"
    RIGHT_ASCENSION = "right_ascension"
    DECLINATION = "declination"
    AZIMUTH = "azimuth"
    ELEVATION = "elevation"


class CartesianState(AstroModel):
    position_km: Vector3
    velocity_km_s: Vector3

    @field_validator("position_km", "velocity_km_s", mode="before")
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Cartesian state vector")

    @field_validator("position_km", "velocity_km_s")
    @classmethod
    def values_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Cartesian state values must be finite")
        return value

    def position_array(self) -> NDArray[np.float64]:
        return np.array(self.position_km, dtype=np.float64)

    def velocity_array(self) -> NDArray[np.float64]:
        return np.array(self.velocity_km_s, dtype=np.float64)


class OrbitState(AstroModel):
    epoch: datetime
    time_scale: TimeScale
    frame: Frame
    central_body: Body
    representation: OrbitRepresentation
    cartesian: CartesianState

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "OrbitState epoch")

    @model_validator(mode="after")
    def validate_epoch(self) -> OrbitState:
        _datetime_must_be_aware(self.epoch, "OrbitState epoch")
        return self


class Spacecraft(AstroModel):
    name: str = Field(min_length=1)
    mass_kg: FiniteFloat = Field(gt=0.0)
    area_m2: FiniteFloat = Field(gt=0.0)
    drag_coefficient: FiniteFloat = Field(ge=0.0, le=10.0)
    reflectivity_coefficient: FiniteFloat = Field(ge=0.0, le=5.0)

    @field_validator(
        "mass_kg",
        "area_m2",
        "drag_coefficient",
        "reflectivity_coefficient",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Spacecraft scalar")


class ForceModelConfig(AstroModel):
    gravity: ForceModelName
    atmospheric_drag: bool = False
    solar_radiation_pressure: bool = False
    third_body_gravity: bool = False

    def enabled_high_fidelity_flags(self) -> tuple[str, ...]:
        flags: list[str] = []
        if self.atmospheric_drag:
            flags.append("atmospheric_drag")
        if self.solar_radiation_pressure:
            flags.append("solar_radiation_pressure")
        if self.third_body_gravity:
            flags.append("third_body_gravity")
        return tuple(flags)


class PropagationConfig(AstroModel):
    duration_s: FiniteFloat = Field(gt=0.0)
    step_s: FiniteFloat = Field(gt=0.0)

    @field_validator("duration_s", "step_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Propagation scalar")

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s / self.step_s)) + 1

    @model_validator(mode="after")
    def validate_steps(self) -> PropagationConfig:
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("Propagation duration_s must be an integer multiple of step_s")
        return self


class EarthOrientationConfig(AstroModel):
    """Small Earth-orientation correction set for suite geodetic station transforms."""

    ut1_minus_utc_s: FiniteFloat = 0.0
    polar_motion_x_arcsec: FiniteFloat = 0.0
    polar_motion_y_arcsec: FiniteFloat = 0.0
    source: str = "zero"

    @field_validator(
        "ut1_minus_utc_s",
        "polar_motion_x_arcsec",
        "polar_motion_y_arcsec",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Earth orientation scalar")

    @field_validator("ut1_minus_utc_s", "polar_motion_x_arcsec", "polar_motion_y_arcsec")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Earth orientation values must be finite")
        return value


class GroundStation(AstroModel):
    name: str = Field(min_length=1)
    position_eci_km: Vector3 | None = None
    latitude_deg: FiniteFloat | None = Field(default=None, ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat | None = Field(default=None, ge=-360.0, le=360.0)
    altitude_km: FiniteFloat | None = None
    frame: Frame
    elevation_mask_deg: FiniteFloat = Field(ge=-90.0, le=90.0)

    @field_validator("position_eci_km", mode="before")
    @classmethod
    def position_input_must_be_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        return _numeric_sequence_input_must_be_numbers(value, "Ground station position")

    @field_validator("latitude_deg", "longitude_deg", "altitude_km", mode="before")
    @classmethod
    def geodetic_input_must_be_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        return _numeric_scalar_input_must_be_number(value, "Ground station geodetic coordinate")

    @field_validator("elevation_mask_deg", mode="before")
    @classmethod
    def elevation_mask_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Ground station elevation mask")

    @field_validator("position_eci_km")
    @classmethod
    def position_must_be_finite(cls, value: Vector3 | None) -> Vector3 | None:
        if value is None:
            return None
        if not all(isfinite(component) for component in value):
            raise ValueError("Ground station position values must be finite")
        return value

    @field_validator("latitude_deg", "longitude_deg", "altitude_km")
    @classmethod
    def geodetic_values_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("Ground station geodetic coordinate values must be finite")
        return value

    @model_validator(mode="after")
    def validate_position_definition(self) -> GroundStation:
        has_eci_position = self.position_eci_km is not None
        geodetic_values = (self.latitude_deg, self.longitude_deg, self.altitude_km)
        has_any_geodetic = any(value is not None for value in geodetic_values)
        has_all_geodetic = all(value is not None for value in geodetic_values)

        if has_eci_position and has_any_geodetic:
            raise ValueError(
                "Ground station must define position_eci_km or geodetic fields, not both"
            )
        if not has_eci_position and not has_any_geodetic:
            raise ValueError("Ground station must define position_eci_km or geodetic fields")
        if has_any_geodetic and not has_all_geodetic:
            raise ValueError(
                "Ground station geodetic definition requires latitude_deg, longitude_deg, "
                "and altitude_km"
            )
        return self

    def position_array(
        self,
        epoch: datetime | None = None,
        earth_orientation: EarthOrientationConfig | None = None,
    ) -> NDArray[np.float64]:
        if self.position_eci_km is not None:
            return np.array(self.position_eci_km, dtype=np.float64)
        if epoch is None:
            raise ValueError("Ground station geodetic position requires an epoch")
        return _ecef_to_eci(_geodetic_to_ecef(self), epoch, earth_orientation)


def _geodetic_to_ecef(station: GroundStation) -> NDArray[np.float64]:
    if (
        station.latitude_deg is None
        or station.longitude_deg is None
        or station.altitude_km is None
    ):
        raise ValueError("Ground station geodetic definition is incomplete")

    latitude_rad = radians(station.latitude_deg)
    longitude_rad = radians(station.longitude_deg)
    eccentricity_squared = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
    sin_latitude = sin(latitude_rad)
    prime_vertical_radius = WGS84_EQUATORIAL_RADIUS_KM / sqrt(
        1.0 - eccentricity_squared * sin_latitude**2
    )

    radius_at_altitude = prime_vertical_radius + station.altitude_km
    x_km = radius_at_altitude * cos(latitude_rad) * cos(longitude_rad)
    y_km = radius_at_altitude * cos(latitude_rad) * sin(longitude_rad)
    z_km = (
        prime_vertical_radius * (1.0 - eccentricity_squared) + station.altitude_km
    ) * sin_latitude
    return np.array((x_km, y_km, z_km), dtype=np.float64)


def _ecef_to_eci(
    position_ecef_km: NDArray[np.float64],
    epoch: datetime,
    earth_orientation: EarthOrientationConfig | None = None,
) -> NDArray[np.float64]:
    earth_orientation = earth_orientation or EarthOrientationConfig()
    polar_corrected_km = _apply_polar_motion(position_ecef_km, earth_orientation)
    earth_rotation_angle_rad = _greenwich_sidereal_angle_rad(
        epoch,
        ut1_minus_utc_s=earth_orientation.ut1_minus_utc_s,
    )
    cos_angle = cos(earth_rotation_angle_rad)
    sin_angle = sin(earth_rotation_angle_rad)
    x_km, y_km, z_km = polar_corrected_km
    return np.array(
        (
            cos_angle * x_km - sin_angle * y_km,
            sin_angle * x_km + cos_angle * y_km,
            z_km,
        ),
        dtype=np.float64,
    )


def _apply_polar_motion(
    position_ecef_km: NDArray[np.float64],
    earth_orientation: EarthOrientationConfig,
) -> NDArray[np.float64]:
    arcsec_to_rad = radians(1.0 / 3600.0)
    xp_rad = earth_orientation.polar_motion_x_arcsec * arcsec_to_rad
    yp_rad = earth_orientation.polar_motion_y_arcsec * arcsec_to_rad
    if xp_rad == 0.0 and yp_rad == 0.0:
        return position_ecef_km

    x_km, y_km, z_km = position_ecef_km
    cos_xp = cos(-xp_rad)
    sin_xp = sin(-xp_rad)
    cos_yp = cos(-yp_rad)
    sin_yp = sin(-yp_rad)

    # Apply a compact polar-motion correction before the Earth rotation. This is an
    # input-driven engineering approximation, not a replacement for full IERS reductions.
    x_after_y = cos_xp * x_km + sin_xp * z_km
    y_after_y = y_km
    z_after_y = -sin_xp * x_km + cos_xp * z_km

    return np.array(
        (
            x_after_y,
            cos_yp * y_after_y - sin_yp * z_after_y,
            sin_yp * y_after_y + cos_yp * z_after_y,
        ),
        dtype=np.float64,
    )


def _greenwich_sidereal_angle_rad(epoch: datetime, *, ut1_minus_utc_s: float = 0.0) -> float:
    epoch_utc = epoch.astimezone(UTC)
    julian_date = (
        epoch_utc.timestamp() + ut1_minus_utc_s
    ) / SECONDS_PER_DAY + UNIX_EPOCH_JULIAN_DATE
    centuries_since_j2000 = (julian_date - J2000_JULIAN_DATE) / 36525.0
    sidereal_degrees = (
        280.46061837
        + 360.98564736629 * (julian_date - J2000_JULIAN_DATE)
        + 0.000387933 * centuries_since_j2000**2
        - centuries_since_j2000**3 / 38710000.0
    )
    return radians(sidereal_degrees % 360.0)


class MeasurementNoise(AstroModel):
    range_sigma_km: FiniteFloat = Field(gt=0.0, default=0.01)
    range_rate_sigma_km_s: FiniteFloat = Field(gt=0.0, default=1.0e-5)
    doppler_sigma_hz: FiniteFloat = Field(gt=0.0, default=0.1)
    angle_sigma_deg: FiniteFloat = Field(gt=0.0, default=0.001)
    seed: int = 42

    @field_validator(
        "range_sigma_km",
        "range_rate_sigma_km_s",
        "doppler_sigma_hz",
        "angle_sigma_deg",
        mode="before",
    )
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Measurement noise scalar")

    @field_validator("seed", mode="before")
    @classmethod
    def seed_must_be_integer_input(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "Measurement noise seed")


class MeasurementConfig(AstroModel):
    types: tuple[MeasurementType, ...] = Field(
        default=(MeasurementType.RANGE, MeasurementType.RANGE_RATE),
        min_length=1,
    )
    cadence_s: FiniteFloat = Field(gt=0.0, default=60.0)
    doppler_transmit_frequency_hz: FiniteFloat = Field(gt=0.0, default=8.4e9)
    noise: MeasurementNoise = Field(default_factory=MeasurementNoise)

    @field_validator("cadence_s", "doppler_transmit_frequency_hz", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Measurement config scalar")


class MeasurementRecord(AstroModel):
    measurement_type: MeasurementType
    epoch: datetime
    observer: str
    observed_object: str
    value: FiniteFloat
    sigma: FiniteFloat = Field(gt=0.0)
    units: Literal["km", "km/s", "Hz", "deg"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "MeasurementRecord epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "MeasurementRecord epoch")

    @field_validator("value", "sigma", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Measurement scalar")

    @field_validator("value", "sigma")
    @classmethod
    def numeric_values_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Measurement numeric values must be finite")
        return value

    @model_validator(mode="after")
    def measurement_units_must_match_type(self) -> MeasurementRecord:
        expected_units = {
            MeasurementType.RANGE: "km",
            MeasurementType.RANGE_RATE: "km/s",
            MeasurementType.DOPPLER: "Hz",
            MeasurementType.RIGHT_ASCENSION: "deg",
            MeasurementType.DECLINATION: "deg",
            MeasurementType.AZIMUTH: "deg",
            MeasurementType.ELEVATION: "deg",
        }
        if self.units != expected_units[self.measurement_type]:
            expected_unit = expected_units[self.measurement_type]
            raise ValueError(f"{self.measurement_type} measurements must use units {expected_unit}")
        return self


class TrajectoryEvent(AstroModel):
    event_type: str = Field(min_length=1)
    epoch: datetime
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "TrajectoryEvent epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "TrajectoryEvent epoch")


class Maneuver(AstroModel):
    name: str = Field(min_length=1)
    epoch: datetime
    frame: Frame
    delta_v_km_s: Vector3
    duration_s: FiniteFloat = Field(ge=0.0, default=0.0)
    thrust_vector_n: Vector3 | None = None
    specific_impulse_s: FiniteFloat | None = Field(default=None, gt=0.0)
    thrust_direction_mode: Literal["inertial", "velocity_aligned"] = "inertial"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "Maneuver epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "Maneuver epoch")

    @field_validator("delta_v_km_s", "thrust_vector_n", mode="before")
    @classmethod
    def delta_v_input_must_be_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        return _numeric_sequence_input_must_be_numbers(value, "Maneuver delta-v")

    @field_validator("delta_v_km_s", "thrust_vector_n")
    @classmethod
    def delta_v_must_be_finite(cls, value: Vector3) -> Vector3:
        if value is None:
            return value
        if not all(isfinite(component) for component in value):
            raise ValueError("Maneuver delta-v values must be finite")
        return value

    @field_validator("duration_s", "specific_impulse_s", mode="before")
    @classmethod
    def duration_input_must_be_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        return _numeric_scalar_input_must_be_number(value, "Maneuver duration")

    @model_validator(mode="after")
    def validate_thrust_vector_mass_flow(self) -> Maneuver:
        has_thrust_vector = self.thrust_vector_n is not None
        has_specific_impulse = self.specific_impulse_s is not None
        if has_thrust_vector != has_specific_impulse:
            raise ValueError(
                "thrust-vector maneuvers require both thrust_vector_n and specific_impulse_s"
            )
        if not has_thrust_vector:
            return self
        if self.duration_s <= 0.0:
            raise ValueError("thrust-vector maneuvers require duration_s > 0")
        if self.thrust_vector_n == (0.0, 0.0, 0.0):
            raise ValueError("thrust-vector maneuvers require nonzero thrust_vector_n")
        return self


class CovarianceSample(AstroModel):
    epoch: datetime
    covariance: list[list[FiniteFloat]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "CovarianceSample epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "CovarianceSample epoch")

    @field_validator("covariance", mode="before")
    @classmethod
    def covariance_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_matrix_input_must_be_numbers(value, "CovarianceSample covariance")

    @field_validator("covariance")
    @classmethod
    def covariance_must_be_6x6_and_finite(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 6 or any(len(row) != 6 for row in value):
            raise ValueError("CovarianceSample covariance must be 6x6")
        if any(not isfinite(component) for row in value for component in row):
            raise ValueError("CovarianceSample covariance values must be finite")
        return value


class TrajectorySample(AstroModel):
    epoch: datetime
    state: CartesianState
    mass_kg: FiniteFloat | None = Field(default=None, gt=0.0)

    @field_validator("epoch", mode="before")
    @classmethod
    def epoch_input_must_be_datetime_or_string(cls, value: Any) -> Any:
        return _datetime_input_must_be_datetime_or_string(value, "TrajectorySample epoch")

    @field_validator("epoch")
    @classmethod
    def epoch_must_be_aware(cls, value: datetime) -> datetime:
        return _datetime_must_be_aware(value, "TrajectorySample epoch")

    @field_validator("mass_kg", mode="before")
    @classmethod
    def mass_input_must_be_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        return _numeric_scalar_input_must_be_number(value, "TrajectorySample mass")


class Trajectory(AstroModel):
    scenario_id: str = Field(min_length=1)
    samples: list[TrajectorySample] = Field(min_length=1)
    force_model: ForceModelConfig
    backend: str = Field(min_length=1)
    events: list[TrajectoryEvent] = Field(default_factory=list)
    maneuvers: list[Maneuver] = Field(default_factory=list)
    covariance_history: list[CovarianceSample] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def epochs_must_be_monotonic(self) -> Trajectory:
        epochs = [sample.epoch for sample in self.samples]
        for epoch in epochs:
            _datetime_must_be_aware(epoch, "Trajectory sample epoch")

        try:
            is_strictly_increasing = all(
                previous_epoch < next_epoch
                for previous_epoch, next_epoch in zip(epochs, epochs[1:], strict=False)
            )
        except TypeError as exc:
            raise ValueError("Trajectory sample epochs must be comparable aware datetimes") from exc

        if not is_strictly_increasing:
            raise ValueError("Trajectory sample epochs must be strictly increasing")
        return self


class EstimateResult(AstroModel):
    estimated_state: OrbitState
    residuals: list[FiniteFloat]
    covariance: list[list[FiniteFloat]]
    rms: FiniteFloat
    iterations: int = Field(ge=0)
    converged: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("residuals", mode="before")
    @classmethod
    def residual_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "EstimateResult residuals")

    @field_validator("residuals")
    @classmethod
    def residuals_must_be_finite(cls, value: list[float]) -> list[float]:
        if not all(isfinite(residual) for residual in value):
            raise ValueError("EstimateResult residuals must be finite")
        return value

    @field_validator("rms", mode="before")
    @classmethod
    def rms_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "EstimateResult rms")

    @field_validator("rms")
    @classmethod
    def rms_must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("EstimateResult rms must be finite")
        return value

    @field_validator("covariance", mode="before")
    @classmethod
    def covariance_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_matrix_input_must_be_numbers(value, "EstimateResult covariance")

    @field_validator("covariance")
    @classmethod
    def covariance_must_be_6x6_and_finite(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 6 or any(len(row) != 6 for row in value):
            raise ValueError("EstimateResult covariance must be 6x6")
        if not all(isfinite(component) for row in value for component in row):
            raise ValueError("EstimateResult covariance values must be finite")
        return value

    @field_validator("iterations", mode="before")
    @classmethod
    def iterations_must_be_integer_input(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "EstimateResult iterations")


class Scenario(AstroModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    scenario_id: str = Field(min_length=1)
    description: str = ""
    spacecraft: Spacecraft
    initial_state: OrbitState
    force_model: ForceModelConfig
    propagation: PropagationConfig
    earth_orientation: EarthOrientationConfig = Field(default_factory=EarthOrientationConfig)
    maneuvers: list[Maneuver] = Field(default_factory=list)
    initial_covariance: list[list[FiniteFloat]] | None = None
    covariance_process_noise_acceleration_km_s2: FiniteFloat = Field(ge=0.0, default=0.0)
    ground_stations: list[GroundStation] = Field(default_factory=list)
    measurements: MeasurementConfig = Field(default_factory=MeasurementConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("covariance_process_noise_acceleration_km_s2", mode="before")
    @classmethod
    def covariance_process_noise_input_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(
            value,
            "Scenario covariance process noise acceleration",
        )

    @field_validator("initial_covariance", mode="before")
    @classmethod
    def initial_covariance_inputs_must_be_numeric(cls, value: Any) -> Any:
        if value is None:
            return None
        return _numeric_matrix_input_must_be_numbers(value, "Scenario initial_covariance")

    @field_validator("initial_covariance")
    @classmethod
    def initial_covariance_must_be_6x6_and_finite(
        cls,
        value: list[list[float]] | None,
    ) -> list[list[float]] | None:
        if value is None:
            return None
        if len(value) != 6 or any(len(row) != 6 for row in value):
            raise ValueError("Scenario initial_covariance must be 6x6")
        if any(not isfinite(component) for row in value for component in row):
            raise ValueError("Scenario initial_covariance values must be finite")
        return value
