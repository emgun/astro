from __future__ import annotations

from math import atan2, cos, degrees, sin, sqrt

from astro_core.constants import R_EARTH_KM
from astro_core.models import Trajectory, _greenwich_sidereal_angle_rad
from astro_reentry.models import ReentryInitialState, ReentryScenario, ReentryVehicle

EARTH_ROTATION_RAD_S = 7.2921159e-5


def trajectory_to_reentry_scenario(
    trajectory: Trajectory,
    template: ReentryScenario,
    *,
    sample_index: int = -1,
    scenario_id: str | None = None,
    use_sample_mass: bool = False,
) -> ReentryScenario:
    resolved_sample_index = (
        sample_index if sample_index >= 0 else len(trajectory.samples) + sample_index
    )
    try:
        sample = trajectory.samples[sample_index]
    except IndexError as exc:
        raise ValueError(
            f"reentry handoff sample_index {sample_index} is outside trajectory samples"
        ) from exc

    position_ecef_km, velocity_ecef_km_s = _eci_state_to_ecef(
        sample.state.position_km,
        sample.state.velocity_km_s,
        sample.epoch,
    )
    radius_km = _norm(position_ecef_km)
    latitude_rad = atan2(
        position_ecef_km[2],
        sqrt(position_ecef_km[0] ** 2 + position_ecef_km[1] ** 2),
    )
    longitude_rad = atan2(position_ecef_km[1], position_ecef_km[0])
    up = (
        cos(latitude_rad) * cos(longitude_rad),
        cos(latitude_rad) * sin(longitude_rad),
        sin(latitude_rad),
    )
    east = (-sin(longitude_rad), cos(longitude_rad), 0.0)
    north = (
        -sin(latitude_rad) * cos(longitude_rad),
        -sin(latitude_rad) * sin(longitude_rad),
        cos(latitude_rad),
    )
    radial_velocity = _dot(velocity_ecef_km_s, up)
    east_velocity = _dot(velocity_ecef_km_s, east)
    north_velocity = _dot(velocity_ecef_km_s, north)
    horizontal_velocity = sqrt(east_velocity**2 + north_velocity**2)
    speed_km_s = sqrt(radial_velocity**2 + horizontal_velocity**2)
    if speed_km_s <= 0.0:
        raise ValueError("reentry handoff trajectory sample has zero atmosphere-relative speed")

    initial_state = ReentryInitialState(
        epoch=sample.epoch,
        altitude_km=radius_km - R_EARTH_KM,
        velocity_km_s=speed_km_s,
        flight_path_angle_deg=degrees(atan2(radial_velocity, horizontal_velocity)),
        heading_deg=degrees(atan2(east_velocity, north_velocity)) % 360.0,
        latitude_deg=degrees(latitude_rad),
        longitude_deg=degrees(longitude_rad),
    )
    vehicle = template.vehicle
    if use_sample_mass:
        if sample.mass_kg is None:
            raise ValueError(
                "reentry handoff requested sample mass but trajectory has no mass sample"
            )
        vehicle = ReentryVehicle(
            **{
                **template.vehicle.model_dump(),
                "mass_kg": sample.mass_kg,
            }
        )
    payload = template.model_dump(mode="python")
    payload.update(
        {
            "scenario_id": scenario_id or f"{trajectory.scenario_id}-reentry",
            "initial_state": initial_state,
            "vehicle": vehicle,
            "metadata": {
                **template.metadata,
                "workflow": "trajectory_reentry_handoff",
                "source_trajectory_scenario_id": trajectory.scenario_id,
                "source_trajectory_backend": trajectory.backend,
                "source_sample_index": resolved_sample_index,
                "requested_sample_index": sample_index,
                "source_frame": "EME2000",
                "handoff_frame": "Earth-fixed spherical",
                "earth_rotation_model": "GMST plus constant angular velocity",
                "sample_mass_applied": use_sample_mass,
            },
        }
    )
    return ReentryScenario.model_validate(payload)


def _eci_state_to_ecef(
    position_eci_km: tuple[float, float, float],
    velocity_eci_km_s: tuple[float, float, float],
    epoch: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from datetime import datetime

    if not isinstance(epoch, datetime):
        raise TypeError("reentry handoff epoch must be a datetime")
    angle = _greenwich_sidereal_angle_rad(epoch)
    cosine = cos(angle)
    sine = sin(angle)
    position_ecef = (
        cosine * position_eci_km[0] + sine * position_eci_km[1],
        -sine * position_eci_km[0] + cosine * position_eci_km[1],
        position_eci_km[2],
    )
    velocity_rotated = (
        cosine * velocity_eci_km_s[0] + sine * velocity_eci_km_s[1],
        -sine * velocity_eci_km_s[0] + cosine * velocity_eci_km_s[1],
        velocity_eci_km_s[2],
    )
    rotation_velocity = (
        -EARTH_ROTATION_RAD_S * position_ecef[1],
        EARTH_ROTATION_RAD_S * position_ecef[0],
        0.0,
    )
    velocity_ecef = tuple(
        velocity_rotated[index] - rotation_velocity[index] for index in range(3)
    )
    return position_ecef, velocity_ecef  # type: ignore[return-value]


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _norm(vector: tuple[float, float, float]) -> float:
    return sqrt(sum(component**2 for component in vector))
