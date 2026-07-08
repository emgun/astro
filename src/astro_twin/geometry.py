from __future__ import annotations

from math import asin, cos, degrees, radians, sin, sqrt

from astro_core.constants import R_EARTH_KM
from astro_core.models import Trajectory
from astro_twin.models import GroundSiteConfig, TimelineGeometrySample

_EARTH_ROTATION_RAD_S = 7.2921159e-5


def build_geometry_timeline(trajectory: Trajectory) -> tuple[TimelineGeometrySample, ...]:
    samples: list[TimelineGeometrySample] = []
    start_epoch = trajectory.samples[0].epoch
    for sample in trajectory.samples:
        position = sample.state.position_km
        radius_km = sqrt(sum(component * component for component in position))
        samples.append(
            TimelineGeometrySample(
                epoch=sample.epoch,
                elapsed_s=(sample.epoch - start_epoch).total_seconds(),
                position_km=position,
                altitude_km=radius_km - R_EARTH_KM,
                sunlit=_is_sunlit(position),
            )
        )
    return tuple(samples)


def _is_sunlit(position_km: tuple[float, float, float]) -> bool:
    x, y, z = position_km
    if x >= 0.0:
        return True
    perpendicular_distance_km = sqrt(y * y + z * z)
    return perpendicular_distance_km > R_EARTH_KM


def elevation_and_range_km(
    position_km: tuple[float, float, float],
    site: GroundSiteConfig,
    elapsed_s: float,
) -> tuple[float, float]:
    site_position = _site_position_eci_km(site, elapsed_s)
    relative = tuple(position_km[index] - site_position[index] for index in range(3))
    range_km = sqrt(sum(component * component for component in relative))
    zenith = _unit(site_position)
    elevation_rad = asin(sum(relative[index] * zenith[index] for index in range(3)) / range_km)
    return degrees(elevation_rad), range_km


def _site_position_eci_km(site: GroundSiteConfig, elapsed_s: float) -> tuple[float, float, float]:
    latitude = radians(site.latitude_deg)
    longitude = radians(site.longitude_deg) + _EARTH_ROTATION_RAD_S * elapsed_s
    radius_km = R_EARTH_KM + site.altitude_m / 1000.0
    return (
        radius_km * cos(latitude) * cos(longitude),
        radius_km * cos(latitude) * sin(longitude),
        radius_km * sin(latitude),
    )


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(component * component for component in vector))
    return tuple(component / norm for component in vector)
