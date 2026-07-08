from __future__ import annotations

from math import sqrt

from astro_core.constants import R_EARTH_KM
from astro_core.models import Trajectory
from astro_twin.models import TimelineGeometrySample


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
