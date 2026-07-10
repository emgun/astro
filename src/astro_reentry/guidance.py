from __future__ import annotations

from math import atan2, cos, degrees, radians, sin

from astro_reentry.models import ReentryGuidanceConfig, ReentryTarget


def normalize_longitude_deg(longitude_deg: float) -> float:
    return (longitude_deg + 180.0) % 360.0 - 180.0


def normalize_heading_deg(heading_deg: float) -> float:
    return heading_deg % 360.0


def signed_heading_error_deg(target_heading_deg: float, heading_deg: float) -> float:
    return (target_heading_deg - heading_deg + 180.0) % 360.0 - 180.0


def initial_bearing_deg(
    latitude_deg: float,
    longitude_deg: float,
    target: ReentryTarget,
) -> float:
    latitude = radians(latitude_deg)
    target_latitude = radians(target.latitude_deg)
    delta_longitude = radians(normalize_longitude_deg(target.longitude_deg - longitude_deg))
    y = sin(delta_longitude) * cos(target_latitude)
    x = cos(latitude) * sin(target_latitude) - sin(latitude) * cos(target_latitude) * cos(
        delta_longitude
    )
    return normalize_heading_deg(degrees(atan2(y, x)))


def schedule_bank_angle_deg(config: ReentryGuidanceConfig, velocity_km_s: float) -> float:
    if not config.bank_schedule:
        return float(config.bank_angle_deg)
    points = config.bank_schedule
    if velocity_km_s >= points[0].velocity_km_s:
        return float(points[0].bank_angle_deg)
    if velocity_km_s <= points[-1].velocity_km_s:
        return float(points[-1].bank_angle_deg)
    for high, low in zip(points, points[1:], strict=False):
        if high.velocity_km_s >= velocity_km_s >= low.velocity_km_s:
            fraction = (high.velocity_km_s - velocity_km_s) / (
                high.velocity_km_s - low.velocity_km_s
            )
            return float(
                high.bank_angle_deg + fraction * (low.bank_angle_deg - high.bank_angle_deg)
            )
    return float(points[-1].bank_angle_deg)


def commanded_bank_angle_deg(
    config: ReentryGuidanceConfig,
    *,
    velocity_km_s: float,
    latitude_deg: float,
    longitude_deg: float,
    heading_deg: float,
    target: ReentryTarget | None,
    previous_bank_sign: int,
) -> tuple[float, int]:
    if config.mode == "ballistic":
        return 0.0, 0
    if velocity_km_s <= config.minimum_control_velocity_km_s:
        return 0.0, previous_bank_sign
    scheduled = schedule_bank_angle_deg(config, velocity_km_s)
    if config.mode in {"constant_bank", "bank_schedule"}:
        sign = 1 if scheduled > 0.0 else -1 if scheduled < 0.0 else 0
        return scheduled, sign
    if target is None:
        raise ValueError("target_tracking guidance requires a target")
    target_heading = initial_bearing_deg(latitude_deg, longitude_deg, target)
    error = signed_heading_error_deg(target_heading, heading_deg)
    sign = previous_bank_sign or 1
    if error > config.heading_deadband_deg:
        sign = 1
    elif error < -config.heading_deadband_deg:
        sign = -1
    return abs(scheduled) * sign, sign
