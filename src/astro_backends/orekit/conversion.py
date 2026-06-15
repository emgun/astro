from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from astro_core.errors import UnsupportedBackendError
from astro_core.models import Body, Frame, OrbitState, TimeScale

M_PER_KM = 1000.0


def km_to_m(value_km: float) -> float:
    return value_km * M_PER_KM


def m_to_km(value_m: float) -> float:
    return value_m / M_PER_KM


def km_s_to_m_s(value_km_s: float) -> float:
    return value_km_s * M_PER_KM


def m_s_to_km_s(value_m_s: float) -> float:
    return value_m_s / M_PER_KM


def validate_orekit_state_support(state: OrbitState) -> None:
    if state.time_scale is not TimeScale.UTC:
        raise UnsupportedBackendError("Orekit phase 1 supports only UTC scenario epochs")
    if state.frame is not Frame.EME2000:
        raise UnsupportedBackendError("Orekit phase 1 supports only EME2000 states")
    if state.central_body is not Body.EARTH:
        raise UnsupportedBackendError("Orekit phase 1 supports only Earth-centered states")


def absolute_date_from_datetime(runtime: Any, epoch: datetime) -> Any:
    utc_epoch = epoch.astimezone(timezone.utc)
    seconds_with_fraction = utc_epoch.second + utc_epoch.microsecond / 1_000_000.0
    utc = runtime.time_scales_factory.getUTC()
    return runtime.absolute_date(
        utc_epoch.year,
        utc_epoch.month,
        utc_epoch.day,
        utc_epoch.hour,
        utc_epoch.minute,
        seconds_with_fraction,
        utc,
    )
