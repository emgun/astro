from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astro_backends.orekit.runtime import OrekitRuntime
from astro_core.models import Scenario

EXPONENTIAL_ATMOSPHERE_REFERENCE_DENSITY_KG_M3 = 4.0e-13
EXPONENTIAL_ATMOSPHERE_REFERENCE_ALTITUDE_M = 400_000.0
EXPONENTIAL_ATMOSPHERE_SCALE_HEIGHT_M = 60_000.0


@dataclass(frozen=True)
class OrekitForceModel:
    model: Any
    name: str
    metadata: dict[str, object]


def build_earth_shape(runtime: OrekitRuntime, earth_fixed_frame: Any) -> Any:
    return runtime.one_axis_ellipsoid(
        runtime.constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        runtime.constants.WGS84_EARTH_FLATTENING,
        earth_fixed_frame,
    )


def build_atmospheric_drag_force_model(
    scenario: Scenario,
    runtime: OrekitRuntime,
    earth_shape: Any,
) -> OrekitForceModel:
    atmosphere = runtime.simple_exponential_atmosphere(
        earth_shape,
        EXPONENTIAL_ATMOSPHERE_REFERENCE_DENSITY_KG_M3,
        EXPONENTIAL_ATMOSPHERE_REFERENCE_ALTITUDE_M,
        EXPONENTIAL_ATMOSPHERE_SCALE_HEIGHT_M,
    )
    spacecraft = runtime.isotropic_drag(
        scenario.spacecraft.area_m2,
        scenario.spacecraft.drag_coefficient,
    )
    return OrekitForceModel(
        model=runtime.drag_force(atmosphere, spacecraft),
        name="DragForce",
        metadata={
            "atmosphere_model": "SimpleExponentialAtmosphere",
            "atmosphere_reference_density_kg_m3": EXPONENTIAL_ATMOSPHERE_REFERENCE_DENSITY_KG_M3,
            "atmosphere_reference_altitude_m": EXPONENTIAL_ATMOSPHERE_REFERENCE_ALTITUDE_M,
            "atmosphere_scale_height_m": EXPONENTIAL_ATMOSPHERE_SCALE_HEIGHT_M,
            "drag_spacecraft_model": "IsotropicDrag",
            "drag_area_m2": scenario.spacecraft.area_m2,
            "drag_coefficient": scenario.spacecraft.drag_coefficient,
        },
    )
