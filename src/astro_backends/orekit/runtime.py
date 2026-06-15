from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from astro_core.errors import UnsupportedBackendError

WRAPPER = "orekit_jpype"
DISTRIBUTION = "orekit-jpype"


class OrekitRuntimeUnavailable(UnsupportedBackendError):
    """Raised when the optional Orekit runtime cannot be initialized."""

    def __init__(self, message: str, *, wrapper_version: str | None = None) -> None:
        super().__init__(message)
        self.wrapper_version = wrapper_version


@dataclass(frozen=True)
class OrekitRuntime:
    wrapper: str
    wrapper_version: str
    frames_factory: Any
    time_scales_factory: Any
    absolute_date: Any
    vector3d: Any
    pv_coordinates: Any
    cartesian_orbit: Any
    keplerian_propagator: Any


def _runtime_unavailable(
    message: str,
    *,
    wrapper_version: str | None = None,
) -> OrekitRuntimeUnavailable:
    return OrekitRuntimeUnavailable(
        f"Orekit backend unavailable: {message}",
        wrapper_version=wrapper_version,
    )


def load_orekit_runtime(*, strict: bool = False) -> OrekitRuntime:
    try:
        wrapper_version = version(DISTRIBUTION)
    except PackageNotFoundError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            "Orekit JPype wrapper is not installed; install astro-suite[orekit] "
            "to enable orekit-jpype.",
        ) from exc

    try:
        orekit = import_module(WRAPPER)
    except ImportError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"Orekit JPype wrapper import failed: {exc}",
            wrapper_version=wrapper_version,
        ) from exc

    try:
        orekit.initVM()
        frames_module = import_module("org.orekit.frames")
        time_module = import_module("org.orekit.time")
        geometry_module = import_module("org.hipparchus.geometry.euclidean.threed")
        utils_module = import_module("org.orekit.utils")
        orbits_module = import_module("org.orekit.orbits")
        propagation_module = import_module("org.orekit.propagation.analytical")
    except Exception as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"JVM, Orekit imports, or data context failed: {exc}",
            wrapper_version=wrapper_version,
        ) from exc

    return OrekitRuntime(
        wrapper=WRAPPER,
        wrapper_version=wrapper_version,
        frames_factory=frames_module.FramesFactory,
        time_scales_factory=time_module.TimeScalesFactory,
        absolute_date=time_module.AbsoluteDate,
        vector3d=geometry_module.Vector3D,
        pv_coordinates=utils_module.PVCoordinates,
        cartesian_orbit=orbits_module.CartesianOrbit,
        keplerian_propagator=propagation_module.KeplerianPropagator,
    )
