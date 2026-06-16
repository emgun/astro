from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from astro_core.errors import UnsupportedBackendError

WRAPPER = "orekit_jpype"
DISTRIBUTION = "orekit-jpype"
OREKIT_DATA_ENV_VARS = ("ASTRO_OREKIT_DATA_PATH", "OREKIT_DATA_PATH")
DEFAULT_OREKIT_DATA_PATH = Path.home() / ".orekit" / "orekit-data.zip"


class OrekitRuntimeUnavailable(UnsupportedBackendError):
    """Raised when the optional Orekit runtime cannot be initialized."""

    def __init__(self, message: str, *, wrapper_version: str | None = None) -> None:
        super().__init__(message)
        self.wrapper_version = wrapper_version


@dataclass(frozen=True)
class OrekitRuntime:
    wrapper: str
    wrapper_version: str
    data_path: str
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


def _configured_data_path() -> Path:
    for env_var in OREKIT_DATA_ENV_VARS:
        configured_path = os.environ.get(env_var)
        if configured_path:
            return Path(configured_path).expanduser()
    return DEFAULT_OREKIT_DATA_PATH


def _missing_data_error(data_path: Path, *, wrapper_version: str) -> OrekitRuntimeUnavailable:
    return _runtime_unavailable(
        f"Orekit data path {data_path} does not exist; download orekit-data.zip "
        "or set ASTRO_OREKIT_DATA_PATH to a valid Orekit data zip/folder.",
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

    data_path = _configured_data_path()
    if not data_path.exists():
        if strict:
            raise FileNotFoundError(data_path)
        raise _missing_data_error(data_path, wrapper_version=wrapper_version)

    try:
        orekit.initVM()
        pyhelpers_module = import_module("orekit_jpype.pyhelpers")
        pyhelpers_module.setup_orekit_data(
            filenames=str(data_path),
            from_pip_library=False,
        )
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
        data_path=str(data_path),
        frames_factory=frames_module.FramesFactory,
        time_scales_factory=time_module.TimeScalesFactory,
        absolute_date=time_module.AbsoluteDate,
        vector3d=geometry_module.Vector3D,
        pv_coordinates=utils_module.PVCoordinates,
        cartesian_orbit=orbits_module.CartesianOrbit,
        keplerian_propagator=propagation_module.KeplerianPropagator,
    )
