from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from astro_backends.import_timeout import (
    DEFAULT_OPTIONAL_IMPORT_TIMEOUT_S,
    import_optional_module,
)
from astro_core.errors import UnsupportedBackendError

PACKAGE = "rocketpy"
DISTRIBUTION = "rocketpy"


class RocketPyRuntimeUnavailable(UnsupportedBackendError):
    """Raised when the optional RocketPy runtime cannot be imported."""

    def __init__(self, message: str, *, package_version: str | None = None) -> None:
        super().__init__(message)
        self.package_version = package_version


@dataclass(frozen=True)
class RocketPyRuntime:
    package: str
    package_version: str
    module: Any
    environment: Any
    solid_motor: Any
    rocket: Any
    flight: Any


def _runtime_unavailable(
    message: str,
    *,
    package_version: str | None = None,
) -> RocketPyRuntimeUnavailable:
    return RocketPyRuntimeUnavailable(
        f"RocketPy backend unavailable: {message}",
        package_version=package_version,
    )


def load_rocketpy_runtime(
    *,
    strict: bool = False,
    import_timeout_s: float = DEFAULT_OPTIONAL_IMPORT_TIMEOUT_S,
) -> RocketPyRuntime:
    try:
        package_version = version(DISTRIBUTION)
    except PackageNotFoundError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            "RocketPy is not installed; install astro-suite[launch] to enable RocketPy.",
        ) from exc

    try:
        rocketpy_module = import_optional_module(
            PACKAGE,
            import_module,
            timeout_s=import_timeout_s,
        )
    except ImportError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"RocketPy import failed: {exc}",
            package_version=package_version,
        ) from exc
    except TimeoutError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"RocketPy import timed out: {exc}",
            package_version=package_version,
        ) from exc

    try:
        environment = rocketpy_module.Environment
        solid_motor = rocketpy_module.SolidMotor
        rocket = rocketpy_module.Rocket
        flight = rocketpy_module.Flight
    except AttributeError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"RocketPy required API is unavailable: {exc}",
            package_version=package_version,
        ) from exc

    return RocketPyRuntime(
        package=PACKAGE,
        package_version=package_version,
        module=rocketpy_module,
        environment=environment,
        solid_motor=solid_motor,
        rocket=rocket,
        flight=flight,
    )
