from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from astro_core.errors import UnsupportedBackendError

PACKAGE = "tudatpy"
DISTRIBUTION = "tudatpy"


class TudatRuntimeUnavailable(UnsupportedBackendError):
    """Raised when the optional TudatPy runtime cannot be imported."""

    def __init__(self, message: str, *, package_version: str | None = None) -> None:
        super().__init__(message)
        self.package_version = package_version


@dataclass(frozen=True)
class TudatRuntime:
    package: str
    package_version: str
    module: Any


def _runtime_unavailable(
    message: str,
    *,
    package_version: str | None = None,
) -> TudatRuntimeUnavailable:
    return TudatRuntimeUnavailable(
        f"Tudat backend unavailable: {message}",
        package_version=package_version,
    )


def load_tudat_runtime(*, strict: bool = False) -> TudatRuntime:
    package_version: str | None
    metadata_error: PackageNotFoundError | None = None
    try:
        package_version = version(DISTRIBUTION)
    except PackageNotFoundError as exc:
        if strict:
            raise
        package_version = None
        metadata_error = exc

    try:
        tudat_module = import_module(PACKAGE)
    except ImportError as exc:
        if strict:
            raise
        if package_version is None:
            raise _runtime_unavailable(
                "TudatPy is not installed. Install TudatPy using its supported distribution "
                "channel for your platform before enabling the Tudat adapter.",
            ) from metadata_error or exc
        raise _runtime_unavailable(
            f"TudatPy import failed: {exc}",
            package_version=package_version,
        ) from exc

    if package_version is None:
        package_version = str(getattr(tudat_module, "__version__", "unknown"))

    return TudatRuntime(
        package=PACKAGE,
        package_version=package_version,
        module=tudat_module,
    )
