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

PACKAGE = "dymos"
OPENMDAO_PACKAGE = "openmdao"
OPENMDAO_API_MODULE = "openmdao.api"


class DymosRuntimeUnavailable(UnsupportedBackendError):
    """Raised when the optional Dymos/OpenMDAO runtime cannot be imported."""

    def __init__(
        self,
        message: str,
        *,
        dymos_version: str | None = None,
        openmdao_version: str | None = None,
    ) -> None:
        super().__init__(message)
        self.dymos_version = dymos_version
        self.openmdao_version = openmdao_version


@dataclass(frozen=True)
class DymosRuntime:
    dymos_version: str
    openmdao_version: str
    dymos_module: Any
    openmdao_module: Any
    trajectory: Any
    phase: Any
    transcription: Any
    problem: Any


def _runtime_unavailable(
    message: str,
    *,
    dymos_version: str | None = None,
    openmdao_version: str | None = None,
) -> DymosRuntimeUnavailable:
    return DymosRuntimeUnavailable(
        f"Dymos backend unavailable: {message}",
        dymos_version=dymos_version,
        openmdao_version=openmdao_version,
    )


def _load_versions(*, strict: bool) -> tuple[str, str]:
    try:
        dymos_version = version(PACKAGE)
        openmdao_version = version(OPENMDAO_PACKAGE)
    except PackageNotFoundError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            "Dymos/OpenMDAO is not installed; install astro-suite[optimization] "
            "to enable Dymos launch optimization.",
        ) from exc
    return dymos_version, openmdao_version


def load_dymos_runtime(
    *,
    strict: bool = False,
    import_timeout_s: float = DEFAULT_OPTIONAL_IMPORT_TIMEOUT_S,
) -> DymosRuntime:
    dymos_version, openmdao_version = _load_versions(strict=strict)

    try:
        dymos_module = import_optional_module(
            PACKAGE,
            import_module,
            timeout_s=import_timeout_s,
        )
        openmdao_module = import_optional_module(
            OPENMDAO_API_MODULE,
            import_module,
            timeout_s=import_timeout_s,
        )
    except ImportError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"Dymos/OpenMDAO import failed: {exc}",
            dymos_version=dymos_version,
            openmdao_version=openmdao_version,
        ) from exc
    except TimeoutError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"Dymos/OpenMDAO import timed out: {exc}",
            dymos_version=dymos_version,
            openmdao_version=openmdao_version,
        ) from exc

    try:
        trajectory = dymos_module.Trajectory
        phase = dymos_module.Phase
        transcription = dymos_module.GaussLobatto
        problem = openmdao_module.Problem
    except AttributeError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"Dymos/OpenMDAO required API is unavailable: {exc}",
            dymos_version=dymos_version,
            openmdao_version=openmdao_version,
        ) from exc

    return DymosRuntime(
        dymos_version=dymos_version,
        openmdao_version=openmdao_version,
        dymos_module=dymos_module,
        openmdao_module=openmdao_module,
        trajectory=trajectory,
        phase=phase,
        transcription=transcription,
        problem=problem,
    )
