from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from typing import cast

from astro_backends.dymos.runtime import (
    PACKAGE,
    DymosRuntimeUnavailable,
    load_dymos_runtime,
)


@dataclass(frozen=True)
class DymosSmokeResult:
    available: bool
    package: str
    version: str | None
    openmdao_version: str | None
    message: str

    def to_dict(self) -> dict[str, bool | str | None]:
        return cast(dict[str, bool | str | None], asdict(self))


def _not_installed_result(
    *,
    dymos_version: str | None = None,
    openmdao_version: str | None = None,
) -> DymosSmokeResult:
    return DymosSmokeResult(
        available=False,
        package=PACKAGE,
        version=dymos_version,
        openmdao_version=openmdao_version,
        message="Dymos/OpenMDAO is not installed.",
    )


def run_dymos_smoke(
    *,
    strict: bool = False,
    force_unavailable: bool = False,
) -> DymosSmokeResult:
    if force_unavailable:
        return _not_installed_result()

    try:
        runtime = load_dymos_runtime(strict=strict)
    except PackageNotFoundError:
        if strict:
            raise
        return _not_installed_result()
    except ImportError as exc:
        if strict:
            raise
        return DymosSmokeResult(
            available=False,
            package=PACKAGE,
            version="unknown",
            openmdao_version="unknown",
            message=f"Dymos/OpenMDAO import failed: {exc}",
        )
    except DymosRuntimeUnavailable as exc:
        message = str(exc)
        if "not installed" in message:
            return _not_installed_result(
                dymos_version=exc.dymos_version,
                openmdao_version=exc.openmdao_version,
            )
        return DymosSmokeResult(
            available=False,
            package=PACKAGE,
            version=exc.dymos_version,
            openmdao_version=exc.openmdao_version,
            message=message,
        )

    return DymosSmokeResult(
        available=True,
        package=PACKAGE,
        version=runtime.dymos_version,
        openmdao_version=runtime.openmdao_version,
        message="Dymos Trajectory/Phase and OpenMDAO Problem APIs are available.",
    )
