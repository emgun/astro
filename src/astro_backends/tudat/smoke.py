from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from typing import cast

from astro_backends.tudat.runtime import (
    PACKAGE,
    TudatRuntimeUnavailable,
    load_tudat_runtime,
)


@dataclass(frozen=True)
class TudatSmokeResult:
    available: bool
    package: str
    version: str | None
    message: str

    def to_dict(self) -> dict[str, bool | str | None]:
        return cast(dict[str, bool | str | None], asdict(self))


def _not_installed_result(package_version: str | None = None) -> TudatSmokeResult:
    return TudatSmokeResult(
        available=False,
        package=PACKAGE,
        version=package_version,
        message="TudatPy is not installed.",
    )


def run_tudat_smoke(
    *,
    strict: bool = False,
    force_unavailable: bool = False,
) -> TudatSmokeResult:
    if force_unavailable:
        return _not_installed_result()

    try:
        runtime = load_tudat_runtime(strict=strict)
    except PackageNotFoundError:
        if strict:
            raise
        return _not_installed_result()
    except ImportError as exc:
        if strict:
            raise
        return TudatSmokeResult(
            available=False,
            package=PACKAGE,
            version="unknown",
            message=f"TudatPy import failed: {exc}",
        )
    except TudatRuntimeUnavailable as exc:
        message = str(exc)
        if "not installed" in message:
            return _not_installed_result(exc.package_version)
        return TudatSmokeResult(
            available=False,
            package=PACKAGE,
            version=exc.package_version,
            message=message,
        )

    return TudatSmokeResult(
        available=True,
        package=PACKAGE,
        version=runtime.package_version,
        message="TudatPy module is available.",
    )
