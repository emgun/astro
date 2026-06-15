from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from typing import cast

from astro_backends.orekit.runtime import (
    WRAPPER,
    OrekitRuntimeUnavailable,
    load_orekit_runtime,
)


@dataclass(frozen=True)
class OrekitSmokeResult:
    available: bool
    wrapper: str
    version: str | None
    message: str

    def to_dict(self) -> dict[str, bool | str | None]:
        return cast(dict[str, bool | str | None], asdict(self))


def _not_installed_result(wrapper_version: str | None = None) -> OrekitSmokeResult:
    return OrekitSmokeResult(
        available=False,
        wrapper=WRAPPER,
        version=wrapper_version,
        message="Orekit JPype wrapper is not installed.",
    )


def _import_failed_result(wrapper_version: str, exc: ImportError) -> OrekitSmokeResult:
    return OrekitSmokeResult(
        available=False,
        wrapper=WRAPPER,
        version=wrapper_version,
        message=f"Orekit JPype wrapper import failed: {exc}",
    )


def run_orekit_smoke(
    *,
    strict: bool = False,
    force_unavailable: bool = False,
) -> OrekitSmokeResult:
    if force_unavailable:
        return _not_installed_result()

    try:
        runtime = load_orekit_runtime(strict=strict)
    except PackageNotFoundError:
        if strict:
            raise
        return _not_installed_result()
    except ImportError as exc:
        if strict:
            raise
        return _import_failed_result("unknown", exc)
    except OrekitRuntimeUnavailable as exc:
        message = str(exc)
        if "not installed" in message:
            return _not_installed_result(exc.wrapper_version)
        return OrekitSmokeResult(
            available=False,
            wrapper=WRAPPER,
            version=exc.wrapper_version,
            message=message,
        )

    try:
        runtime.frames_factory.getEME2000()
        runtime.time_scales_factory.getUTC()
    except Exception as exc:
        if strict:
            raise
        return OrekitSmokeResult(
            available=False,
            wrapper=WRAPPER,
            version=runtime.wrapper_version,
            message=f"Orekit JPype VM/frame/time smoke failure: {exc}",
        )

    return OrekitSmokeResult(
        available=True,
        wrapper=WRAPPER,
        version=runtime.wrapper_version,
        message="Orekit JPype VM, EME2000 frame, and UTC time scale are available.",
    )
