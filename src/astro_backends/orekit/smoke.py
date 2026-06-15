from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, cast

WRAPPER = "orekit_jpype"
DISTRIBUTION = "orekit-jpype"


class _OrekitModule(Protocol):
    def initVM(self) -> object: ...


class _FramesFactory(Protocol):
    def getEME2000(self) -> object: ...


class _TimeScalesFactory(Protocol):
    def getUTC(self) -> object: ...


class _FramesModule(Protocol):
    FramesFactory: _FramesFactory


class _TimeModule(Protocol):
    TimeScalesFactory: _TimeScalesFactory


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


def run_orekit_smoke(
    *,
    strict: bool = False,
    force_unavailable: bool = False,
) -> OrekitSmokeResult:
    if force_unavailable:
        return _not_installed_result()

    wrapper_version: str | None = None
    try:
        wrapper_version = version(DISTRIBUTION)
        orekit = cast(_OrekitModule, import_module(WRAPPER))
    except (ImportError, PackageNotFoundError):
        if strict:
            raise
        return _not_installed_result(wrapper_version)

    try:
        orekit.initVM()
        frames_module = cast(_FramesModule, import_module("org.orekit.frames"))
        time_module = cast(_TimeModule, import_module("org.orekit.time"))
        frames_module.FramesFactory.getEME2000()
        time_module.TimeScalesFactory.getUTC()
    except Exception as exc:
        if strict:
            raise
        return OrekitSmokeResult(
            available=False,
            wrapper=WRAPPER,
            version=wrapper_version,
            message=f"Orekit JPype VM/frame/time smoke failure: {exc}",
        )

    return OrekitSmokeResult(
        available=True,
        wrapper=WRAPPER,
        version=wrapper_version,
        message="Orekit JPype VM, EME2000 frame, and UTC time scale are available.",
    )
