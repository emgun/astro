from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.rocketpy.runtime import load_rocketpy_runtime
from astro_core.errors import UnsupportedBackendError


def test_load_rocketpy_runtime_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.rocketpy.runtime.version", missing_version)

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[launch\]"):
        load_rocketpy_runtime()


def test_load_rocketpy_runtime_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("astro_backends.rocketpy.runtime.version", lambda _distribution: "1.12.1")

    def fail_import(_module_name: str) -> object:
        raise ImportError("rocketpy import failed")

    monkeypatch.setattr("astro_backends.rocketpy.runtime.import_module", fail_import)

    with pytest.raises(UnsupportedBackendError, match="RocketPy import failed"):
        load_rocketpy_runtime()


def test_load_rocketpy_runtime_reports_import_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("astro_backends.rocketpy.runtime.version", lambda _distribution: "1.12.1")

    def timeout_import(_module_name: str) -> object:
        raise TimeoutError("timed out importing optional backend module")

    monkeypatch.setattr("astro_backends.rocketpy.runtime.import_module", timeout_import)

    with pytest.raises(UnsupportedBackendError, match="import timed out"):
        load_rocketpy_runtime()


def test_load_rocketpy_runtime_returns_required_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rocketpy_module = SimpleNamespace(
        Environment=object,
        SolidMotor=object,
        Rocket=object,
        Flight=object,
    )
    monkeypatch.setattr("astro_backends.rocketpy.runtime.version", lambda _distribution: "1.12.1")
    monkeypatch.setattr(
        "astro_backends.rocketpy.runtime.import_module",
        lambda module_name: rocketpy_module,
    )

    runtime = load_rocketpy_runtime()

    assert runtime.package == "rocketpy"
    assert runtime.package_version == "1.12.1"
    assert runtime.environment is object
    assert runtime.solid_motor is object
    assert runtime.rocket is object
    assert runtime.flight is object
