from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.rocketpy.smoke import run_rocketpy_smoke


def test_run_rocketpy_smoke_reports_forced_unavailable() -> None:
    result = run_rocketpy_smoke(force_unavailable=True)

    assert result.available is False
    assert result.package == "rocketpy"
    assert result.version is None
    assert "not installed" in result.message


def test_run_rocketpy_smoke_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.rocketpy.runtime.version", missing_version)

    result = run_rocketpy_smoke()

    assert result.available is False
    assert result.version is None
    assert "not installed" in result.message


def test_run_rocketpy_smoke_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("astro_backends.rocketpy.runtime.version", lambda _distribution: "1.12.1")

    def fail_import(_module_name: str) -> object:
        raise ImportError("rocketpy import failed")

    monkeypatch.setattr("astro_backends.rocketpy.runtime.import_module", fail_import)

    result = run_rocketpy_smoke()

    assert result.available is False
    assert result.version == "1.12.1"
    assert "RocketPy import failed" in result.message


def test_run_rocketpy_smoke_reports_required_api_available(
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

    result = run_rocketpy_smoke()

    assert result.available is True
    assert result.version == "1.12.1"
    assert "available" in result.message
