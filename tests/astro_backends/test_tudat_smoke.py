from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.tudat.smoke import run_tudat_smoke


def test_run_tudat_smoke_reports_forced_unavailable() -> None:
    result = run_tudat_smoke(force_unavailable=True)

    assert result.available is False
    assert result.package == "tudatpy"
    assert result.version is None
    assert "not installed" in result.message


def test_run_tudat_smoke_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.tudat.runtime.version", missing_version)

    result = run_tudat_smoke()

    assert result.available is False
    assert result.version is None
    assert "not installed" in result.message


def test_run_tudat_smoke_reports_available_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("astro_backends.tudat.runtime.version", lambda _distribution: "1.0.0")
    monkeypatch.setattr(
        "astro_backends.tudat.runtime.import_module",
        lambda _module_name: SimpleNamespace(),
    )

    result = run_tudat_smoke()

    assert result.available is True
    assert result.version == "1.0.0"
    assert "available" in result.message


def test_run_tudat_smoke_accepts_conda_module_without_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.tudat.runtime.version", missing_version)
    monkeypatch.setattr(
        "astro_backends.tudat.runtime.import_module",
        lambda _module_name: SimpleNamespace(__version__="1.0.0"),
    )

    result = run_tudat_smoke()

    assert result.available is True
    assert result.version == "1.0.0"
    assert "available" in result.message
