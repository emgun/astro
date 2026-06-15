from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.dymos.smoke import run_dymos_smoke


def test_run_dymos_smoke_reports_forced_unavailable() -> None:
    result = run_dymos_smoke(force_unavailable=True)

    assert result.available is False
    assert result.package == "dymos"
    assert result.version is None
    assert result.openmdao_version is None
    assert "not installed" in result.message


def test_run_dymos_smoke_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.dymos.runtime.version", missing_version)

    result = run_dymos_smoke()

    assert result.available is False
    assert result.version is None
    assert "not installed" in result.message


def test_run_dymos_smoke_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "astro_backends.dymos.runtime.version",
        lambda distribution: "1.15.1" if distribution == "dymos" else "3.44.0",
    )

    def fail_import(_module_name: str) -> object:
        raise ImportError("dymos import failed")

    monkeypatch.setattr("astro_backends.dymos.runtime.import_module", fail_import)

    result = run_dymos_smoke()

    assert result.available is False
    assert result.version == "1.15.1"
    assert result.openmdao_version == "3.44.0"
    assert "Dymos/OpenMDAO import failed" in result.message


def test_run_dymos_smoke_reports_required_api_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dymos_module = SimpleNamespace(Trajectory=object, Phase=object, GaussLobatto=object)
    openmdao_module = SimpleNamespace(Problem=object)
    monkeypatch.setattr(
        "astro_backends.dymos.runtime.version",
        lambda distribution: "1.15.1" if distribution == "dymos" else "3.44.0",
    )
    monkeypatch.setattr(
        "astro_backends.dymos.runtime.import_module",
        lambda module_name: dymos_module if module_name == "dymos" else openmdao_module,
    )

    result = run_dymos_smoke()

    assert result.available is True
    assert result.version == "1.15.1"
    assert result.openmdao_version == "3.44.0"
    assert "available" in result.message
