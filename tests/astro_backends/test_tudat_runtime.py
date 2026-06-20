from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.tudat.runtime import load_tudat_runtime
from astro_core.errors import UnsupportedBackendError


def test_load_tudat_runtime_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.tudat.runtime.version", missing_version)

    with pytest.raises(UnsupportedBackendError, match="TudatPy is not installed"):
        load_tudat_runtime()


def test_load_tudat_runtime_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("astro_backends.tudat.runtime.version", lambda _distribution: "1.0.0")

    def fail_import(_module_name: str) -> object:
        raise ImportError("tudat import failed")

    monkeypatch.setattr("astro_backends.tudat.runtime.import_module", fail_import)

    with pytest.raises(UnsupportedBackendError, match="TudatPy import failed"):
        load_tudat_runtime()


def test_load_tudat_runtime_returns_module(monkeypatch: pytest.MonkeyPatch) -> None:
    tudat_module = SimpleNamespace()
    monkeypatch.setattr("astro_backends.tudat.runtime.version", lambda _distribution: "1.0.0")
    monkeypatch.setattr(
        "astro_backends.tudat.runtime.import_module",
        lambda _module_name: tudat_module,
    )

    runtime = load_tudat_runtime()

    assert runtime.package == "tudatpy"
    assert runtime.package_version == "1.0.0"
    assert runtime.module is tudat_module


def test_load_tudat_runtime_accepts_conda_module_without_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    tudat_module = SimpleNamespace(__version__="1.0.0")
    monkeypatch.setattr("astro_backends.tudat.runtime.version", missing_version)
    monkeypatch.setattr(
        "astro_backends.tudat.runtime.import_module",
        lambda _module_name: tudat_module,
    )

    runtime = load_tudat_runtime()

    assert runtime.package == "tudatpy"
    assert runtime.package_version == "1.0.0"
    assert runtime.module is tudat_module


def test_load_tudat_runtime_strict_preserves_missing_distribution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.tudat.runtime.version", missing_version)
    monkeypatch.setattr(
        "astro_backends.tudat.runtime.import_module",
        lambda _module_name: SimpleNamespace(__version__="1.0.0"),
    )

    with pytest.raises(PackageNotFoundError):
        load_tudat_runtime(strict=True)
