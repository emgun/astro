from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.dymos.runtime import load_dymos_runtime
from astro_core.errors import UnsupportedBackendError


def test_load_dymos_runtime_reports_missing_dymos_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(distribution: str) -> str:
        if distribution == "dymos":
            raise PackageNotFoundError
        return "3.44.0"

    monkeypatch.setattr("astro_backends.dymos.runtime.version", missing_version)

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[optimization\]"):
        load_dymos_runtime()


def test_load_dymos_runtime_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "astro_backends.dymos.runtime.version",
        lambda distribution: "1.15.1" if distribution == "dymos" else "3.44.0",
    )

    def fail_import(_module_name: str) -> object:
        raise ImportError("dymos import failed")

    monkeypatch.setattr("astro_backends.dymos.runtime.import_module", fail_import)

    with pytest.raises(UnsupportedBackendError, match="Dymos/OpenMDAO import failed"):
        load_dymos_runtime()


def test_load_dymos_runtime_returns_required_api(
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

    runtime = load_dymos_runtime()

    assert runtime.dymos_version == "1.15.1"
    assert runtime.openmdao_version == "3.44.0"
    assert runtime.trajectory is object
    assert runtime.phase is object
    assert runtime.problem is object
