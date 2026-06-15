from importlib.metadata import PackageNotFoundError

import pytest

from astro_backends.orekit.runtime import load_orekit_runtime
from astro_core.errors import UnsupportedBackendError


def test_load_orekit_runtime_reports_missing_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_version(distribution_name: str) -> str:
        raise PackageNotFoundError(distribution_name)

    monkeypatch.setattr("astro_backends.orekit.runtime.version", missing_version)

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[orekit\]"):
        load_orekit_runtime()


def test_load_orekit_runtime_reports_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("astro_backends.orekit.runtime.version", lambda _distribution: "13.1.0")

    def fail_import(module_name: str) -> object:
        raise ImportError(f"cannot import {module_name}")

    monkeypatch.setattr("astro_backends.orekit.runtime.import_module", fail_import)

    with pytest.raises(UnsupportedBackendError, match="Orekit JPype wrapper import failed"):
        load_orekit_runtime()
