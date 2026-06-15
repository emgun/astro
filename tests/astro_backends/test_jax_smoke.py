from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.jax.smoke import run_jax_smoke


def test_run_jax_smoke_reports_forced_unavailable() -> None:
    result = run_jax_smoke(force_unavailable=True)

    assert result.available is False
    assert result.package == "jax"
    assert result.version is None
    assert result.jaxlib_version is None
    assert "not installed" in result.message


def test_run_jax_smoke_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.jax.runtime.version", missing_version)

    result = run_jax_smoke()

    assert result.available is False
    assert result.version is None
    assert "not installed" in result.message


def test_run_jax_smoke_reports_available_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    jax_module = SimpleNamespace()
    jnp_module = SimpleNamespace()
    monkeypatch.setattr(
        "astro_backends.jax.runtime.version",
        lambda distribution: "0.10.1" if distribution == "jax" else "0.10.1",
    )
    monkeypatch.setattr(
        "astro_backends.jax.runtime.import_module",
        lambda module_name: jnp_module if module_name == "jax.numpy" else jax_module,
    )

    result = run_jax_smoke()

    assert result.available is True
    assert result.version == "0.10.1"
    assert result.jaxlib_version == "0.10.1"
    assert "available" in result.message
