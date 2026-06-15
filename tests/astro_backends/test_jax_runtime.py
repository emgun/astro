from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from astro_backends.jax.runtime import load_jax_runtime
from astro_core.errors import UnsupportedBackendError


def test_load_jax_runtime_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("astro_backends.jax.runtime.version", missing_version)

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[research\]"):
        load_jax_runtime()


def test_load_jax_runtime_reports_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "astro_backends.jax.runtime.version",
        lambda distribution: "0.10.1" if distribution == "jax" else "0.10.1",
    )

    def fail_import(_module_name: str) -> object:
        raise ImportError("jax import failed")

    monkeypatch.setattr("astro_backends.jax.runtime.import_module", fail_import)

    with pytest.raises(UnsupportedBackendError, match="JAX import failed"):
        load_jax_runtime()


def test_load_jax_runtime_returns_modules(monkeypatch: pytest.MonkeyPatch) -> None:
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

    runtime = load_jax_runtime()

    assert runtime.jax_version == "0.10.1"
    assert runtime.jaxlib_version == "0.10.1"
    assert runtime.jax_module is jax_module
    assert runtime.jnp_module is jnp_module
