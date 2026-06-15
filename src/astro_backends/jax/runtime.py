from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from astro_core.errors import UnsupportedBackendError

PACKAGE = "jax"
JAXLIB_DISTRIBUTION = "jaxlib"
JAX_NUMPY_MODULE = "jax.numpy"


class JaxRuntimeUnavailable(UnsupportedBackendError):
    """Raised when the optional JAX runtime cannot be imported."""

    def __init__(
        self,
        message: str,
        *,
        jax_version: str | None = None,
        jaxlib_version: str | None = None,
    ) -> None:
        super().__init__(message)
        self.jax_version = jax_version
        self.jaxlib_version = jaxlib_version


@dataclass(frozen=True)
class JaxRuntime:
    jax_version: str
    jaxlib_version: str
    jax_module: Any
    jnp_module: Any


def _runtime_unavailable(
    message: str,
    *,
    jax_version: str | None = None,
    jaxlib_version: str | None = None,
) -> JaxRuntimeUnavailable:
    return JaxRuntimeUnavailable(
        f"JAX backend unavailable: {message}",
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
    )


def _load_versions(*, strict: bool) -> tuple[str, str]:
    try:
        jax_version = version(PACKAGE)
        jaxlib_version = version(JAXLIB_DISTRIBUTION)
    except PackageNotFoundError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            "JAX/JAXLIB is not installed; install astro-suite[research] to enable "
            "JAX research propagation.",
        ) from exc
    return jax_version, jaxlib_version


def load_jax_runtime(*, strict: bool = False) -> JaxRuntime:
    jax_version, jaxlib_version = _load_versions(strict=strict)

    try:
        jax_module = import_module(PACKAGE)
        jnp_module = import_module(JAX_NUMPY_MODULE)
    except ImportError as exc:
        if strict:
            raise
        raise _runtime_unavailable(
            f"JAX import failed: {exc}",
            jax_version=jax_version,
            jaxlib_version=jaxlib_version,
        ) from exc

    return JaxRuntime(
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
        jax_module=jax_module,
        jnp_module=jnp_module,
    )
