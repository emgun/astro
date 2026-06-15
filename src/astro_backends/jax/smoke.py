from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from typing import cast

from astro_backends.jax.runtime import (
    PACKAGE,
    JaxRuntimeUnavailable,
    load_jax_runtime,
)


@dataclass(frozen=True)
class JaxSmokeResult:
    available: bool
    package: str
    version: str | None
    jaxlib_version: str | None
    message: str

    def to_dict(self) -> dict[str, bool | str | None]:
        return cast(dict[str, bool | str | None], asdict(self))


def _not_installed_result(
    *,
    jax_version: str | None = None,
    jaxlib_version: str | None = None,
) -> JaxSmokeResult:
    return JaxSmokeResult(
        available=False,
        package=PACKAGE,
        version=jax_version,
        jaxlib_version=jaxlib_version,
        message="JAX/JAXLIB is not installed.",
    )


def run_jax_smoke(
    *,
    strict: bool = False,
    force_unavailable: bool = False,
) -> JaxSmokeResult:
    if force_unavailable:
        return _not_installed_result()

    try:
        runtime = load_jax_runtime(strict=strict)
    except PackageNotFoundError:
        if strict:
            raise
        return _not_installed_result()
    except ImportError as exc:
        if strict:
            raise
        return JaxSmokeResult(
            available=False,
            package=PACKAGE,
            version="unknown",
            jaxlib_version="unknown",
            message=f"JAX import failed: {exc}",
        )
    except JaxRuntimeUnavailable as exc:
        message = str(exc)
        if "not installed" in message:
            return _not_installed_result(
                jax_version=exc.jax_version,
                jaxlib_version=exc.jaxlib_version,
            )
        return JaxSmokeResult(
            available=False,
            package=PACKAGE,
            version=exc.jax_version,
            jaxlib_version=exc.jaxlib_version,
            message=message,
        )

    return JaxSmokeResult(
        available=True,
        package=PACKAGE,
        version=runtime.jax_version,
        jaxlib_version=runtime.jaxlib_version,
        message="JAX and jax.numpy modules are available.",
    )
