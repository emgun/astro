from __future__ import annotations

from collections.abc import Callable

from astro_backends.jax.runtime import JaxRuntime, load_jax_runtime
from astro_core.errors import UnsupportedBackendError
from astro_core.models import Scenario
from astro_dynamics.monte_carlo import MonteCarloResult

JaxRuntimeLoader = Callable[[], JaxRuntime]
JaxResearchRunner = Callable[[Scenario, JaxRuntime, int, float, float, int], MonteCarloResult]


def _with_jax_provenance(
    result: MonteCarloResult,
    runtime: JaxRuntime,
) -> MonteCarloResult:
    metadata = {
        **result.metadata,
        "adapter": "jax",
        "source_backend": result.backend,
        "jax_version": runtime.jax_version,
        "jaxlib_version": runtime.jaxlib_version,
    }
    return result.model_copy(update={"backend": "jax", "metadata": metadata})


def research_propagate_jax(
    scenario: Scenario,
    *,
    cases: int,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
    seed: int,
    runtime_loader: JaxRuntimeLoader = load_jax_runtime,
    research_runner: JaxResearchRunner | None = None,
) -> MonteCarloResult:
    runtime = runtime_loader()
    if research_runner is None:
        raise UnsupportedBackendError(
            "JAX research propagation requires a validated JAX research runner; "
            "use --backend local for the current seeded Monte Carlo workflow."
        )

    result = research_runner(
        scenario,
        runtime,
        cases,
        position_sigma_km,
        velocity_sigma_km_s,
        seed,
    )
    return _with_jax_provenance(result, runtime)
