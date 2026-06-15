from types import SimpleNamespace

import numpy as np
import pytest

from astro_backends.jax.propagation import research_propagate_jax
from astro_backends.jax.runtime import JaxRuntime
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName, Scenario
from astro_dynamics.local import propagate_local
from astro_dynamics.monte_carlo import MonteCarloResult, run_initial_state_monte_carlo


def _fake_runtime() -> JaxRuntime:
    return JaxRuntime(
        jax_version="0.10.1",
        jaxlib_version="0.10.1",
        jax_module=SimpleNamespace(),
        jnp_module=np,
    )


def test_research_propagate_jax_reports_runtime_unavailable() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")

    def fail_runtime() -> JaxRuntime:
        raise UnsupportedBackendError("JAX backend unavailable: install astro-suite[research]")

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[research\]"):
        research_propagate_jax(
            scenario,
            cases=2,
            position_sigma_km=0.01,
            velocity_sigma_km_s=0.000001,
            seed=7,
            runtime_loader=fail_runtime,
        )


def test_research_propagate_jax_runs_builtin_two_body_runner() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    local_trajectory = propagate_local(scenario)

    result = research_propagate_jax(
        scenario,
        cases=1,
        position_sigma_km=0.0,
        velocity_sigma_km_s=0.0,
        seed=7,
        runtime_loader=_fake_runtime,
    )

    final_jax = result.cases[0].trajectory.samples[-1].state
    final_local = local_trajectory.samples[-1].state
    assert result.backend == "jax"
    assert result.metadata["runner"] == "jax_vectorized_two_body_rk4"
    assert result.metadata["jax_version"] == "0.10.1"
    assert result.cases[0].trajectory.backend == "jax"
    assert final_jax.position_km == pytest.approx(final_local.position_km)
    assert final_jax.velocity_km_s == pytest.approx(final_local.velocity_km_s)


def test_research_propagate_jax_rejects_unsupported_force_model() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml").model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.J2)}
    )

    with pytest.raises(UnsupportedBackendError, match="supports only two_body"):
        research_propagate_jax(
            scenario,
            cases=1,
            position_sigma_km=0.0,
            velocity_sigma_km_s=0.0,
            seed=7,
            runtime_loader=_fake_runtime,
        )


def test_research_propagate_jax_returns_monte_carlo_product_with_fake_runner() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    seen_runtime: list[JaxRuntime] = []

    def fake_runner(
        candidate: Scenario,
        runtime: JaxRuntime,
        cases: int,
        position_sigma_km: float,
        velocity_sigma_km_s: float,
        seed: int,
    ) -> MonteCarloResult:
        assert candidate is scenario
        seen_runtime.append(runtime)
        return run_initial_state_monte_carlo(
            candidate,
            cases=cases,
            position_sigma_km=position_sigma_km,
            velocity_sigma_km_s=velocity_sigma_km_s,
            seed=seed,
            backend="local",
        )

    result = research_propagate_jax(
        scenario,
        cases=2,
        position_sigma_km=0.01,
        velocity_sigma_km_s=0.000001,
        seed=7,
        runtime_loader=_fake_runtime,
        research_runner=fake_runner,
    )

    assert len(seen_runtime) == 1
    assert result.backend == "jax"
    assert result.metadata["adapter"] == "jax"
    assert result.metadata["source_backend"] == "local"
    assert result.metadata["jax_version"] == "0.10.1"
    assert result.metadata["jaxlib_version"] == "0.10.1"
