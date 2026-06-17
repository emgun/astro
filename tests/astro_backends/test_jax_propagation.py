from types import SimpleNamespace

import numpy as np
import pytest

from astro_backends.jax.propagation import (
    research_estimate_jax,
    research_od_sensitivity_jax,
    research_propagate_jax,
)
from astro_backends.jax.runtime import JaxRuntime, load_jax_runtime
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName, Scenario
from astro_dynamics.local import propagate_local
from astro_dynamics.monte_carlo import MonteCarloResult, run_initial_state_monte_carlo
from astro_od.measurements import generate_synthetic_measurements


def _fake_runtime() -> JaxRuntime:
    return JaxRuntime(
        jax_version="0.10.1",
        jaxlib_version="0.10.1",
        jax_module=SimpleNamespace(),
        jnp_module=np,
    )


def _fake_autodiff_runtime() -> JaxRuntime:
    def jacfwd(function: object) -> object:
        def jacobian(state_vector: object) -> np.ndarray:
            state = np.asarray(state_vector, dtype=np.float64)
            columns = []
            for index in range(state.size):
                step = max(abs(float(state[index])) * 1.0e-6, 1.0e-8)
                perturbation = np.zeros_like(state)
                perturbation[index] = step
                plus = np.asarray(function(state + perturbation), dtype=np.float64)
                minus = np.asarray(function(state - perturbation), dtype=np.float64)
                columns.append((plus - minus) / (2.0 * step))
            return np.stack(columns, axis=1)

        return jacobian

    return JaxRuntime(
        jax_version="0.10.1",
        jaxlib_version="0.10.1",
        jax_module=SimpleNamespace(jacfwd=jacfwd),
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


def test_research_propagate_jax_runs_builtin_j2_runner() -> None:
    scenario = load_scenario("examples/scenarios/leo_j2.yaml")
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
    assert result.metadata["runner"] == "jax_vectorized_j2_rk4"
    assert result.metadata["force_model"] == "j2"
    assert result.cases[0].trajectory.metadata["runner"] == "jax_vectorized_j2_rk4"
    assert final_jax.position_km == pytest.approx(final_local.position_km)
    assert final_jax.velocity_km_s == pytest.approx(final_local.velocity_km_s)


def test_research_propagate_jax_can_include_final_state_sensitivities() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")

    result = research_propagate_jax(
        scenario,
        cases=1,
        position_sigma_km=0.0,
        velocity_sigma_km_s=0.0,
        seed=7,
        runtime_loader=_fake_autodiff_runtime,
        include_sensitivities=True,
    )

    transition_matrix = result.metadata["final_state_transition_matrix"]
    assert result.metadata["sensitivity_model"] == "jax_jacfwd_final_state_transition"
    assert len(transition_matrix) == 6
    assert all(len(row) == 6 for row in transition_matrix)
    assert np.all(np.isfinite(np.asarray(transition_matrix)))
    assert abs(transition_matrix[0][0]) > 0.0
    assert transition_matrix[0][3] > 0.0


def test_research_od_sensitivity_jax_returns_residual_jacobian_product() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_station_od.yaml")
    trajectory = propagate_local(scenario)
    noisy_measurements = generate_synthetic_measurements(scenario, trajectory)
    truth_measurements = [
        record.model_copy(update={"value": record.metadata["truth"]})
        for record in noisy_measurements
    ]

    result = research_od_sensitivity_jax(
        scenario,
        truth_measurements,
        runtime_loader=_fake_autodiff_runtime,
    )

    jacobian = np.asarray(result.jacobian)
    assert result.scenario_id == scenario.scenario_id
    assert result.backend == "jax"
    assert result.measurement_count == len(truth_measurements)
    assert result.state_dimension == 6
    assert result.metadata["sensitivity_model"] == "jax_jacfwd_od_residuals"
    assert result.metadata["measurement_types"] == ["range", "range_rate"]
    assert max(abs(value) for value in result.residuals) < 1.0e-6
    assert jacobian.shape == (len(truth_measurements), 6)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.matrix_rank(jacobian) >= 6


def test_research_od_sensitivity_jax_supports_inertial_angles() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_station_angles.yaml")
    trajectory = propagate_local(scenario)
    noisy_measurements = generate_synthetic_measurements(scenario, trajectory)
    truth_measurements = [
        record.model_copy(update={"value": record.metadata["truth"]})
        for record in noisy_measurements
    ]

    result = research_od_sensitivity_jax(
        scenario,
        truth_measurements,
        runtime_loader=_fake_autodiff_runtime,
    )

    jacobian = np.asarray(result.jacobian)
    assert result.metadata["measurement_types"] == ["right_ascension", "declination"]
    assert result.metadata["angle_residual_convention"] == "wrapped_degrees"
    assert max(abs(value) for value in result.residuals) < 1.0e-6
    assert jacobian.shape == (len(truth_measurements), 6)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.matrix_rank(jacobian) >= 4


def test_research_od_sensitivity_jax_supports_topocentric_angles() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_station_topocentric.yaml")
    trajectory = propagate_local(scenario)
    noisy_measurements = generate_synthetic_measurements(scenario, trajectory)
    truth_measurements = [
        record.model_copy(update={"value": record.metadata["truth"]})
        for record in noisy_measurements
    ]

    result = research_od_sensitivity_jax(
        scenario,
        truth_measurements,
        runtime_loader=_fake_autodiff_runtime,
    )

    jacobian = np.asarray(result.jacobian)
    assert result.metadata["measurement_types"] == ["azimuth", "elevation"]
    assert result.metadata["angle_residual_convention"] == "wrapped_degrees"
    assert max(abs(value) for value in result.residuals) < 1.0e-6
    assert jacobian.shape == (len(truth_measurements), 6)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.matrix_rank(jacobian) >= 4


def test_research_od_sensitivity_jax_regularizes_topocentric_angle_singularities() -> None:
    try:
        runtime = load_jax_runtime()
    except UnsupportedBackendError as exc:
        pytest.skip(str(exc))
    scenario = load_scenario("examples/scenarios/leo_two_station_topocentric.yaml")
    trajectory = propagate_local(scenario)
    noisy_measurements = generate_synthetic_measurements(scenario, trajectory)
    truth_measurements = [
        record.model_copy(update={"value": record.metadata["truth"]})
        for record in noisy_measurements
    ]

    result = research_od_sensitivity_jax(
        scenario,
        truth_measurements,
        runtime_loader=lambda: runtime,
    )

    jacobian = np.asarray(result.jacobian)
    assert result.metadata["topocentric_angle_sensitivity_regularization"] == (
        "horizontal_norm_floor"
    )
    assert np.all(np.isfinite(jacobian))


def test_research_estimate_jax_runs_gauss_newton_correction() -> None:
    truth_scenario = load_scenario("examples/scenarios/leo_two_station_od.yaml")
    truth_measurements = [
        record.model_copy(update={"value": record.metadata["truth"]})
        for record in generate_synthetic_measurements(
            truth_scenario,
            propagate_local(truth_scenario),
        )
    ]
    truth_state = truth_scenario.initial_state.cartesian
    initial_guess = truth_scenario.initial_state.model_copy(
        update={
            "cartesian": truth_state.model_copy(
                update={
                    "position_km": (
                        truth_state.position_km[0] + 1.0,
                        truth_state.position_km[1] - 0.8,
                        truth_state.position_km[2] + 0.6,
                    ),
                    "velocity_km_s": (
                        truth_state.velocity_km_s[0] + 0.0005,
                        truth_state.velocity_km_s[1] - 0.001,
                        truth_state.velocity_km_s[2] + 0.0008,
                    ),
                }
            )
        }
    )
    scenario = truth_scenario.model_copy(update={"initial_state": initial_guess})

    result = research_estimate_jax(
        scenario,
        truth_measurements,
        runtime_loader=_fake_autodiff_runtime,
        max_iterations=4,
    )

    initial_error = np.linalg.norm(
        np.array(initial_guess.cartesian.position_km)
        - np.array(truth_state.position_km)
    )
    estimated_error = np.linalg.norm(
        np.array(result.estimated_state.cartesian.position_km)
        - np.array(truth_state.position_km)
    )
    assert result.converged is True
    assert result.iterations <= 4
    assert result.rms < 1.0e-3
    assert estimated_error < initial_error
    assert result.metadata["estimator"] == "jax_research_gauss_newton"
    assert result.metadata["sensitivity_model"] == "jax_jacfwd_od_residuals"
    assert result.metadata["measurement_types"] == ["range", "range_rate"]


def test_research_estimate_jax_damps_topocentric_angle_corrections() -> None:
    truth_scenario = load_scenario("examples/scenarios/leo_two_station_topocentric.yaml")
    truth_measurements = [
        record.model_copy(update={"value": record.metadata["truth"]})
        for record in generate_synthetic_measurements(
            truth_scenario,
            propagate_local(truth_scenario),
        )
    ]
    truth_state = truth_scenario.initial_state.cartesian
    initial_guess = truth_scenario.initial_state.model_copy(
        update={
            "cartesian": truth_state.model_copy(
                update={
                    "position_km": (
                        truth_state.position_km[0] + 0.01,
                        truth_state.position_km[1] - 0.005,
                        truth_state.position_km[2] + 0.005,
                    ),
                    "velocity_km_s": truth_state.velocity_km_s,
                }
            )
        }
    )
    scenario = truth_scenario.model_copy(update={"initial_state": initial_guess})

    result = research_estimate_jax(
        scenario,
        truth_measurements,
        runtime_loader=_fake_autodiff_runtime,
        max_iterations=8,
    )

    initial_error = np.linalg.norm(
        np.array(initial_guess.cartesian.position_km)
        - np.array(truth_state.position_km)
    )
    estimated_error = np.linalg.norm(
        np.array(result.estimated_state.cartesian.position_km)
        - np.array(truth_state.position_km)
    )
    assert result.converged is True
    assert result.rms < 1.0e-5
    assert estimated_error < initial_error
    assert result.metadata["measurement_types"] == ["azimuth", "elevation"]
    assert result.metadata["step_strategy"] == "backtracking_gauss_newton"
    assert result.metadata["line_search_backtrack_count"] > 0


def test_research_od_sensitivity_jax_rejects_unsupported_measurement_type() -> None:
    scenario = load_scenario("examples/scenarios/leo_doppler.yaml")
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))

    with pytest.raises(UnsupportedBackendError, match="azimuth, and elevation"):
        research_od_sensitivity_jax(
            scenario,
            measurements,
            runtime_loader=_fake_autodiff_runtime,
        )


def test_research_propagate_jax_maps_orekit_high_fidelity_to_research_j2_baseline() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml").model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.OREKIT_HIGH_FIDELITY)}
    )

    result = research_propagate_jax(
        scenario,
        cases=1,
        position_sigma_km=0.0,
        velocity_sigma_km_s=0.0,
        seed=7,
        runtime_loader=_fake_runtime,
    )

    assert result.metadata["force_model"] == "orekit_high_fidelity"
    assert result.metadata["research_force_models"] == ["J2"]
    assert result.metadata["research_force_model_policy"] == (
        "screening_only_not_operational_ephemeris"
    )


def test_research_propagate_jax_rejects_high_order_gravity_shape() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml").model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                gravity_degree=8,
                gravity_order=8,
            )
        }
    )

    with pytest.raises(UnsupportedBackendError, match="high-order gravity"):
        research_propagate_jax(
            scenario,
            cases=1,
            position_sigma_km=0.0,
            velocity_sigma_km_s=0.0,
            seed=7,
            runtime_loader=_fake_runtime,
        )


def test_research_propagate_jax_runs_research_drag_and_srp_force_flags() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml").model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                atmospheric_drag=True,
                solar_radiation_pressure=True,
            )
        }
    )

    result = research_propagate_jax(
        scenario,
        cases=1,
        position_sigma_km=0.0,
        velocity_sigma_km_s=0.0,
        seed=7,
        runtime_loader=_fake_runtime,
    )

    final_state = result.cases[0].trajectory.samples[-1].state
    assert result.backend == "jax"
    assert result.metadata["runner"] == "jax_vectorized_orekit_high_fidelity_rk4"
    assert result.metadata["research_force_models"] == [
        "J2",
        "exponential_atmospheric_drag",
        "constant_direction_solar_radiation_pressure",
    ]
    assert result.cases[0].trajectory.metadata["research_force_model_policy"] == (
        "screening_only_not_operational_ephemeris"
    )
    assert np.all(np.isfinite(np.asarray(final_state.position_km)))
    assert np.all(np.isfinite(np.asarray(final_state.velocity_km_s)))


def test_research_propagate_jax_runs_research_third_body_force_flag() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml").model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                third_body_gravity=True,
            )
        }
    )

    result = research_propagate_jax(
        scenario,
        cases=1,
        position_sigma_km=0.0,
        velocity_sigma_km_s=0.0,
        seed=7,
        runtime_loader=_fake_runtime,
    )

    final_state = result.cases[0].trajectory.samples[-1].state
    assert result.metadata["research_force_models"] == [
        "J2",
        "analytic_sun_moon_third_body",
    ]
    assert result.metadata["third_body_ephemeris_model"] == (
        "analytic_circular_sun_moon_screening"
    )
    assert result.cases[0].trajectory.metadata["third_body_ephemeris_model"] == (
        "analytic_circular_sun_moon_screening"
    )
    assert np.all(np.isfinite(np.asarray(final_state.position_km)))
    assert np.all(np.isfinite(np.asarray(final_state.velocity_km_s)))


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
