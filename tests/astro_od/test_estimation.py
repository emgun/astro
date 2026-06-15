from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import astro_od.estimation as estimation
from astro_core.errors import NumericalConvergenceError
from astro_core.io import load_scenario
from astro_core.models import (
    CartesianState,
    Frame,
    GroundStation,
    MeasurementRecord,
    MeasurementType,
    Scenario,
    Trajectory,
    TrajectorySample,
)
from astro_dynamics.local import propagate_local
from astro_od.estimation import estimate_initial_state, residual_vector
from astro_od.measurements import generate_synthetic_measurements


def _observable_scenario() -> Scenario:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    station = GroundStation(
        name="north-eci",
        position_eci_km=(0.0, 6378.1363, 0.0),
        frame=Frame.EME2000,
        elevation_mask_deg=0.0,
    )
    return scenario.model_copy(update={"ground_stations": [*scenario.ground_stations, station]})


def _perturbed_scenario(scenario: Scenario) -> Scenario:
    perturbed_state = scenario.initial_state.model_copy(
        update={
            "cartesian": CartesianState(
                position_km=(7001.0, -0.8, 0.6),
                velocity_km_s=(0.0005, 7.499, 1.0008),
            )
        }
    )
    return scenario.model_copy(update={"initial_state": perturbed_state})


def _single_sample_propagator(scenario: Scenario) -> Trajectory:
    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=[
            TrajectorySample(
                epoch=scenario.initial_state.epoch,
                state=scenario.initial_state.cartesian,
            )
        ],
        force_model=scenario.force_model,
        backend="test",
    )


def test_batch_od_recovers_synthetic_initial_state() -> None:
    truth_scenario = _observable_scenario()
    truth_trajectory = propagate_local(truth_scenario)
    measurements = generate_synthetic_measurements(truth_scenario, truth_trajectory)
    estimate_scenario = _perturbed_scenario(truth_scenario)

    assert {measurement.observer for measurement in measurements} == {"equator-eci", "north-eci"}
    assert {measurement.measurement_type for measurement in measurements} == {
        MeasurementType.RANGE,
        MeasurementType.RANGE_RATE,
    }

    result = estimate_initial_state(estimate_scenario, measurements)

    truth_position = truth_scenario.initial_state.cartesian.position_array()
    estimated_position = result.estimated_state.cartesian.position_array()
    truth_velocity = truth_scenario.initial_state.cartesian.velocity_array()
    estimated_velocity = result.estimated_state.cartesian.velocity_array()

    assert result.converged is True
    assert np.linalg.norm(estimated_position - truth_position) < 0.2
    assert np.linalg.norm(estimated_velocity - truth_velocity) < 2.0e-4
    assert result.rms < 3.0
    assert len(result.residuals) == len(measurements)
    assert len(result.covariance) == 6
    assert all(len(row) == 6 for row in result.covariance)
    assert result.iterations > 0
    assert result.metadata["backend"] == "local_scipy_least_squares"
    assert result.metadata["propagation_backend"] == "local"
    assert isinstance(result.metadata["message"], str)
    assert result.metadata["jacobian_rank"] == 6
    assert len(result.metadata["singular_values"]) == 6
    assert result.metadata["condition_number"] > 0.0


def test_residual_vector_wraps_right_ascension_across_zero_degrees() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    state = scenario.initial_state.cartesian
    state_vector = np.concatenate([state.position_array(), state.velocity_array()])
    measurement = MeasurementRecord(
        measurement_type=MeasurementType.RIGHT_ASCENSION,
        epoch=scenario.initial_state.epoch,
        observer="equator-eci",
        observed_object=scenario.spacecraft.name,
        value=359.9,
        sigma=0.1,
        units="deg",
    )

    residuals = residual_vector(state_vector, scenario, [measurement], _single_sample_propagator)

    assert residuals.tolist() == pytest.approx([1.0])


def test_estimate_initial_state_requires_measurements() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    with pytest.raises(
        NumericalConvergenceError,
        match="At least one measurement is required for estimation",
    ):
        estimate_initial_state(scenario, [])


def test_estimate_initial_state_supports_orekit_propagation_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_scenario = _observable_scenario()
    truth_trajectory = propagate_local(truth_scenario)
    measurements = generate_synthetic_measurements(truth_scenario, truth_trajectory)
    estimate_scenario = _perturbed_scenario(truth_scenario)
    seen_backends: list[str] = []

    def fake_backend_propagation(scenario: Scenario, backend: str) -> object:
        seen_backends.append(backend)
        trajectory = propagate_local(scenario)
        return trajectory.model_copy(update={"backend": backend})

    monkeypatch.setattr(estimation, "propagate_with_backend", fake_backend_propagation)

    result = estimate_initial_state(estimate_scenario, measurements, backend="orekit")

    assert result.converged is True
    assert result.metadata["backend"] == "orekit_scipy_least_squares"
    assert result.metadata["propagation_backend"] == "orekit"
    assert seen_backends
    assert set(seen_backends) == {"orekit"}


def test_estimate_initial_state_rejects_too_few_measurements() -> None:
    truth_scenario = _observable_scenario()
    measurements = generate_synthetic_measurements(truth_scenario, propagate_local(truth_scenario))

    with pytest.raises(NumericalConvergenceError, match="At least 6 measurements"):
        estimate_initial_state(_perturbed_scenario(truth_scenario), measurements[:5])


def test_estimate_initial_state_rejects_rank_deficient_geometry() -> None:
    truth_scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    measurements = generate_synthetic_measurements(truth_scenario, propagate_local(truth_scenario))

    with pytest.raises(NumericalConvergenceError, match="rank deficient"):
        estimate_initial_state(_perturbed_scenario(truth_scenario), measurements)


def test_estimate_initial_state_rejects_mismatched_observed_object() -> None:
    truth_scenario = _observable_scenario()
    measurements = generate_synthetic_measurements(truth_scenario, propagate_local(truth_scenario))
    mismatched_measurements = [
        measurements[0].model_copy(update={"observed_object": "other-sat"}),
        *measurements[1:],
    ]

    with pytest.raises(NumericalConvergenceError, match="observed object"):
        estimate_initial_state(_perturbed_scenario(truth_scenario), mismatched_measurements)


def test_estimate_initial_state_rejects_unknown_observer() -> None:
    truth_scenario = _observable_scenario()
    measurements = generate_synthetic_measurements(truth_scenario, propagate_local(truth_scenario))
    unknown_observer_measurements = [
        measurements[0].model_copy(update={"observer": "missing-station"}),
        *measurements[1:],
    ]

    with pytest.raises(NumericalConvergenceError, match="missing-station"):
        estimate_initial_state(_perturbed_scenario(truth_scenario), unknown_observer_measurements)


def test_estimate_initial_state_rejects_missing_propagated_epoch() -> None:
    truth_scenario = _observable_scenario()
    measurements = generate_synthetic_measurements(truth_scenario, propagate_local(truth_scenario))
    off_grid_epoch_measurements = [
        measurements[0].model_copy(update={"epoch": measurements[0].epoch + timedelta(seconds=30)}),
        *measurements[1:],
    ]

    with pytest.raises(NumericalConvergenceError, match="No propagated sample"):
        estimate_initial_state(_perturbed_scenario(truth_scenario), off_grid_epoch_measurements)


def test_estimate_initial_state_rejects_failed_optimizer_with_full_rank_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_scenario = _observable_scenario()
    measurements = generate_synthetic_measurements(truth_scenario, propagate_local(truth_scenario))
    estimate_scenario = _perturbed_scenario(truth_scenario)

    def failed_least_squares(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            x=estimation._initial_state_vector(estimate_scenario),
            jac=np.eye(6, dtype=np.float64),
            nfev=12,
            success=False,
            message="forced optimizer failure",
        )

    monkeypatch.setattr(estimation, "least_squares", failed_least_squares)

    with pytest.raises(NumericalConvergenceError) as exc_info:
        estimate_initial_state(estimate_scenario, measurements)

    message = str(exc_info.value)
    assert "forced optimizer failure" in message
    assert "nfev=12" in message
    assert "rms=" in message
    assert "jacobian_rank=6" in message
    assert "condition_number=" in message
