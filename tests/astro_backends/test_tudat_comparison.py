import pytest

from astro_backends.tudat.comparison import compare_tudat_to_reference
from astro_core.io import load_scenario
from astro_core.models import CartesianState, Scenario, Trajectory, TrajectorySample
from astro_dynamics.local import propagate_local


def _short_scenario() -> Scenario:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    return scenario.model_copy(
        update={"propagation": scenario.propagation.model_copy(update={"duration_s": 120.0})}
    )


def _offset_trajectory(
    scenario: Scenario,
    *,
    position_offset_km: float,
    velocity_offset_km_s: float,
) -> Trajectory:
    trajectory = propagate_local(scenario)
    return trajectory.model_copy(
        update={
            "backend": "tudat",
            "metadata": {
                **trajectory.metadata,
                "tudat_runner": "native_two_body",
                "tudat_force_models": ["Earth point-mass gravity"],
            },
            "samples": [
                TrajectorySample(
                    epoch=sample.epoch,
                    state=CartesianState(
                        position_km=(
                            sample.state.position_km[0] + position_offset_km,
                            sample.state.position_km[1],
                            sample.state.position_km[2],
                        ),
                        velocity_km_s=(
                            sample.state.velocity_km_s[0] + velocity_offset_km_s,
                            sample.state.velocity_km_s[1],
                            sample.state.velocity_km_s[2],
                        ),
                    ),
                )
                for sample in trajectory.samples
            ],
        }
    )


def test_compare_tudat_to_reference_reports_calibrated_deltas() -> None:
    scenario = _short_scenario()

    result = compare_tudat_to_reference(
        scenario,
        reference_backend="local",
        position_tolerance_km=0.2,
        velocity_tolerance_km_s=0.002,
        tudat_runner=lambda candidate: _offset_trajectory(
            candidate,
            position_offset_km=0.1,
            velocity_offset_km_s=0.001,
        ),
        reference_runner=propagate_local,
    )

    assert result.scenario_id == scenario.scenario_id
    assert result.candidate_backend == "tudat"
    assert result.reference_backend == "local"
    assert result.sample_count == scenario.propagation.sample_count
    assert result.max_position_delta_km == pytest.approx(0.1)
    assert result.rms_position_delta_km == pytest.approx(0.1)
    assert result.final_position_delta_km == pytest.approx(0.1)
    assert result.max_velocity_delta_km_s == pytest.approx(0.001)
    assert result.rms_velocity_delta_km_s == pytest.approx(0.001)
    assert result.final_velocity_delta_km_s == pytest.approx(0.001)
    assert result.passed is True
    assert result.metadata["tudat_runner"] == "native_two_body"
    assert result.metadata["tudat_force_models"] == ["Earth point-mass gravity"]


def test_compare_tudat_to_reference_fails_when_tolerance_is_exceeded() -> None:
    scenario = _short_scenario()

    result = compare_tudat_to_reference(
        scenario,
        reference_backend="local",
        position_tolerance_km=0.05,
        velocity_tolerance_km_s=0.0005,
        tudat_runner=lambda candidate: _offset_trajectory(
            candidate,
            position_offset_km=0.1,
            velocity_offset_km_s=0.001,
        ),
        reference_runner=propagate_local,
    )

    assert result.passed is False
    assert result.max_position_delta_km > result.position_tolerance_km
    assert result.max_velocity_delta_km_s > result.velocity_tolerance_km_s


def test_compare_tudat_to_reference_requires_time_aligned_samples() -> None:
    scenario = _short_scenario()

    def short_reference(candidate: Scenario) -> Trajectory:
        trajectory = propagate_local(candidate)
        return trajectory.model_copy(update={"samples": trajectory.samples[:-1]})

    with pytest.raises(ValueError, match="same number of samples"):
        compare_tudat_to_reference(
            scenario,
            reference_backend="local",
            tudat_runner=lambda candidate: _offset_trajectory(
                candidate,
                position_offset_km=0.0,
                velocity_offset_km_s=0.0,
            ),
            reference_runner=short_reference,
        )
