from __future__ import annotations

import numpy as np

from astro_core.models import Scenario, Trajectory


def assert_covariance_history_invariants(
    trajectory: Trajectory,
    scenario: Scenario,
    *,
    expected_covariance_model: str,
    expected_transition_model: str,
) -> None:
    assert len(trajectory.covariance_history) == scenario.propagation.sample_count
    assert trajectory.metadata["covariance_model"] == expected_covariance_model
    assert trajectory.metadata["covariance_state_transition_storage"] == (
        "per_sample_and_accumulated"
    )
    assert trajectory.covariance_history[0].metadata["state_transition_model"] == "identity"
    assert trajectory.covariance_history[1].metadata["state_transition_model"] == (
        expected_transition_model
    )

    for sample in trajectory.covariance_history:
        covariance = np.asarray(sample.covariance)
        assert covariance.shape == (6, 6)
        assert np.all(np.isfinite(covariance))
        np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-9)
        assert np.all(np.diag(covariance) >= -1.0e-12)

        if sample.state_transition_matrix is not None:
            transition = np.asarray(sample.state_transition_matrix)
            assert transition.shape == (6, 6)
            assert np.all(np.isfinite(transition))

        if sample.accumulated_state_transition_matrix is not None:
            accumulated_transition = np.asarray(sample.accumulated_state_transition_matrix)
            assert accumulated_transition.shape == (6, 6)
            assert np.all(np.isfinite(accumulated_transition))

        if sample.process_noise_covariance is not None:
            process_noise = np.asarray(sample.process_noise_covariance)
            assert process_noise.shape == (6, 6)
            assert np.all(np.isfinite(process_noise))
            np.testing.assert_allclose(process_noise, process_noise.T, rtol=0.0, atol=1.0e-18)

    final_transition = np.asarray(trajectory.covariance_history[-1].state_transition_matrix)
    assert final_transition.shape == (6, 6)
    assert not np.allclose(final_transition, np.eye(6))
