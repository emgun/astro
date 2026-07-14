from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from astro_assurance.errors import MissionAssuranceError
from astro_assurance.models import CorrectionTargetingConfig
from astro_core.models import (
    CartesianState,
    Maneuver,
    OrbitState,
    PropagationConfig,
    Scenario,
    Trajectory,
    Vector3,
)
from astro_dynamics.local import propagate_local
from astro_dynamics.maneuvers import apply_impulsive_maneuver

FloatArray = NDArray[np.float64]


def trajectory_sample_at_elapsed(trajectory: Trajectory, elapsed_s: float) -> CartesianState:
    epoch = trajectory.samples[0].epoch
    for sample in trajectory.samples:
        if abs((sample.epoch - epoch).total_seconds() - elapsed_s) <= 1.0e-9:
            return sample.state
    raise MissionAssuranceError(
        f"trajectory has no sample at elapsed time {elapsed_s}",
        phase="correction",
    )


def design_candidate_correction(
    estimated_scenario: Scenario,
    estimated_trajectory: Trajectory,
    nominal_trajectory: Trajectory,
    config: CorrectionTargetingConfig,
) -> Maneuver:
    burn_state = trajectory_sample_at_elapsed(
        estimated_trajectory, float(config.correction_elapsed_s)
    )
    target_state = trajectory_sample_at_elapsed(
        nominal_trajectory, float(config.verification_elapsed_s)
    )
    burn_epoch = estimated_scenario.initial_state.epoch + timedelta(
        seconds=float(config.correction_elapsed_s)
    )
    coast_duration_s = float(config.verification_elapsed_s - config.correction_elapsed_s)
    coast_scenario = estimated_scenario.model_copy(
        update={
            "scenario_id": f"{estimated_scenario.scenario_id}-targeting-coast",
            "initial_state": estimated_scenario.initial_state.model_copy(
                update={"epoch": burn_epoch, "cartesian": burn_state}
            ),
            "propagation": PropagationConfig(
                duration_s=coast_duration_s,
                step_s=estimated_scenario.propagation.step_s,
            ),
            "maneuvers": [],
        }
    )
    objective = _targeting_objective(
        coast_scenario,
        target_state,
        burn_epoch,
        position_scale_km=float(config.position_scale_km),
        velocity_scale_km_s=float(config.velocity_scale_km_s),
    )
    bound = float(config.maximum_component_delta_v_km_s)
    result = least_squares(
        objective,
        np.zeros(3, dtype=np.float64),
        bounds=(-bound, bound),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=100,
    )
    if not result.success:
        raise MissionAssuranceError(
            f"correction targeting did not converge: {result.message}",
            phase="correction",
        )
    delta_v = np.asarray(result.x, dtype=np.float64)
    if np.any(np.isclose(np.abs(delta_v), bound, rtol=0.0, atol=max(1.0e-12, bound * 1.0e-6))):
        raise MissionAssuranceError(
            "correction targeting reached a component delta-v bound",
            phase="correction",
        )
    total_delta_v = float(np.linalg.norm(delta_v))
    if total_delta_v > config.maximum_total_delta_v_km_s:
        raise MissionAssuranceError(
            "correction targeting exceeds maximum total delta-v",
            phase="correction",
        )
    residual = objective(delta_v)
    delta_v_vector: Vector3 = (
        float(delta_v[0]),
        float(delta_v[1]),
        float(delta_v[2]),
    )
    return Maneuver(
        name="post-launch-recovery-candidate",
        epoch=burn_epoch,
        frame=estimated_scenario.initial_state.frame,
        delta_v_km_s=delta_v_vector,
        duration_s=0.0,
        metadata={
            "disposition": "candidate_for_manual_review",
            "method": "bounded_local_single_impulse_least_squares",
            "optimizer_nfev": int(result.nfev),
            "scaled_residual_norm": float(np.linalg.norm(residual)),
            "specific_impulse_s": float(config.specific_impulse_s),
            "claim_boundary": "deterministic_design_screening_not_flight_command_authority",
        },
    )


def _targeting_objective(
    coast_scenario: Scenario,
    target_state: CartesianState,
    burn_epoch: datetime,
    *,
    position_scale_km: float,
    velocity_scale_km_s: float,
) -> Callable[[FloatArray], FloatArray]:
    def objective(delta_v: FloatArray) -> FloatArray:
        delta_v_vector: Vector3 = (
            float(delta_v[0]),
            float(delta_v[1]),
            float(delta_v[2]),
        )
        maneuver = Maneuver(
            name="targeting-trial",
            epoch=burn_epoch,
            frame=coast_scenario.initial_state.frame,
            delta_v_km_s=delta_v_vector,
        )
        maneuvered_state: OrbitState = apply_impulsive_maneuver(
            coast_scenario.initial_state,
            maneuver,
        )
        trial = coast_scenario.model_copy(update={"initial_state": maneuvered_state})
        final_state = propagate_local(trial).samples[-1].state
        position_error = final_state.position_array() - target_state.position_array()
        velocity_error = final_state.velocity_array() - target_state.velocity_array()
        return np.concatenate(
            [position_error / position_scale_km, velocity_error / velocity_scale_km_s]
        )

    return objective
