from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from astro_backends.tudat.propagation import propagate_tudat
from astro_core.errors import UnsupportedBackendError
from astro_core.models import AstroModel, Scenario, Trajectory
from astro_dynamics.local import propagate_local

TrajectoryRunner = Callable[[Scenario], Trajectory]
FloatArray = NDArray[np.float64]


class TudatReferenceComparison(AstroModel):
    scenario_id: str = Field(min_length=1)
    candidate_backend: str = Field(min_length=1)
    reference_backend: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    position_tolerance_km: float = Field(ge=0.0)
    velocity_tolerance_km_s: float = Field(ge=0.0)
    max_position_delta_km: float = Field(ge=0.0)
    rms_position_delta_km: float = Field(ge=0.0)
    final_position_delta_km: float = Field(ge=0.0)
    max_velocity_delta_km_s: float = Field(ge=0.0)
    rms_velocity_delta_km_s: float = Field(ge=0.0)
    final_velocity_delta_km_s: float = Field(ge=0.0)
    passed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


def _reference_runner_for_backend(reference_backend: str) -> TrajectoryRunner:
    if reference_backend == "local":
        return propagate_local
    if reference_backend == "orekit":
        from astro_backends.orekit import propagate_orekit

        return propagate_orekit
    raise UnsupportedBackendError(
        f"unsupported Tudat comparison reference backend: {reference_backend}"
    )


def _trajectory_arrays(trajectory: Trajectory) -> tuple[FloatArray, FloatArray]:
    positions = np.array(
        [sample.state.position_km for sample in trajectory.samples],
        dtype=np.float64,
    )
    velocities = np.array(
        [sample.state.velocity_km_s for sample in trajectory.samples],
        dtype=np.float64,
    )
    return positions, velocities


def _validate_time_alignment(candidate: Trajectory, reference: Trajectory) -> None:
    if len(candidate.samples) != len(reference.samples):
        raise ValueError("Tudat comparison trajectories must have the same number of samples")
    for sample_index, (candidate_sample, reference_sample) in enumerate(
        zip(candidate.samples, reference.samples, strict=True)
    ):
        if candidate_sample.epoch != reference_sample.epoch:
            raise ValueError(
                "Tudat comparison trajectories must use the same sample epochs; "
                f"mismatch at index {sample_index}"
            )


def _rms(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def compare_tudat_to_reference(
    scenario: Scenario,
    *,
    reference_backend: str = "local",
    position_tolerance_km: float = 1.0e-3,
    velocity_tolerance_km_s: float = 1.0e-6,
    tudat_runner: TrajectoryRunner = propagate_tudat,
    reference_runner: TrajectoryRunner | None = None,
) -> TudatReferenceComparison:
    candidate = tudat_runner(scenario)
    reference = (
        _reference_runner_for_backend(reference_backend)(scenario)
        if reference_runner is None
        else reference_runner(scenario)
    )
    _validate_time_alignment(candidate, reference)

    candidate_positions, candidate_velocities = _trajectory_arrays(candidate)
    reference_positions, reference_velocities = _trajectory_arrays(reference)
    position_deltas = np.linalg.norm(candidate_positions - reference_positions, axis=1)
    velocity_deltas = np.linalg.norm(candidate_velocities - reference_velocities, axis=1)
    max_position_delta_km = float(np.max(position_deltas))
    max_velocity_delta_km_s = float(np.max(velocity_deltas))

    return TudatReferenceComparison(
        scenario_id=scenario.scenario_id,
        candidate_backend=candidate.backend,
        reference_backend=reference_backend,
        sample_count=len(candidate.samples),
        position_tolerance_km=position_tolerance_km,
        velocity_tolerance_km_s=velocity_tolerance_km_s,
        max_position_delta_km=max_position_delta_km,
        rms_position_delta_km=_rms(position_deltas),
        final_position_delta_km=float(position_deltas[-1]),
        max_velocity_delta_km_s=max_velocity_delta_km_s,
        rms_velocity_delta_km_s=_rms(velocity_deltas),
        final_velocity_delta_km_s=float(velocity_deltas[-1]),
        passed=(
            max_position_delta_km <= position_tolerance_km
            and max_velocity_delta_km_s <= velocity_tolerance_km_s
        ),
        metadata={
            "workflow": "tudat_reference_comparison",
            "candidate_force_model": candidate.force_model.model_dump(mode="json"),
            "reference_force_model": reference.force_model.model_dump(mode="json"),
            "tudat_runner": candidate.metadata.get("tudat_runner"),
            "tudat_force_models": candidate.metadata.get("tudat_force_models"),
            "reference_trajectory_backend": reference.backend,
        },
    )
