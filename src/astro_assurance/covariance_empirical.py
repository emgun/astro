from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from astro_assurance.covariance_validation_models import (
    CovarianceUnitsPolicy,
    EmpiricalCovarianceArtifact,
    EmpiricalCovarianceCampaignProvenance,
    EmpiricalCovarianceRawSample,
    Matrix6,
    Vector6,
)
from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario, load_trajectory
from astro_core.models import CartesianState, Scenario, Trajectory
from astro_dynamics.backends import propagate_with_backend

FloatArray = NDArray[np.float64]
TruthPropagator = Callable[[Scenario, str], Trajectory]
BackendName = Literal["orekit", "tudat", "local"]


def _state_vector(trajectory: Trajectory, epoch_index: int = -1) -> FloatArray:
    state = trajectory.samples[epoch_index].state
    return np.asarray([*state.position_km, *state.velocity_km_s], dtype=np.float64)


def _vector6(values: FloatArray) -> Vector6:
    return cast(Vector6, tuple(float(value) for value in values))


def _tuple3(values: FloatArray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _matrix6(values: FloatArray) -> Matrix6:
    return cast(
        Matrix6,
        tuple(tuple(float(value) for value in row) for row in values),
    )


def _truth_scenario(scenario: Scenario, perturbation: FloatArray) -> Scenario:
    initial = scenario.initial_state.cartesian
    state = np.asarray(
        [*initial.position_km, *initial.velocity_km_s], dtype=np.float64
    ) + perturbation
    return scenario.model_copy(
        update={
            "initial_state": scenario.initial_state.model_copy(
                update={
                    "cartesian": CartesianState(
                        position_km=_tuple3(state[:3]),
                        velocity_km_s=_tuple3(state[3:]),
                    )
                }
            ),
            "initial_covariance": None,
            "covariance_state_transition_model": "finite_difference",
        }
    )


def _default_truth_propagator(scenario: Scenario, backend: str) -> Trajectory:
    return propagate_with_backend(scenario, backend)


def run_empirical_covariance_campaign(
    scenario_path: Path | str,
    predictor_trajectory_path: Path | str,
    *,
    truth_backend: str,
    samples: int,
    seed: int,
    truth_propagator: TruthPropagator = _default_truth_propagator,
) -> EmpiricalCovarianceArtifact:
    if samples < 2:
        raise InvalidScenarioError("empirical covariance campaign requires at least 2 samples")
    if seed < 0:
        raise InvalidScenarioError("empirical covariance campaign seed must be nonnegative")
    if truth_backend not in {"orekit", "tudat", "local"}:
        raise InvalidScenarioError(f"unsupported empirical truth backend: {truth_backend}")
    typed_truth_backend = cast(BackendName, truth_backend)
    scenario_source = Path(scenario_path)
    predictor_source = Path(predictor_trajectory_path)
    scenario_bytes = scenario_source.read_bytes()
    predictor_bytes = predictor_source.read_bytes()
    scenario = load_scenario(scenario_source)
    predictor = load_trajectory(predictor_source)
    if scenario.initial_covariance is None:
        raise InvalidScenarioError("empirical campaign scenario requires initial_covariance")
    if scenario.covariance_process_noise_acceleration_km_s2 != 0.0:
        raise InvalidScenarioError(
            "empirical initial-state campaign requires zero covariance process noise"
        )
    if predictor.scenario_id != scenario.scenario_id:
        raise InvalidScenarioError("empirical predictor scenario id does not match scenario")
    if predictor.backend == truth_backend:
        raise InvalidScenarioError("empirical predictor and truth backends must be independent")
    if not predictor.covariance_history:
        raise InvalidScenarioError("empirical predictor contains no covariance history")
    predictor_implementation = predictor.metadata.get("covariance_implementation")
    if not isinstance(predictor_implementation, str) or not predictor_implementation:
        raise InvalidScenarioError("empirical predictor lacks covariance implementation provenance")
    expected_units = CovarianceUnitsPolicy().model_dump(mode="json")
    if predictor.metadata.get("covariance_units_policy") != expected_units:
        raise InvalidScenarioError("empirical predictor lacks exact covariance units provenance")

    initial_covariance = np.asarray(scenario.initial_covariance, dtype=np.float64)
    try:
        covariance_factor = np.linalg.cholesky(initial_covariance)
    except np.linalg.LinAlgError as exc:
        raise InvalidScenarioError(
            "empirical campaign initial covariance must be positive definite"
        ) from exc
    predicted_sample = predictor.covariance_history[-1]
    predicted_covariance = np.asarray(predicted_sample.covariance, dtype=np.float64)
    if predictor.samples[-1].epoch != predicted_sample.epoch:
        raise InvalidScenarioError(
            "empirical predictor terminal state and covariance epochs must match"
        )

    nominal_scenario = _truth_scenario(scenario, np.zeros(6, dtype=np.float64))
    nominal_trajectory = truth_propagator(nominal_scenario, truth_backend)
    if nominal_trajectory.backend != truth_backend:
        raise InvalidScenarioError("empirical truth propagator returned the wrong backend")
    if nominal_trajectory.samples[-1].epoch != predicted_sample.epoch:
        raise InvalidScenarioError("empirical truth and predictor terminal epochs must match")
    nominal_state = _state_vector(nominal_trajectory)
    rng = np.random.default_rng(seed)
    raw_samples: list[EmpiricalCovarianceRawSample] = []
    for sample_index in range(samples):
        perturbation = cast(FloatArray, covariance_factor @ rng.standard_normal(6))
        realized_trajectory = truth_propagator(
            _truth_scenario(scenario, perturbation), truth_backend
        )
        if realized_trajectory.backend != truth_backend:
            raise InvalidScenarioError(
                f"empirical truth sample {sample_index} returned the wrong backend"
            )
        if realized_trajectory.samples[-1].epoch != predicted_sample.epoch:
            raise InvalidScenarioError(
                f"empirical truth sample {sample_index} terminal epoch does not match"
            )
        realized_state = _state_vector(realized_trajectory)
        raw_samples.append(
            EmpiricalCovarianceRawSample(
                sample_id=f"truth-realization-{sample_index:04d}",
                epoch=predicted_sample.epoch,
                state_error=_vector6(realized_state - nominal_state),
                predicted_covariance=_matrix6(predicted_covariance),
                independent_truth=True,
                initial_state_perturbation=_vector6(perturbation),
                nominal_truth_state=_vector6(nominal_state),
                realized_truth_state=_vector6(realized_state),
            )
        )

    return EmpiricalCovarianceArtifact(
        artifact_id=f"{scenario.scenario_id}-empirical-covariance",
        units_policy=CovarianceUnitsPolicy(),
        population_definition=(
            "Independent zero-mean Gaussian initial-state perturbations drawn from the "
            "declared P0 and propagated nonlinearly to the terminal epoch with zero "
            "realized process noise."
        ),
        independent_realizations=True,
        independence_basis=(
            "Each realization uses a distinct deterministic PCG64 Gaussian draw; the "
            "truth backend differs from the covariance predictor backend."
        ),
        campaign_provenance=EmpiricalCovarianceCampaignProvenance(
            scenario_id=scenario.scenario_id,
            scenario_sha256=sha256(scenario_bytes).hexdigest(),
            predictor_trajectory_sha256=sha256(predictor_bytes).hexdigest(),
            predictor_backend=cast(BackendName, predictor.backend),
            predictor_implementation=predictor_implementation,
            truth_backend=typed_truth_backend,
            seed=seed,
            evaluation_epoch=predicted_sample.epoch,
            initial_covariance=_matrix6(initial_covariance),
            sample_count=samples,
            force_model=scenario.force_model,
        ),
        samples=tuple(raw_samples),
    )


def write_empirical_covariance_artifact(
    path: Path | str, artifact: EmpiricalCovarianceArtifact
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (artifact.model_dump_json(indent=2) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
