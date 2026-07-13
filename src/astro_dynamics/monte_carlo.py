from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
from pydantic import Field, FiniteFloat, field_validator

from astro_core.models import (
    AstroModel,
    OrbitState,
    Scenario,
    Trajectory,
    Vector3,
    _integer_input_must_be_int,
    _numeric_scalar_input_must_be_number,
    _numeric_sequence_input_must_be_numbers,
)
from astro_dynamics.backends import propagate_with_backend
from astro_uq.adapters.orbit import orbit_parameter_registry
from astro_uq.evaluators import AuthoritativeCallableEvaluator, evaluate_authoritatively
from astro_uq.models import (
    DistributionKind,
    DistributionSpec,
    OutcomeStatus,
    ParameterRealization,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
)

_UQ_CAMPAIGN_KERNEL = "astro_uq.evaluators.evaluate_authoritatively"
_UQ_CAMPAIGN_KERNEL_VERSION = "1.0"
_CARTESIAN_PARAMETERS = (
    ("position_x", "orbit.initial_state.cartesian.position_x_km", "km", 0),
    ("position_y", "orbit.initial_state.cartesian.position_y_km", "km", 1),
    ("position_z", "orbit.initial_state.cartesian.position_z_km", "km", 2),
    ("velocity_x", "orbit.initial_state.cartesian.velocity_x_km_s", "km/s", 0),
    ("velocity_y", "orbit.initial_state.cartesian.velocity_y_km_s", "km/s", 1),
    ("velocity_z", "orbit.initial_state.cartesian.velocity_z_km_s", "km/s", 2),
)


class MonteCarloCase(AstroModel):
    case_index: int = Field(ge=0)
    position_delta_km: Vector3
    velocity_delta_km_s: Vector3
    initial_state: OrbitState
    trajectory: Trajectory

    @field_validator("case_index", mode="before")
    @classmethod
    def case_index_must_be_integer_input(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "Monte Carlo case index")

    @field_validator("position_delta_km", "velocity_delta_km_s", mode="before")
    @classmethod
    def vector_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_sequence_input_must_be_numbers(value, "Monte Carlo perturbation")

    @field_validator("position_delta_km", "velocity_delta_km_s")
    @classmethod
    def vectors_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Monte Carlo perturbation values must be finite")
        return value


class MonteCarloResult(AstroModel):
    scenario_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    seed: int
    position_sigma_km: FiniteFloat = Field(ge=0.0)
    velocity_sigma_km_s: FiniteFloat = Field(ge=0.0)
    cases: list[MonteCarloCase] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("seed", mode="before")
    @classmethod
    def seed_must_be_integer_input(cls, value: Any) -> Any:
        return _integer_input_must_be_int(value, "Monte Carlo seed")

    @field_validator("position_sigma_km", "velocity_sigma_km_s", mode="before")
    @classmethod
    def scalar_inputs_must_be_numeric(cls, value: Any) -> Any:
        return _numeric_scalar_input_must_be_number(value, "Monte Carlo sigma")


def _tuple3(array: np.ndarray[tuple[int], np.dtype[np.float64]]) -> Vector3:
    return (float(array[0]), float(array[1]), float(array[2]))


def _validate_inputs(
    *,
    cases: int,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
) -> None:
    if isinstance(cases, bool) or cases <= 0:
        raise ValueError("cases must be positive")
    if not isfinite(position_sigma_km) or not isfinite(velocity_sigma_km_s):
        raise ValueError("sigmas must be finite")
    if position_sigma_km < 0.0 or velocity_sigma_km_s < 0.0:
        raise ValueError("sigmas must be nonnegative")


def _distribution(mean: float, sigma: float) -> DistributionSpec:
    if sigma == 0.0:
        return DistributionSpec(kind=DistributionKind.CONSTANT, value=mean)
    return DistributionSpec(kind=DistributionKind.NORMAL, mean=mean, sigma=sigma)


def _initial_state_uncertainty(
    scenario: Scenario,
    *,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
) -> UncertaintyModel:
    position = scenario.initial_state.cartesian.position_km
    velocity = scenario.initial_state.cartesian.velocity_km_s
    parameters = []
    for parameter_id, target, unit, index in _CARTESIAN_PARAMETERS:
        is_position = parameter_id.startswith("position_")
        mean = float(position[index] if is_position else velocity[index])
        sigma = position_sigma_km if is_position else velocity_sigma_km_s
        parameters.append(
            UncertainParameter(
                parameter_id=parameter_id,
                target=target,
                unit=unit,
                uncertainty_kind=UncertaintyKind.ALEATORY,
                distribution=_distribution(mean, sigma),
            )
        )
    return UncertaintyModel(parameters=tuple(parameters))


def _realization(
    *,
    case_index: int,
    base_position: np.ndarray[tuple[int], np.dtype[np.float64]],
    base_velocity: np.ndarray[tuple[int], np.dtype[np.float64]],
    position_delta: np.ndarray[tuple[int], np.dtype[np.float64]],
    velocity_delta: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> ParameterRealization:
    position = base_position + position_delta
    velocity = base_velocity + velocity_delta
    values: dict[str, float | str] = {
        parameter_id: float(
            position[index] if parameter_id.startswith("position_") else velocity[index]
        )
        for parameter_id, _target, _unit, index in _CARTESIAN_PARAMETERS
    }
    return ParameterRealization(
        sample_id=f"legacy-monte-carlo-{case_index}",
        sample_index=case_index,
        physical_values=values,
    )


def run_initial_state_monte_carlo(
    scenario: Scenario,
    *,
    cases: int,
    position_sigma_km: float,
    velocity_sigma_km_s: float,
    seed: int,
    backend: str = "local",
) -> MonteCarloResult:
    _validate_inputs(
        cases=cases,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
    )

    rng = np.random.default_rng(seed)
    base_position = scenario.initial_state.cartesian.position_array()
    base_velocity = scenario.initial_state.cartesian.velocity_array()
    uncertainty = _initial_state_uncertainty(
        scenario,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
    )
    parameters = orbit_parameter_registry()
    evaluator = AuthoritativeCallableEvaluator[Scenario, Trajectory](
        evaluator_id=f"orbit-propagation-{backend}",
        evaluate_callable=lambda resolved: propagate_with_backend(resolved, backend),
        serialize_callable=lambda _trajectory: (),
    )
    monte_carlo_cases: list[MonteCarloCase] = []

    for case_index in range(cases):
        position_delta = rng.normal(0.0, position_sigma_km, size=3)
        velocity_delta = rng.normal(0.0, velocity_sigma_km_s, size=3)
        realization = _realization(
            case_index=case_index,
            base_position=base_position,
            base_velocity=base_velocity,
            position_delta=position_delta,
            velocity_delta=velocity_delta,
        )
        resolved, scenario_evidence = parameters.apply(
            workflow="orbit",
            scenario=scenario,
            uncertainty=uncertainty,
            realization=realization,
        )
        perturbed_scenario = Scenario.model_validate(resolved)
        outcome, trajectory = evaluate_authoritatively(
            evaluator,
            perturbed_scenario,
            scenario_evidence,
        )
        if outcome.status is not OutcomeStatus.SUCCESS or trajectory is None:
            detail = outcome.error_message or outcome.status.value
            raise RuntimeError(f"Monte Carlo case {case_index} evaluation failed: {detail}")
        initial_state = perturbed_scenario.initial_state
        monte_carlo_cases.append(
            MonteCarloCase(
                case_index=case_index,
                position_delta_km=_tuple3(position_delta),
                velocity_delta_km_s=_tuple3(velocity_delta),
                initial_state=initial_state,
                trajectory=trajectory,
            )
        )

    return MonteCarloResult(
        scenario_id=scenario.scenario_id,
        backend=backend,
        seed=seed,
        position_sigma_km=position_sigma_km,
        velocity_sigma_km_s=velocity_sigma_km_s,
        cases=monte_carlo_cases,
        metadata={
            "uq_campaign_kernel": _UQ_CAMPAIGN_KERNEL,
            "uq_campaign_kernel_version": _UQ_CAMPAIGN_KERNEL_VERSION,
        },
    )
