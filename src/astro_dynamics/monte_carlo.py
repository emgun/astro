from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
from pydantic import Field, FiniteFloat, field_validator

from astro_core.models import (
    AstroModel,
    CartesianState,
    OrbitState,
    Scenario,
    Trajectory,
    Vector3,
    _integer_input_must_be_int,
    _numeric_scalar_input_must_be_number,
    _numeric_sequence_input_must_be_numbers,
)
from astro_dynamics.backends import propagate_with_backend


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
    monte_carlo_cases: list[MonteCarloCase] = []

    for case_index in range(cases):
        position_delta = rng.normal(0.0, position_sigma_km, size=3)
        velocity_delta = rng.normal(0.0, velocity_sigma_km_s, size=3)
        cartesian = CartesianState(
            position_km=_tuple3(base_position + position_delta),
            velocity_km_s=_tuple3(base_velocity + velocity_delta),
        )
        initial_state = scenario.initial_state.model_copy(update={"cartesian": cartesian})
        perturbed_scenario = scenario.model_copy(update={"initial_state": initial_state})
        trajectory = propagate_with_backend(perturbed_scenario, backend)
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
    )
