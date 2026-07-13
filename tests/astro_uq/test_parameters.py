from __future__ import annotations

import pytest

from astro_core.io import load_scenario
from astro_core.models import AstroModel, Scenario
from astro_uq.models import (
    DistributionKind,
    DistributionSpec,
    ParameterRealization,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
)
from astro_uq.parameters import ParameterBinding, ParameterBindingError, ParameterRegistry


def _replace_mass(model: AstroModel, value: float | str) -> AstroModel:
    scenario = Scenario.model_validate(model)
    spacecraft = scenario.spacecraft.model_copy(update={"mass_kg": value})
    return scenario.model_copy(update={"spacecraft": spacecraft})


def _binding(*, lower: float | None = 0.001) -> ParameterBinding:
    return ParameterBinding(
        target="orbit.spacecraft.mass_kg",
        workflow="orbit",
        unit="kg",
        value_type=float,
        lower=lower,
        getter=lambda model: float(Scenario.model_validate(model).spacecraft.mass_kg),
        updater=_replace_mass,
    )


def _registry(*, lower: float | None = 0.001) -> ParameterRegistry:
    registry = ParameterRegistry()
    registry.register(_binding(lower=lower))
    return registry


def _uncertainty(unit: str = "kg") -> UncertaintyModel:
    return UncertaintyModel(
        parameters=(
            UncertainParameter(
                parameter_id="mass",
                target="orbit.spacecraft.mass_kg",
                unit=unit,
                uncertainty_kind=UncertaintyKind.EPISTEMIC,
                distribution=DistributionSpec(
                    kind=DistributionKind.UNIFORM,
                    low=90.0,
                    high=110.0,
                ),
            ),
        ),
    )


def _realization(value: float = 105.0) -> ParameterRealization:
    return ParameterRealization(
        sample_id="sample-000000",
        sample_index=0,
        normalized_values={"mass": 0.75},
        physical_values={"mass": value},
    )


def test_registry_applies_value_without_mutating_base_scenario() -> None:
    base = load_scenario("examples/scenarios/leo_two_body.yaml")
    original_mass = base.spacecraft.mass_kg

    resolved, evidence = _registry().apply(
        workflow="orbit",
        scenario=base,
        uncertainty=_uncertainty(),
        realization=_realization(),
    )

    assert isinstance(resolved, Scenario)
    assert resolved.spacecraft.mass_kg == 105.0
    assert base.spacecraft.mass_kg == original_mass
    assert evidence.base_scenario_digest != evidence.resolved_scenario_digest
    assert evidence.bindings[0].target == "orbit.spacecraft.mass_kg"


def test_registry_rejects_unregistered_target() -> None:
    uncertainty = _uncertainty().model_copy(
        update={
            "parameters": (
                _uncertainty().parameters[0].model_copy(
                    update={"target": "orbit.spacecraft.unknown"}
                ),
            )
        }
    )

    with pytest.raises(ParameterBindingError, match="unregistered"):
        _registry().apply(
            workflow="orbit",
            scenario=load_scenario("examples/scenarios/leo_two_body.yaml"),
            uncertainty=uncertainty,
            realization=_realization(),
        )


def test_registry_rejects_unit_mismatch() -> None:
    with pytest.raises(ParameterBindingError, match="unit mismatch"):
        _registry().apply(
            workflow="orbit",
            scenario=load_scenario("examples/scenarios/leo_two_body.yaml"),
            uncertainty=_uncertainty(unit="g"),
            realization=_realization(),
        )


def test_registry_revalidates_nested_updates() -> None:
    with pytest.raises(ParameterBindingError, match="greater than 0"):
        _registry(lower=None).apply(
            workflow="orbit",
            scenario=load_scenario("examples/scenarios/leo_two_body.yaml"),
            uncertainty=_uncertainty(),
            realization=_realization(value=-1.0),
        )


def test_registry_rejects_duplicate_registration() -> None:
    registry = _registry()

    with pytest.raises(ParameterBindingError, match="already registered"):
        registry.register(_binding())
