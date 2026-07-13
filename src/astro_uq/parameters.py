from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any

from astro_core.models import AstroModel
from astro_uq.models import (
    AppliedBinding,
    ParameterRealization,
    ScenarioRealization,
    UncertaintyModel,
)

ParameterValue = float | str
ParameterGetter = Callable[[AstroModel], ParameterValue]
ParameterUpdater = Callable[[AstroModel, ParameterValue], AstroModel]


class ParameterBindingError(ValueError):
    """Raised when a realization cannot be safely bound to a workflow scenario."""


@dataclass(frozen=True)
class ParameterBinding:
    target: str
    workflow: str
    unit: str
    value_type: type[float] | type[str]
    getter: ParameterGetter
    updater: ParameterUpdater
    lower: float | None = None
    upper: float | None = None

    def validate_value(self, value: ParameterValue) -> None:
        if isinstance(value, bool):
            raise ParameterBindingError(f"{self.target} does not accept boolean values")
        if self.value_type is float:
            if not isinstance(value, int | float):
                raise ParameterBindingError(f"{self.target} requires a numeric value")
            numeric = float(value)
            if not isfinite(numeric):
                raise ParameterBindingError(f"{self.target} requires a finite value")
            if self.lower is not None and numeric < self.lower:
                raise ParameterBindingError(f"{self.target} must be >= {self.lower}")
            if self.upper is not None and numeric > self.upper:
                raise ParameterBindingError(f"{self.target} must be <= {self.upper}")
        elif not isinstance(value, str) or not value:
            raise ParameterBindingError(f"{self.target} requires a non-empty string value")


class ParameterRegistry:
    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], ParameterBinding] = {}

    def register(self, binding: ParameterBinding) -> None:
        key = (binding.workflow, binding.target)
        if key in self._bindings:
            raise ParameterBindingError(
                f"parameter binding already registered for {binding.workflow}:{binding.target}"
            )
        self._bindings[key] = binding

    def resolve(self, workflow: str, target: str) -> ParameterBinding:
        try:
            return self._bindings[(workflow, target)]
        except KeyError as exc:
            raise ParameterBindingError(
                f"unregistered parameter target for {workflow}: {target}"
            ) from exc

    def apply(
        self,
        *,
        workflow: str,
        scenario: AstroModel,
        uncertainty: UncertaintyModel,
        realization: ParameterRealization,
    ) -> tuple[AstroModel, ScenarioRealization]:
        base_digest = model_digest(scenario)
        parameter_by_id = {
            parameter.parameter_id: parameter for parameter in uncertainty.parameters
        }
        unknown = set(realization.physical_values) - set(parameter_by_id)
        if unknown:
            raise ParameterBindingError(
                f"realization contains unknown parameters: {', '.join(sorted(unknown))}"
            )
        missing = set(parameter_by_id) - set(realization.physical_values)
        if missing:
            raise ParameterBindingError(
                f"realization is missing parameters: {', '.join(sorted(missing))}"
            )

        targets = [parameter.target for parameter in uncertainty.parameters]
        targets.extend(realization.model_variants)
        if len(set(targets)) != len(targets):
            raise ParameterBindingError("a realization may write each target only once")

        updated: AstroModel = scenario
        applied: list[AppliedBinding] = []
        for parameter in uncertainty.parameters:
            value = realization.physical_values[parameter.parameter_id]
            binding = self.resolve(workflow, parameter.target)
            if binding.unit != parameter.unit:
                raise ParameterBindingError(
                    f"unit mismatch for {parameter.parameter_id}: "
                    f"expected {binding.unit}, received {parameter.unit}"
                )
            binding.validate_value(value)
            updated = binding.updater(updated, value)
            applied.append(
                AppliedBinding(
                    parameter_id=parameter.parameter_id,
                    target=parameter.target,
                    unit=parameter.unit,
                    value=value,
                )
            )

        for target, value in realization.model_variants.items():
            binding = self.resolve(workflow, target)
            binding.validate_value(value)
            updated = binding.updater(updated, value)
            applied.append(
                AppliedBinding(
                    parameter_id=target,
                    target=target,
                    unit=binding.unit,
                    value=value,
                )
            )

        try:
            validated = type(scenario).model_validate(updated.model_dump(mode="python"))
        except Exception as exc:
            raise ParameterBindingError(f"resolved scenario is invalid: {exc}") from exc
        resolved_digest = model_digest(validated)
        return validated, ScenarioRealization(
            sample_id=realization.sample_id,
            base_scenario_digest=base_digest,
            resolved_scenario_digest=resolved_digest,
            bindings=tuple(applied),
        )

    def targets(self, workflow: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                target
                for registered_workflow, target in self._bindings
                if registered_workflow == workflow
            )
        )


def model_digest(model: AstroModel) -> str:
    payload = model.model_dump_json(exclude_none=False, by_alias=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def replace_model_field(model: AstroModel, field: str, value: Any) -> AstroModel:
    payload = model.model_dump(mode="python")
    payload[field] = value
    return type(model).model_validate(payload)
