from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any

from astro_core.models import AstroModel
from astro_uq.models import (
    MetricSpec,
    MetricValue,
    MetricValueKind,
    RequirementOperator,
    RequirementOutcome,
    RequirementSpec,
)


class MetricError(ValueError):
    """Raised when a metric cannot be extracted or evaluated safely."""


Extractor = Callable[[AstroModel], MetricValue]


@dataclass(frozen=True)
class MetricExtractor:
    extractor_id: str
    workflow: str
    value_kind: MetricValueKind
    unit: str | None
    extract: Extractor


class MetricRegistry:
    def __init__(self) -> None:
        self._extractors: dict[tuple[str, str], MetricExtractor] = {}

    def register(self, extractor: MetricExtractor) -> None:
        key = (extractor.workflow, extractor.extractor_id)
        if key in self._extractors:
            raise MetricError(
                f"metric extractor already registered for "
                f"{extractor.workflow}:{extractor.extractor_id}"
            )
        self._extractors[key] = extractor

    def resolve(self, workflow: str, extractor_id: str) -> MetricExtractor:
        try:
            return self._extractors[(workflow, extractor_id)]
        except KeyError as exc:
            raise MetricError(
                f"unregistered metric extractor for {workflow}: {extractor_id}"
            ) from exc

    def extract(
        self,
        *,
        workflow: str,
        result: AstroModel,
        specifications: tuple[MetricSpec, ...],
    ) -> dict[str, MetricValue]:
        values: dict[str, MetricValue] = {}
        for specification in specifications:
            try:
                extractor = self.resolve(workflow, specification.extractor)
            except MetricError:
                raise
            if extractor.value_kind is not specification.value_kind:
                raise MetricError(f"metric kind mismatch for {specification.metric_id}")
            if extractor.unit != specification.unit:
                raise MetricError(f"metric unit mismatch for {specification.metric_id}")
            value = extractor.extract(result)
            _validate_metric_value(value, specification)
            values[specification.metric_id] = value
        return values


def evaluate_requirements(
    values: dict[str, MetricValue],
    requirements: tuple[RequirementSpec, ...],
) -> tuple[RequirementOutcome, ...]:
    outcomes: list[RequirementOutcome] = []
    for requirement in requirements:
        metric = values.get(requirement.metric_id)
        if metric is None:
            outcomes.append(
                RequirementOutcome(
                    requirement_id=requirement.requirement_id,
                    passed=None,
                    reason="metric_missing_or_not_applicable",
                )
            )
            continue
        passed, margin = _evaluate_requirement(metric, requirement)
        outcomes.append(
            RequirementOutcome(
                requirement_id=requirement.requirement_id,
                passed=passed,
                margin=margin,
            )
        )
    return tuple(outcomes)


def _validate_metric_value(value: MetricValue, specification: MetricSpec) -> None:
    if value is None:
        return
    if specification.value_kind is MetricValueKind.BOOLEAN:
        if not isinstance(value, bool):
            raise MetricError(f"{specification.metric_id} must produce a boolean")
        return
    if specification.value_kind is MetricValueKind.CATEGORY:
        if not isinstance(value, str) or not value:
            raise MetricError(f"{specification.metric_id} must produce a non-empty category")
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(float(value)):
        raise MetricError(f"{specification.metric_id} must produce a finite numeric value")


def _evaluate_requirement(metric: MetricValue, requirement: RequirementSpec) -> tuple[bool, float]:
    operator = requirement.operator
    if operator is RequirementOperator.IS_TRUE:
        if not isinstance(metric, bool):
            raise MetricError(f"{requirement.requirement_id} requires a boolean metric")
        return metric, 1.0 if metric else -1.0
    if operator is RequirementOperator.IS_FALSE:
        if not isinstance(metric, bool):
            raise MetricError(f"{requirement.requirement_id} requires a boolean metric")
        return not metric, 1.0 if not metric else -1.0
    if metric is None or isinstance(metric, bool | str):
        raise MetricError(f"{requirement.requirement_id} requires a numeric metric")
    numeric = float(metric)
    if operator in {RequirementOperator.GE, RequirementOperator.GT}:
        threshold = _numeric_operand(requirement.value)
        margin = numeric - threshold
        return (margin >= 0.0 if operator is RequirementOperator.GE else margin > 0.0), margin
    if operator in {RequirementOperator.LE, RequirementOperator.LT}:
        threshold = _numeric_operand(requirement.value)
        margin = threshold - numeric
        return (margin >= 0.0 if operator is RequirementOperator.LE else margin > 0.0), margin
    if operator is RequirementOperator.BETWEEN:
        lower = _numeric_operand(requirement.lower)
        upper = _numeric_operand(requirement.upper)
        margin = min(numeric - lower, upper - numeric)
        return margin >= 0.0, margin
    target = _numeric_operand(requirement.value)
    tolerance = _numeric_operand(requirement.tolerance)
    margin = tolerance - abs(numeric - target)
    return margin >= 0.0, margin


def _numeric_operand(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise MetricError("numeric requirement operand is missing")
    if not isinstance(value, int | float):
        raise MetricError("numeric requirement operand must be numeric")
    return float(value)
