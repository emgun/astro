from __future__ import annotations

import pytest

from astro_core.models import AstroModel
from astro_uq.metrics import MetricError, MetricExtractor, MetricRegistry, evaluate_requirements
from astro_uq.models import (
    MetricSpec,
    MetricValueKind,
    RequirementOperator,
    RequirementSpec,
)


class FixtureResult(AstroModel):
    altitude_km: float
    passed: bool
    optional_value: float | None = None


def _registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(
        MetricExtractor(
            extractor_id="fixture.altitude_km",
            workflow="fixture",
            value_kind=MetricValueKind.NUMERIC,
            unit="km",
            extract=lambda result: FixtureResult.model_validate(result).altitude_km,
        )
    )
    registry.register(
        MetricExtractor(
            extractor_id="fixture.optional",
            workflow="fixture",
            value_kind=MetricValueKind.NUMERIC,
            unit="1",
            extract=lambda result: FixtureResult.model_validate(result).optional_value,
        )
    )
    return registry


def test_registry_extracts_allow_listed_metric() -> None:
    values = _registry().extract(
        workflow="fixture",
        result=FixtureResult(altitude_km=220.0, passed=True),
        specifications=(
            MetricSpec(
                metric_id="altitude",
                extractor="fixture.altitude_km",
                value_kind=MetricValueKind.NUMERIC,
                unit="km",
            ),
        ),
    )

    assert values == {"altitude": 220.0}


def test_registry_rejects_unit_mismatch() -> None:
    with pytest.raises(MetricError, match="unit mismatch"):
        _registry().extract(
            workflow="fixture",
            result=FixtureResult(altitude_km=220.0, passed=True),
            specifications=(
                MetricSpec(
                    metric_id="altitude",
                    extractor="fixture.altitude_km",
                    value_kind=MetricValueKind.NUMERIC,
                    unit="m",
                ),
            ),
        )


def test_requirements_report_signed_margins_and_missing_values() -> None:
    outcomes = evaluate_requirements(
        {"altitude": 220.0, "optional": None},
        (
            RequirementSpec(
                requirement_id="minimum_altitude",
                metric_id="altitude",
                operator=RequirementOperator.GE,
                value=200.0,
            ),
            RequirementSpec(
                requirement_id="optional_gate",
                metric_id="optional",
                operator=RequirementOperator.GE,
                value=0.0,
            ),
        ),
    )

    assert outcomes[0].passed is True
    assert outcomes[0].margin == 20.0
    assert outcomes[1].passed is None
    assert outcomes[1].reason == "metric_missing_or_not_applicable"


def test_between_requirement_fails_outside_interval() -> None:
    outcome = evaluate_requirements(
        {"temperature": 330.0},
        (
            RequirementSpec(
                requirement_id="thermal_band",
                metric_id="temperature",
                operator=RequirementOperator.BETWEEN,
                lower=270.0,
                upper=320.0,
            ),
        ),
    )[0]

    assert outcome.passed is False
    assert outcome.margin == -10.0
