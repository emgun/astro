from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_uq.models import (
    CampaignDefinition,
    ConfidenceIntervalStopping,
    CorrelationModel,
    DistributionKind,
    DistributionSpec,
    EvaluatorKind,
    EvaluatorSpec,
    MetricSpec,
    MetricValueKind,
    RequirementOperator,
    RequirementSpec,
    SamplerKind,
    SamplerSpec,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
    WorkflowSpec,
)


def _parameter(parameter_id: str = "mass") -> UncertainParameter:
    return UncertainParameter(
        parameter_id=parameter_id,
        target="orbit.spacecraft.mass_kg",
        unit="kg",
        uncertainty_kind=UncertaintyKind.EPISTEMIC,
        distribution=DistributionSpec(
            kind=DistributionKind.TRIANGULAR,
            low=90.0,
            mode=100.0,
            high=120.0,
        ),
    )


def _definition() -> CampaignDefinition:
    return CampaignDefinition(
        campaign_id="orbit-robustness",
        workflow=WorkflowSpec(kind="orbit", scenario="examples/scenarios/leo_two_body.yaml"),
        uncertainty=UncertaintyModel(parameters=(_parameter(),)),
        sampler=SamplerSpec(kind=SamplerKind.LATIN_HYPERCUBE, samples=16, seed=7),
        evaluator=EvaluatorSpec(
            evaluator_id="local-orbit",
            kind=EvaluatorKind.AUTHORITATIVE,
            workflow="orbit",
            implementation_version="1",
            backend="local",
            claim_boundary="deterministic_design_screening",
        ),
        metrics=(
            MetricSpec(
                metric_id="final_altitude",
                extractor="orbit.final_altitude_km",
                value_kind=MetricValueKind.NUMERIC,
                unit="km",
            ),
        ),
        requirements=(
            RequirementSpec(
                requirement_id="minimum_altitude",
                metric_id="final_altitude",
                operator=RequirementOperator.GE,
                value=180.0,
            ),
        ),
    )


def test_campaign_definition_accepts_typed_contract() -> None:
    definition = _definition()

    assert definition.schema_version == "1.0"
    assert definition.sampler.samples == 16
    assert definition.requirements[0].metric_id == "final_altitude"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"kind": "uniform", "low": 2.0, "high": 1.0}, "low < high"),
        ({"kind": "normal", "mean": 0.0, "sigma": 0.0}, "sigma > 0"),
        (
            {"kind": "categorical", "labels": ["a", "b"], "probabilities": [0.8, 0.8]},
            "sum to one",
        ),
    ],
)
def test_distribution_rejects_invalid_shape(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        DistributionSpec.model_validate(payload)


def test_distribution_rejects_boolean_numeric_input() -> None:
    with pytest.raises(ValidationError, match="not booleans"):
        DistributionSpec(kind=DistributionKind.CONSTANT, value=True)


def test_uncertainty_model_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        UncertaintyModel(parameters=(_parameter(), _parameter()))


def test_uncertainty_model_rejects_unknown_correlation_parameter() -> None:
    with pytest.raises(ValidationError, match="unknown parameter"):
        UncertaintyModel(
            parameters=(_parameter(),),
            correlations=(
                CorrelationModel(
                    parameter_ids=("mass", "area"),
                    matrix=((1.0, 0.0), (0.0, 1.0)),
                ),
            ),
        )


def test_campaign_rejects_requirement_for_unknown_metric() -> None:
    payload = _definition().model_dump()
    payload["requirements"][0]["metric_id"] = "missing"

    with pytest.raises(ValidationError, match="configured metric"):
        CampaignDefinition.model_validate(payload)


def test_surrogate_evaluator_requires_model_artifact() -> None:
    with pytest.raises(ValidationError, match="model artifact"):
        EvaluatorSpec(
            evaluator_id="orbit-surrogate",
            kind=EvaluatorKind.SURROGATE,
            workflow="orbit",
            implementation_version="1",
            claim_boundary="experimental",
        )


def test_campaign_rejects_unreachable_adaptive_stopping_maximum() -> None:
    payload = _definition().model_dump(mode="python")
    payload["sampler"]["samples"] = 4
    payload["stopping"] = ConfidenceIntervalStopping(
        requirement_id="minimum_altitude",
        target_half_width=0.1,
        minimum_samples=4,
        maximum_samples=8,
    ).model_dump(mode="python")

    with pytest.raises(ValidationError, match="reach adaptive stopping maximum"):
        CampaignDefinition.model_validate(payload)
