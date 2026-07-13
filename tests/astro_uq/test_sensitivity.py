from __future__ import annotations

import numpy as np
import pytest

from astro_uq.io import load_campaign_definition
from astro_uq.models import (
    CampaignDefinition,
    CaseObservation,
    DistributionKind,
    DistributionSpec,
    EvaluatorKind,
    EvaluatorSpec,
    MetricSpec,
    MetricValueKind,
    OutcomeStatus,
    ParameterRealization,
    RequirementOperator,
    RequirementOutcome,
    RequirementSpec,
    SamplerKind,
    SamplerSpec,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
    WorkflowSpec,
)
from astro_uq.sensitivity import SensitivityAnalysisError, analyze_campaign_sensitivity


def _parameter(parameter_id: str) -> UncertainParameter:
    return UncertainParameter(
        parameter_id=parameter_id,
        target=f"fixture.{parameter_id}",
        unit="1",
        uncertainty_kind=UncertaintyKind.EPISTEMIC,
        distribution=DistributionSpec(kind=DistributionKind.UNIFORM, low=0.0, high=1.0),
    )


def _definition(samples: int = 32) -> CampaignDefinition:
    return CampaignDefinition(
        campaign_id="sensitivity-fixture",
        workflow=WorkflowSpec(kind="fixture", scenario="fixture.yaml"),
        uncertainty=UncertaintyModel(parameters=(_parameter("x"), _parameter("z"))),
        sampler=SamplerSpec(kind=SamplerKind.LATIN_HYPERCUBE, samples=samples, seed=7),
        evaluator=EvaluatorSpec(
            evaluator_id="fixture",
            kind=EvaluatorKind.AUTHORITATIVE,
            workflow="fixture",
            implementation_version="1",
            claim_boundary="fixture_design_space",
        ),
        metrics=(
            MetricSpec(
                metric_id="response",
                extractor="fixture.response",
                value_kind=MetricValueKind.NUMERIC,
                unit="1",
            ),
        ),
        requirements=(
            RequirementSpec(
                requirement_id="positive_margin",
                metric_id="response",
                operator=RequirementOperator.GE,
                value=0.0,
            ),
        ),
    )


def _evidence(
    count: int = 32,
) -> tuple[tuple[ParameterRealization, ...], tuple[CaseObservation, ...]]:
    rng = np.random.default_rng(17)
    x_values = (np.arange(count, dtype=float) + 0.5) / count
    z_values = (rng.permutation(count).astype(float) + 0.5) / count
    response = 3.0 * x_values + 0.35 * z_values + 0.03 * np.sin(5.0 * z_values)
    samples = tuple(
        ParameterRealization(
            sample_id=f"sample-{index:04d}",
            sample_index=index,
            physical_values={"x": float(x_values[index]), "z": float(z_values[index])},
            normalized_values={"x": float(x_values[index]), "z": float(z_values[index])},
            weight=1.0 / count,
        )
        for index in range(count)
    )
    observations = tuple(
        CaseObservation(
            sample_id=sample.sample_id,
            outcome_status=OutcomeStatus.SUCCESS,
            metric_values={"response": float(response[index])},
            requirements=(
                RequirementOutcome(
                    requirement_id="positive_margin",
                    passed=True,
                    margin=float(response[index]),
                ),
            ),
            evaluator_id="fixture",
            claim_boundary="fixture_design_space",
        )
        for index, sample in enumerate(samples)
    )
    return samples, observations


def _analyze(
    samples: tuple[ParameterRealization, ...],
    observations: tuple[CaseObservation, ...],
):
    return analyze_campaign_sensitivity(
        _definition(samples=len(samples)),
        samples,
        observations,
        metric_ids=("response",),
        requirement_margin_ids=("positive_margin",),
        definition_digest="d" * 64,
        samples_digest="a" * 64,
        cases_digest="b" * 64,
    )


def test_rank_attribution_identifies_strongest_parameter() -> None:
    samples, observations = _evidence()

    report = _analyze(samples, observations)

    assert report.sample_count == 32
    assert report.parameter_count == 2
    assert report.effective_sample_size == 32.0
    assert report.ranked_design_rank == 2
    assert report.residual_degrees_of_freedom == 29
    assert report.ranked_design_condition_number < 2.0
    assert {target.target_id for target in report.targets} == {
        "response",
        "positive_margin",
    }
    for target in report.targets:
        estimates = {estimate.parameter_id: estimate for estimate in target.estimates}
        assert target.largest_absolute_prcc_parameter_id == "x"
        assert estimates["x"].absolute_prcc_rank == 1
        assert estimates["x"].partial_rank_correlation > 0.95
        assert abs(estimates["x"].partial_rank_correlation) > abs(
            estimates["z"].partial_rank_correlation
        )
    assert "not causal" in report.warnings[0]


def test_sensitivity_rejects_insufficient_cases() -> None:
    samples, observations = _evidence(count=15)

    with pytest.raises(SensitivityAnalysisError, match="at least 30 cases"):
        _analyze(samples, observations)


def test_sensitivity_rejects_unequal_weights() -> None:
    samples, observations = _evidence()
    samples = (
        samples[0].model_copy(update={"weight": 0.5}),
        *samples[1:],
    )

    with pytest.raises(SensitivityAnalysisError, match="equal sample weights"):
        _analyze(samples, observations)


def test_sensitivity_rejects_failed_case() -> None:
    samples, observations = _evidence()
    observations = (
        observations[0].model_copy(
            update={"outcome_status": OutcomeStatus.EXECUTION_FAILURE}
        ),
        *observations[1:],
    )

    with pytest.raises(SensitivityAnalysisError, match="every campaign case to succeed"):
        _analyze(samples, observations)


def test_sensitivity_rejects_constant_target() -> None:
    samples, observations = _evidence()
    observations = tuple(
        observation.model_copy(update={"metric_values": {"response": 1.0}})
        for observation in observations
    )

    with pytest.raises(SensitivityAnalysisError, match="target 'response' is constant"):
        analyze_campaign_sensitivity(
            _definition(),
            samples,
            observations,
            metric_ids=("response",),
            definition_digest="d" * 64,
            samples_digest="a" * 64,
            cases_digest="b" * 64,
        )


def test_sensitivity_rejects_singular_ranked_design() -> None:
    samples, observations = _evidence()
    samples = tuple(
        sample.model_copy(
            update={
                "physical_values": {
                    "x": sample.physical_values["x"],
                    "z": sample.physical_values["x"],
                }
            }
        )
        for sample in samples
    )

    with pytest.raises(SensitivityAnalysisError, match="design is singular"):
        _analyze(samples, observations)


def test_sensitivity_joins_only_completed_adaptive_cases() -> None:
    samples, observations = _evidence()

    report = _analyze(samples, observations[:30])

    assert report.sample_count == 30
    assert report.effective_sample_size == 30.0
    assert all(target.sample_count == 30 for target in report.targets)


def test_sensitivity_rejects_boolean_requirement_margin() -> None:
    samples, observations = _evidence()
    definition = _definition().model_copy(
        update={
            "requirements": (
                RequirementSpec(
                    requirement_id="positive_margin",
                    metric_id="response",
                    operator=RequirementOperator.IS_TRUE,
                ),
            )
        }
    )

    with pytest.raises(SensitivityAnalysisError, match="boolean sentinel margin"):
        analyze_campaign_sensitivity(
            definition,
            samples,
            observations,
            requirement_margin_ids=("positive_margin",),
            definition_digest="d" * 64,
            samples_digest="a" * 64,
            cases_digest="b" * 64,
        )


def test_sensitivity_rejects_input_with_fewer_than_five_unique_values() -> None:
    samples, observations = _evidence()
    samples = tuple(
        sample.model_copy(
            update={
                "physical_values": {
                    **sample.physical_values,
                    "z": float(sample.sample_index % 4),
                }
            }
        )
        for sample in samples
    )

    with pytest.raises(SensitivityAnalysisError, match="at least 5 unique values"):
        _analyze(samples, observations)


def test_checked_lifecycle_sensitivity_campaign_has_sufficient_design() -> None:
    definition = load_campaign_definition(
        "examples/campaigns/leo_lifecycle_sensitivity.yaml"
    )
    parameter_count = len(definition.uncertainty.parameters)

    assert definition.sampler.kind is SamplerKind.LATIN_HYPERCUBE
    assert definition.sampler.samples >= max(30, 5 * (parameter_count + 1))
    assert definition.retention.policy.value == "none"
    assert definition.metadata["purpose"] == "lifecycle_rank_sensitivity_and_margin_attribution"
    assert {
        "twin_battery_soc",
        "twin_thermal",
        "twin_pointing",
        "twin_torque",
        "twin_slew_rate",
        "twin_actuator_utilization",
        "twin_contact_available",
        "twin_propellant_fraction",
        "twin_mass_budget_rollup",
        "twin_worst_observed_link_margin",
    } <= {requirement.requirement_id for requirement in definition.requirements}


def test_prcc_matches_independent_tied_confounding_oracle() -> None:
    count = 40
    rng = np.random.default_rng(123)
    z_values = np.round(rng.normal(size=count), 1)
    x_values = np.round(0.85 * z_values + 0.35 * rng.normal(size=count), 1)
    response = np.round(
        1.1 * z_values - 0.75 * x_values + 0.25 * rng.normal(size=count),
        1,
    )
    samples = tuple(
        ParameterRealization(
            sample_id=f"oracle-{index:04d}",
            sample_index=index,
            physical_values={"x": float(x_values[index]), "z": float(z_values[index])},
            weight=1.0 / count,
        )
        for index in range(count)
    )
    observations = tuple(
        CaseObservation(
            sample_id=sample.sample_id,
            outcome_status=OutcomeStatus.SUCCESS,
            metric_values={"response": float(response[index])},
            evaluator_id="fixture",
            claim_boundary="fixture_design_space",
        )
        for index, sample in enumerate(samples)
    )

    report = analyze_campaign_sensitivity(
        _definition(samples=count),
        samples,
        observations,
        metric_ids=("response",),
        definition_digest="d" * 64,
        samples_digest="a" * 64,
        cases_digest="b" * 64,
    )

    estimates = {item.parameter_id: item for item in report.targets[0].estimates}
    assert estimates["x"].spearman_rho == pytest.approx(0.6715930488949894)
    assert estimates["x"].partial_rank_correlation == pytest.approx(-0.5293930574730037)
    assert estimates["z"].spearman_rho == pytest.approx(0.8229401913039145)
    assert estimates["z"].partial_rank_correlation == pytest.approx(0.7594854990930522)
    assert estimates["x"].input_tie_fraction == pytest.approx(0.4)
