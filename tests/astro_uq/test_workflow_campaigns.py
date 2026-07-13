from __future__ import annotations

from pathlib import Path

import pytest

from astro_reentry.backends import simulate_reentry_with_backend
from astro_reentry.io import load_reentry_scenario
from astro_reentry.models import ReentryScenario
from astro_twin.io import load_twin_scenario
from astro_twin.models import DigitalTwinScenario
from astro_twin.runner import run_digital_twin
from astro_uq.adapters.reentry import reentry_metric_registry, reentry_parameter_registry
from astro_uq.adapters.twin import twin_metric_registry, twin_parameter_registry
from astro_uq.models import (
    CampaignDefinition,
    CaseObservation,
    DistributionKind,
    DistributionSpec,
    EvaluatorKind,
    EvaluatorSpec,
    MetricSpec,
    MetricValueKind,
    SamplerKind,
    SamplerSpec,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
    WorkflowSpec,
)
from astro_uq.runner import CampaignRuntime, run_campaign


@pytest.mark.parametrize("workflow", ["digital_twin", "reentry"])
def test_checked_workflow_lhs_campaigns_are_deterministic(tmp_path: Path, workflow: str) -> None:
    if workflow == "digital_twin":
        scenario = load_twin_scenario("examples/twin/leo_observer.yaml")
        target = "digital_twin.power.solar_array_efficiency"
        metric = MetricSpec(
            metric_id="minimum_soc",
            extractor="digital_twin.min_battery_soc_fraction",
            value_kind=MetricValueKind.NUMERIC,
            unit="1",
        )
        runtime = CampaignRuntime(
            scenario=scenario,
            parameters=twin_parameter_registry(scenario),
            metrics=twin_metric_registry(scenario),
            evaluate=lambda model: run_digital_twin(DigitalTwinScenario.model_validate(model)),
        )
    else:
        scenario = load_reentry_scenario("examples/reentry/ballistic_capsule.yaml")
        target = "reentry.atmosphere.density_scale_factor"
        metric = MetricSpec(
            metric_id="peak_heat_rate",
            extractor="reentry.peak_heat_rate_w_m2",
            value_kind=MetricValueKind.NUMERIC,
            unit="W/m^2",
        )
        runtime = CampaignRuntime(
            scenario=scenario,
            parameters=reentry_parameter_registry(),
            metrics=reentry_metric_registry(),
            evaluate=lambda model: simulate_reentry_with_backend(
                ReentryScenario.model_validate(model), "local"
            ),
        )
    definition = CampaignDefinition(
        campaign_id=f"{workflow}-lhs",
        workflow=WorkflowSpec(kind=workflow, scenario="checked-fixture.yaml"),
        uncertainty=UncertaintyModel(
            parameters=(
                UncertainParameter(
                    parameter_id="scale",
                    target=target,
                    unit="1",
                    uncertainty_kind=UncertaintyKind.EPISTEMIC,
                    distribution=DistributionSpec(
                        kind=DistributionKind.UNIFORM,
                        low=0.95,
                        high=1.0 if workflow == "digital_twin" else 1.05,
                    ),
                ),
            )
        ),
        sampler=SamplerSpec(kind=SamplerKind.LATIN_HYPERCUBE, samples=2, seed=17),
        evaluator=EvaluatorSpec(
            evaluator_id=f"local-{workflow}",
            kind=EvaluatorKind.AUTHORITATIVE,
            workflow=workflow,
            implementation_version="1",
            backend="local",
            claim_boundary="deterministic_design_screening",
        ),
        metrics=(metric,),
    )

    first = run_campaign(
        definition,
        runtime,
        output_dir=tmp_path / f"{workflow}-first",
        software_compatibility={"astro": "test"},
    )
    second = run_campaign(
        definition,
        runtime,
        output_dir=tmp_path / f"{workflow}-second",
        software_compatibility={"astro": "test"},
    )

    assert first.statistics == second.statistics
    assert first.statistics.outcome_counts == {"success": 2}
    parsed_first_cases = [
        CaseObservation.model_validate_json(line)
        for line in (tmp_path / f"{workflow}-first" / "cases.jsonl").read_text().splitlines()
    ]
    assert all(case.evaluation_timing is not None for case in parsed_first_cases)
    assert all(case.metadata.get("source_warnings") for case in parsed_first_cases)
    first_cases = [
        case.model_dump(mode="json", exclude={"evaluation_timing"}) for case in parsed_first_cases
    ]
    second_cases = [
        CaseObservation.model_validate_json(line).model_dump(
            mode="json", exclude={"evaluation_timing"}
        )
        for line in (tmp_path / f"{workflow}-second" / "cases.jsonl").read_text().splitlines()
    ]
    assert first_cases == second_cases
