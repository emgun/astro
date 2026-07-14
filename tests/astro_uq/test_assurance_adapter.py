from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from astro_assurance.io import load_post_launch_assurance_scenario
from astro_assurance.models import PostLaunchAssuranceScenario
from astro_assurance.runner import run_post_launch_assurance
from astro_twin.io import load_twin_scenario
from astro_uq.adapters.assurance import assurance_metric_registry, assurance_parameter_registry
from astro_uq.models import (
    DistributionKind,
    DistributionSpec,
    MetricSpec,
    MetricValueKind,
    ParameterRealization,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
)


def _parameter(parameter_id: str, target: str, unit: str) -> UncertainParameter:
    return UncertainParameter(
        parameter_id=parameter_id,
        target=target,
        unit=unit,
        uncertainty_kind=UncertaintyKind.EPISTEMIC,
        distribution=DistributionSpec(kind=DistributionKind.CONSTANT, value=1.0),
    )


def test_assurance_registry_applies_cross_workflow_overrides_and_extracts_metrics() -> None:
    scenario = load_post_launch_assurance_scenario(
        "examples/assurance/post_launch_orbit_acquisition.yaml"
    )
    twin = load_twin_scenario(scenario.twin_scenario)
    parameters = (
        _parameter("position_x", "mission_assurance.insertion.position_x", "km"),
        _parameter("range_sigma", "mission_assurance.tracking.range_sigma_km", "km"),
        _parameter("execution_scale", "mission_assurance.correction.execution_scale", "1"),
        _parameter(
            "solar_efficiency",
            "mission_assurance.digital_twin.power.solar_array_efficiency",
            "1",
        ),
        _parameter(
            "bus_emissivity",
            "mission_assurance.digital_twin.thermal_nodes.bus.emissivity",
            "1",
        ),
    )
    uncertainty = UncertaintyModel(parameters=parameters)
    resolved, evidence = assurance_parameter_registry(twin).apply(
        workflow="mission_assurance",
        scenario=scenario,
        uncertainty=uncertainty,
        realization=ParameterRealization(
            sample_id="sample-000001",
            sample_index=0,
            normalized_values={item.parameter_id: 0.5 for item in parameters},
            physical_values={
                "position_x": 2.1,
                "range_sigma": 0.012,
                "execution_scale": 0.9,
                "solar_efficiency": 0.285,
                "bus_emissivity": 0.77,
            },
        ),
    )
    parsed = PostLaunchAssuranceScenario.model_validate(resolved).model_copy(
        update={"source_path": scenario.source_path, "source_digest": scenario.source_digest}
    )
    result = run_post_launch_assurance(parsed)

    assert evidence.bindings[0].target == "mission_assurance.insertion.position_x"
    assert result.truth_corrected_scenario.maneuvers[0].delta_v_km_s == pytest.approx(
        tuple(0.9 * value for value in result.correction_maneuver.delta_v_km_s)
    )
    assert result.metadata["correction_execution_scale"] == 0.9
    assert result.measurements[0].sigma == pytest.approx(0.012)
    values = assurance_metric_registry().extract(
        workflow="mission_assurance",
        result=result,
        specifications=(
            MetricSpec(
                metric_id="passed",
                extractor="mission_assurance.passed",
                value_kind=MetricValueKind.BOOLEAN,
            ),
            MetricSpec(
                metric_id="executed",
                extractor="mission_assurance.executed_delta_v_km_s",
                value_kind=MetricValueKind.NUMERIC,
                unit="km/s",
            ),
        ),
    )
    assert values["passed"] is result.passed
    assert values["executed"] == pytest.approx(result.metadata["executed_delta_v_km_s"])


def test_reference_assurance_campaign_validates_and_runs_two_cases(tmp_path: Path) -> None:
    from astro_uq.cli import _bind_resolved_dependencies, _runtime
    from astro_uq.io import load_campaign_definition
    from astro_uq.runner import run_campaign

    definition_path = Path("examples/campaigns/leo_mission_assurance_robustness.yaml")
    definition = _bind_resolved_dependencies(
        load_campaign_definition(definition_path), definition_path
    )
    definition = definition.model_copy(
        update={"sampler": definition.sampler.model_copy(update={"samples": 2})}
    )
    result = run_campaign(
        definition,
        _runtime(definition, definition_path),
        output_dir=tmp_path / "campaign",
        software_compatibility={"astro": "test"},
    )

    assert result.statistics.completed_samples == 2
    assert result.statistics.outcome_counts == {"success": 2}
    assert set(result.statistics.requirement_probabilities) == {
        requirement.requirement_id for requirement in definition.requirements
    }


def test_reference_campaign_resolves_from_outside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_uq.cli import _bind_resolved_dependencies, _runtime
    from astro_uq.io import load_campaign_definition

    definition_path = Path(
        "examples/campaigns/leo_mission_assurance_robustness.yaml"
    ).resolve()
    monkeypatch.chdir(tmp_path)
    definition = _bind_resolved_dependencies(
        load_campaign_definition(definition_path), definition_path
    )
    runtime = _runtime(definition, definition_path)

    scenario = PostLaunchAssuranceScenario.model_validate(runtime.scenario)
    assert Path(scenario.launch_scenario).is_absolute()
    assert Path(scenario.tracking_scenario).is_absolute()
    assert Path(scenario.twin_scenario).is_absolute()


def test_resume_rejects_top_level_assurance_scenario_drift(tmp_path: Path) -> None:
    from astro_uq.cli import _bind_resolved_dependencies, _runtime
    from astro_uq.io import CampaignIOError, load_campaign_definition
    from astro_uq.runner import run_campaign

    repository = Path.cwd()
    assurance_payload = yaml.safe_load(
        (repository / "examples/assurance/post_launch_orbit_acquisition.yaml").read_text()
    )
    for field in ("launch_scenario", "tracking_scenario", "twin_scenario"):
        assurance_payload[field] = str((repository / assurance_payload[field]).resolve())
    assurance_path = tmp_path / "assurance.yaml"
    assurance_path.write_text(yaml.safe_dump(assurance_payload), encoding="utf-8")

    campaign_payload = yaml.safe_load(
        (repository / "examples/campaigns/leo_mission_assurance_robustness.yaml").read_text()
    )
    campaign_payload["workflow"]["scenario"] = str(assurance_path)
    campaign_payload["sampler"]["samples"] = 2
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(campaign_payload), encoding="utf-8")

    definition = _bind_resolved_dependencies(
        load_campaign_definition(campaign_path), campaign_path
    )
    output = tmp_path / "output"
    run_campaign(
        definition,
        _runtime(definition, campaign_path),
        output_dir=output,
        software_compatibility={"astro": "test"},
    )
    assurance_payload["description"] = "changed after completed campaign"
    assurance_path.write_text(yaml.safe_dump(assurance_payload), encoding="utf-8")
    changed = _bind_resolved_dependencies(
        load_campaign_definition(campaign_path), campaign_path
    )

    with pytest.raises(CampaignIOError, match="definition digest"):
        run_campaign(
            changed,
            _runtime(changed, campaign_path),
            output_dir=output,
            software_compatibility={"astro": "test"},
            resume=True,
        )
