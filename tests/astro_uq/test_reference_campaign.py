from __future__ import annotations

from pathlib import Path

import pytest

from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.runner import run_mission_lifecycle
from astro_twin.io import load_twin_scenario
from astro_uq.adapters.lifecycle import (
    lifecycle_metric_registry,
    lifecycle_parameter_registry,
)
from astro_uq.io import load_campaign_definition
from astro_uq.runner import CampaignRuntime, run_campaign


def test_reference_lifecycle_campaign_runs_end_to_end(tmp_path: Path) -> None:
    definition = load_campaign_definition(
        "examples/campaigns/leo_lifecycle_robustness.yaml"
    )
    scenario = load_mission_lifecycle_scenario(definition.workflow.scenario)

    result = run_campaign(
        definition,
        CampaignRuntime(
            scenario=scenario,
            parameters=lifecycle_parameter_registry(
                load_twin_scenario(scenario.twin_scenario)
            ),
            metrics=lifecycle_metric_registry(
                load_twin_scenario(scenario.twin_scenario)
            ),
            evaluate=lambda model: run_mission_lifecycle(type(scenario).model_validate(model)),
        ),
        output_dir=tmp_path / "campaign",
        software_compatibility={"astro": "0.1.0"},
    )

    assert result.statistics.completed_samples == 8
    assert result.statistics.outcome_counts == {"success": 8}
    assert set(result.statistics.requirement_probabilities) == {
        "lifecycle_success",
        "propellant_reserve",
        "twin_actuator_utilization",
        "twin_battery_soc",
        "twin_contact_available",
        "twin_mass_budget_rollup",
        "twin_propellant_fraction",
        "twin_pointing",
        "twin_slew_rate",
        "twin_thermal",
        "twin_torque",
        "twin_worst_observed_link_margin",
    }
    assert result.statistics.requirement_probabilities == {
        "lifecycle_success": 1.0,
        "propellant_reserve": 1.0,
        "twin_actuator_utilization": 1.0,
        "twin_battery_soc": 1.0,
        "twin_contact_available": 1.0,
        "twin_mass_budget_rollup": 1.0,
        "twin_propellant_fraction": 1.0,
        "twin_pointing": 1.0,
        "twin_slew_rate": 1.0,
        "twin_thermal": 1.0,
        "twin_torque": 1.0,
        "twin_worst_observed_link_margin": 1.0,
    }
    means = {metric.metric_id: metric.mean for metric in result.statistics.metrics}
    assert means["deorbit_propellant_used"] == pytest.approx(
        32.8418249815597, rel=1.0e-9
    )
    assert means["reentry_peak_dynamic_pressure"] == pytest.approx(
        32095.676920547012, rel=1.0e-9
    )
    assert means["reentry_peak_heat_rate"] == pytest.approx(
        1463581.2984281622, rel=1.0e-9
    )
    assert (tmp_path / "campaign" / "cases.jsonl").exists()
