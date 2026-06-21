import pytest

from astro_assistant.models import AstroToolName
from astro_assistant.planner import DeterministicPlanner

UNSUPPORTED_PROMPT_MESSAGE = "deterministic planner currently supports the local OD demo only"


def test_deterministic_planner_builds_local_od_workflow() -> None:
    planner = DeterministicPlanner()

    plan = planner.plan("Run the local OD demo and export TDM")

    assert plan.plan_id == "local-od-demo"
    assert [step.tool for step in plan.steps] == [
        AstroToolName.VALIDATE_SCENARIO,
        AstroToolName.SYNTH_MEASUREMENTS,
        AstroToolName.EXPORT_MEASUREMENTS,
        AstroToolName.ESTIMATE_MEASUREMENTS,
    ]
    assert plan.steps[-1].inputs["output"] == "/tmp/astro-assistant/estimate.json"


def test_deterministic_planner_rejects_non_od_substring_matches() -> None:
    planner = DeterministicPlanner()

    with pytest.raises(ValueError, match=UNSUPPORTED_PROMPT_MESSAGE):
        planner.plan("Run food demo")
