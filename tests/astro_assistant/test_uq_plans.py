from typing import Any

import pytest
from pydantic import ValidationError

from astro_assistant.models import (
    ArtifactKind,
    AstroToolName,
    AstroWorkflowPlan,
    RiskLevel,
    WorkflowStep,
)
from astro_assistant.policy import PolicyDecision, evaluate_plan
from astro_assistant.registry import build_command_spec

DEFINITION_PATH = "examples/campaigns/orbit.yaml"
OUTPUT_DIR = "/tmp/astro-assistant/campaign"


def _step(
    tool: AstroToolName,
    inputs: dict[str, Any],
    risk: RiskLevel,
) -> WorkflowStep:
    return WorkflowStep(
        step_id=tool.value,
        tool=tool,
        description=f"Run {tool.value}.",
        inputs=inputs,
        risk=risk,
    )


def _plan(step: WorkflowStep, *, requires_approval: bool = True) -> AstroWorkflowPlan:
    return AstroWorkflowPlan(
        plan_id="campaign-plan",
        title="Campaign plan",
        user_intent="Inspect or execute an uncertainty campaign",
        requires_approval=requires_approval,
        steps=[step],
    )


def test_campaign_contract_enums_are_stable() -> None:
    assert [
        AstroToolName.VALIDATE_CAMPAIGN.value,
        AstroToolName.RUN_CAMPAIGN.value,
        AstroToolName.SUMMARIZE_CAMPAIGN.value,
    ] == ["validate_campaign", "run_campaign", "summarize_campaign"]
    assert [
        ArtifactKind.CAMPAIGN_DEFINITION.value,
        ArtifactKind.CAMPAIGN_RESULT.value,
        ArtifactKind.CAMPAIGN_SUMMARY.value,
    ] == ["campaign_definition", "campaign_result", "campaign_summary"]


def test_validate_campaign_is_a_fixed_read_only_command() -> None:
    command = build_command_spec(
        _step(
            AstroToolName.VALIDATE_CAMPAIGN,
            {"definition_path": DEFINITION_PATH},
            RiskLevel.READ_ONLY,
        ),
        cwd="/workspace",
    )

    assert command.argv == ["astro", "validate-campaign", DEFINITION_PATH]
    assert command.cwd == "/workspace"
    assert command.writes == []


@pytest.mark.parametrize("resume", [False, True])
def test_run_campaign_has_typed_resume_and_records_output_write(resume: bool) -> None:
    command = build_command_spec(
        _step(
            AstroToolName.RUN_CAMPAIGN,
            {
                "definition_path": DEFINITION_PATH,
                "output_dir": OUTPUT_DIR,
                "resume": resume,
            },
            RiskLevel.WRITES_ARTIFACTS,
        )
    )

    expected = [
        "astro",
        "run-campaign",
        DEFINITION_PATH,
        "--output-dir",
        OUTPUT_DIR,
    ]
    if resume:
        expected.append("--resume")
    assert command.argv == expected
    assert command.writes == [OUTPUT_DIR]


def test_summarize_campaign_is_read_only_without_an_output_input() -> None:
    command = build_command_spec(
        _step(
            AstroToolName.SUMMARIZE_CAMPAIGN,
            {"output_dir": OUTPUT_DIR},
            RiskLevel.READ_ONLY,
        )
    )

    assert command.argv == ["astro", "summarize-campaign", OUTPUT_DIR]
    assert command.writes == []


@pytest.mark.parametrize(
    ("tool", "inputs"),
    [
        (AstroToolName.VALIDATE_CAMPAIGN, {}),
        (AstroToolName.RUN_CAMPAIGN, {"output_dir": OUTPUT_DIR}),
        (AstroToolName.RUN_CAMPAIGN, {"definition_path": DEFINITION_PATH}),
        (AstroToolName.SUMMARIZE_CAMPAIGN, {}),
    ],
)
def test_campaign_commands_reject_missing_inputs(
    tool: AstroToolName, inputs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        build_command_spec(_step(tool, inputs, RiskLevel.READ_ONLY))


@pytest.mark.parametrize(
    ("tool", "inputs"),
    [
        (
            AstroToolName.VALIDATE_CAMPAIGN,
            {"definition_path": DEFINITION_PATH, "flags": ["--help"]},
        ),
        (
            AstroToolName.RUN_CAMPAIGN,
            {
                "definition_path": DEFINITION_PATH,
                "output_dir": OUTPUT_DIR,
                "provider": "remote",
            },
        ),
        (
            AstroToolName.RUN_CAMPAIGN,
            {
                "definition_path": DEFINITION_PATH,
                "output_dir": OUTPUT_DIR,
                "resume": "true",
            },
        ),
        (
            AstroToolName.SUMMARIZE_CAMPAIGN,
            {"output_dir": OUTPUT_DIR, "output": "/tmp/summary.txt"},
        ),
        (
            AstroToolName.SUMMARIZE_CAMPAIGN,
            {"output_dir": OUTPUT_DIR, "promote": True},
        ),
    ],
)
def test_campaign_commands_reject_injected_or_untyped_inputs(
    tool: AstroToolName, inputs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        build_command_spec(_step(tool, inputs, RiskLevel.READ_ONLY))


@pytest.mark.parametrize("path", ["", "--help", "-c", "bad\x00path"])
def test_campaign_commands_reject_option_like_or_invalid_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        build_command_spec(
            _step(
                AstroToolName.VALIDATE_CAMPAIGN,
                {"definition_path": path},
                RiskLevel.READ_ONLY,
            )
        )


def test_run_campaign_requires_approval_even_if_risk_is_injected_as_read_only() -> None:
    decision = evaluate_plan(
        _plan(
            _step(
                AstroToolName.RUN_CAMPAIGN,
                {"definition_path": DEFINITION_PATH, "output_dir": OUTPUT_DIR},
                RiskLevel.READ_ONLY,
            )
        ),
        dry_run=False,
        approved=False,
    )

    assert decision.allowed is False
    assert decision.warnings == [
        "run_campaign must use writes_artifacts risk classification",
        "execution requires approval because the plan writes artifacts",
    ]


def test_run_campaign_cannot_disable_approval_at_plan_level() -> None:
    decision = evaluate_plan(
        _plan(
            _step(
                AstroToolName.RUN_CAMPAIGN,
                {"definition_path": DEFINITION_PATH, "output_dir": OUTPUT_DIR},
                RiskLevel.WRITES_ARTIFACTS,
            ),
            requires_approval=False,
        ),
        dry_run=False,
        approved=False,
    )

    assert decision.allowed is False
    assert decision.warnings == [
        "execution requires approval because the plan writes artifacts"
    ]


def test_approved_run_campaign_is_allowed_with_write_risk() -> None:
    decision = evaluate_plan(
        _plan(
            _step(
                AstroToolName.RUN_CAMPAIGN,
                {"definition_path": DEFINITION_PATH, "output_dir": OUTPUT_DIR},
                RiskLevel.WRITES_ARTIFACTS,
            )
        ),
        dry_run=False,
        approved=True,
    )

    assert decision == PolicyDecision(allowed=True, warnings=[])


@pytest.mark.parametrize(
    "tool", [AstroToolName.VALIDATE_CAMPAIGN, AstroToolName.SUMMARIZE_CAMPAIGN]
)
def test_read_only_campaign_tools_reject_write_risk(tool: AstroToolName) -> None:
    inputs = (
        {"definition_path": DEFINITION_PATH}
        if tool is AstroToolName.VALIDATE_CAMPAIGN
        else {"output_dir": OUTPUT_DIR}
    )
    decision = evaluate_plan(
        _plan(_step(tool, inputs, RiskLevel.WRITES_ARTIFACTS)),
        dry_run=False,
        approved=True,
    )

    assert decision.allowed is False
    assert decision.warnings == [f"{tool.value} must use read_only risk classification"]


def test_existing_od_command_and_policy_remain_unchanged() -> None:
    step = _step(
        AstroToolName.VALIDATE_SCENARIO,
        {"scenario_path": "examples/scenarios/leo_two_station_od.yaml"},
        RiskLevel.READ_ONLY,
    )

    command = build_command_spec(step)
    decision = evaluate_plan(_plan(step), dry_run=False, approved=False)

    assert command.argv == [
        "astro",
        "validate",
        "examples/scenarios/leo_two_station_od.yaml",
    ]
    assert decision == PolicyDecision(allowed=True, warnings=[])
