import pytest
from pydantic import ValidationError

from astro_assistant.models import (
    ArtifactKind,
    AstroToolName,
    AstroWorkflowPlan,
    RiskLevel,
    WorkflowArtifact,
    WorkflowStep,
)


def test_workflow_plan_validates_step_ids_and_artifacts() -> None:
    plan = AstroWorkflowPlan(
        plan_id="local-od-demo",
        title="Local OD demo",
        user_intent="Run local OD workflow",
        steps=[
            WorkflowStep(
                step_id="validate_scenario",
                tool=AstroToolName.VALIDATE_SCENARIO,
                description="Validate the OD scenario.",
                inputs={"scenario_path": "examples/scenarios/leo_two_station_od.yaml"},
                outputs=[
                    WorkflowArtifact(
                        path="examples/scenarios/leo_two_station_od.yaml",
                        kind=ArtifactKind.SCENARIO,
                    )
                ],
                risk=RiskLevel.READ_ONLY,
            )
        ],
    )

    assert plan.requires_approval is True
    assert plan.steps[0].tool == AstroToolName.VALIDATE_SCENARIO


def test_workflow_step_rejects_spaces_in_step_id() -> None:
    with pytest.raises(ValidationError):
        WorkflowStep(
            step_id="validate scenario",
            tool=AstroToolName.VALIDATE_SCENARIO,
            description="Invalid id.",
            risk=RiskLevel.READ_ONLY,
        )
