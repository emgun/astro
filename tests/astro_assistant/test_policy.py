from astro_assistant.models import AstroToolName, AstroWorkflowPlan, RiskLevel, WorkflowStep
from astro_assistant.policy import PolicyDecision, evaluate_plan


def test_dry_run_allows_write_artifacts_without_execute() -> None:
    plan = AstroWorkflowPlan(
        plan_id="local-od-demo",
        title="Local OD demo",
        user_intent="Run local OD workflow",
        steps=[
            WorkflowStep(
                step_id="synth",
                tool=AstroToolName.SYNTH_MEASUREMENTS,
                description="Synthesize measurements.",
                inputs={"backend": "local", "output": "/tmp/astro-assistant/measurements.json"},
                risk=RiskLevel.WRITES_ARTIFACTS,
            )
        ],
    )

    decision = evaluate_plan(plan, dry_run=True, approved=False)

    assert decision == PolicyDecision(allowed=True, warnings=[])


def test_execute_requires_approval_when_plan_writes_artifacts() -> None:
    plan = AstroWorkflowPlan(
        plan_id="local-od-demo",
        title="Local OD demo",
        user_intent="Run local OD workflow",
        steps=[
            WorkflowStep(
                step_id="synth",
                tool=AstroToolName.SYNTH_MEASUREMENTS,
                description="Synthesize measurements.",
                inputs={"backend": "local", "output": "/tmp/astro-assistant/measurements.json"},
                risk=RiskLevel.WRITES_ARTIFACTS,
            )
        ],
    )

    decision = evaluate_plan(plan, dry_run=False, approved=False)

    assert decision.allowed is False
    assert decision.warnings == ["execution requires approval because the plan writes artifacts"]


def test_execute_allows_write_artifacts_when_approved() -> None:
    plan = AstroWorkflowPlan(
        plan_id="local-od-demo",
        title="Local OD demo",
        user_intent="Run local OD workflow",
        steps=[
            WorkflowStep(
                step_id="synth",
                tool=AstroToolName.SYNTH_MEASUREMENTS,
                description="Synthesize measurements.",
                inputs={"backend": "local", "output": "/tmp/astro-assistant/measurements.json"},
                risk=RiskLevel.WRITES_ARTIFACTS,
            )
        ],
    )

    decision = evaluate_plan(plan, dry_run=False, approved=True)

    assert decision == PolicyDecision(allowed=True, warnings=[])


def test_execute_blocks_optional_backend() -> None:
    plan = AstroWorkflowPlan(
        plan_id="optional-backend-demo",
        title="Optional backend demo",
        user_intent="Run optional backend workflow",
        steps=[
            WorkflowStep(
                step_id="estimate",
                tool=AstroToolName.ESTIMATE_MEASUREMENTS,
                description="Estimate measurements with an optional backend.",
                inputs={"backend": "orekit"},
                risk=RiskLevel.OPTIONAL_BACKEND,
            )
        ],
    )

    decision = evaluate_plan(plan, dry_run=False, approved=True)

    assert decision.allowed is False
    assert decision.warnings == [
        "optional backend execution is not enabled in the first assistant slice"
    ]
