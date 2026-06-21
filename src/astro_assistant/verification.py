from astro_assistant.models import (
    ArtifactKind,
    AstroWorkflowPlan,
    VerificationDiagnostic,
    VerificationResult,
    WorkflowStep,
)
from astro_assistant.scenarios import (
    ResolvedLocalODScenario,
    resolve_local_od_scenario,
)

LOCAL_OD_STEP_IDS = (
    "validate_scenario",
    "synth_measurements",
    "export_measurements_tdm",
    "estimate_state",
)


def verify_plan(plan: AstroWorkflowPlan) -> VerificationResult:
    if plan.plan_id != "local-od-demo":
        return VerificationResult(passed=True)

    diagnostics: list[VerificationDiagnostic] = []
    try:
        resolved = resolve_local_od_scenario(plan.user_intent)
    except ValueError as exc:
        diagnostics.append(
            _diagnostic("unsupported_scenario", f"could not verify scenario: {exc}")
        )
        return VerificationResult(passed=False, diagnostics=diagnostics)

    _verify_step_order(plan, diagnostics)
    if len(plan.steps) == len(LOCAL_OD_STEP_IDS):
        _verify_local_od_steps(plan.steps, resolved, diagnostics)

    return VerificationResult(passed=not diagnostics, diagnostics=diagnostics)


def _verify_step_order(
    plan: AstroWorkflowPlan, diagnostics: list[VerificationDiagnostic]
) -> None:
    step_ids = tuple(step.step_id for step in plan.steps)
    if step_ids != LOCAL_OD_STEP_IDS:
        diagnostics.append(
            _diagnostic(
                "unexpected_step_order",
                "local OD workflow must validate, synthesize, export, then estimate",
            )
        )


def _verify_local_od_steps(
    steps: list[WorkflowStep],
    resolved: ResolvedLocalODScenario,
    diagnostics: list[VerificationDiagnostic],
) -> None:
    measurements_json = f"{resolved.artifact_dir}/measurements.json"
    measurements_tdm = f"{resolved.artifact_dir}/measurements.tdm"
    estimate_json = f"{resolved.artifact_dir}/estimate.json"

    _expect_input(
        steps[0],
        "scenario_path",
        resolved.path,
        "requested scenario must match the validated scenario",
        diagnostics,
    )
    _expect_output(steps[0], resolved.path, ArtifactKind.SCENARIO, diagnostics)

    _expect_input(
        steps[1],
        "scenario_path",
        resolved.path,
        "requested scenario must match the synthesized measurement scenario",
        diagnostics,
    )
    _expect_input(
        steps[1],
        "backend",
        "local",
        "local OD synthesis backend must be local",
        diagnostics,
    )
    _expect_input(
        steps[1],
        "output",
        measurements_json,
        "measurement JSON output must stay inside the scenario artifact directory",
        diagnostics,
    )
    _expect_output(steps[1], measurements_json, ArtifactKind.MEASUREMENTS_JSON, diagnostics)

    _expect_input(
        steps[2],
        "measurements_path",
        measurements_json,
        "export input must use the generated measurement JSON",
        diagnostics,
    )
    _expect_input(steps[2], "format", "tdm", "measurement export format must be tdm", diagnostics)
    _expect_input(
        steps[2],
        "output",
        measurements_tdm,
        "TDM output must stay inside the scenario artifact directory",
        diagnostics,
    )
    _expect_output(steps[2], measurements_tdm, ArtifactKind.MEASUREMENTS_TDM, diagnostics)

    _expect_input(
        steps[3],
        "scenario_path",
        resolved.path,
        "requested scenario must match the estimation scenario",
        diagnostics,
    )
    _expect_input(
        steps[3],
        "measurements_path",
        measurements_json,
        "estimate input must use the generated measurement JSON",
        diagnostics,
    )
    _expect_input(
        steps[3],
        "backend",
        "local",
        "local OD estimate backend must be local",
        diagnostics,
    )
    _expect_input(
        steps[3],
        "output",
        estimate_json,
        "estimate output must stay inside the scenario artifact directory",
        diagnostics,
    )
    _expect_output(steps[3], estimate_json, ArtifactKind.ESTIMATE_JSON, diagnostics)


def _expect_input(
    step: WorkflowStep,
    key: str,
    expected: str,
    message: str,
    diagnostics: list[VerificationDiagnostic],
) -> None:
    actual = step.inputs.get(key)
    if actual != expected:
        diagnostics.append(
            _diagnostic(
                "unexpected_step_input",
                f"{message}; expected {expected!r}, got {actual!r}",
            )
        )


def _expect_output(
    step: WorkflowStep,
    expected_path: str,
    expected_kind: ArtifactKind,
    diagnostics: list[VerificationDiagnostic],
) -> None:
    if len(step.outputs) != 1:
        diagnostics.append(
            _diagnostic(
                "unexpected_step_output",
                f"{step.step_id} must declare exactly one output artifact",
            )
        )
        return
    artifact = step.outputs[0]
    if artifact.path != expected_path or artifact.kind != expected_kind:
        diagnostics.append(
            _diagnostic(
                "unexpected_step_output",
                (
                    f"{step.step_id} output must be {expected_kind} at {expected_path!r}; "
                    f"got {artifact.kind} at {artifact.path!r}"
                ),
            )
        )


def _diagnostic(code: str, message: str) -> VerificationDiagnostic:
    return VerificationDiagnostic(code=code, message=message)
