import os
from pathlib import Path

from astro_assistant.models import (
    ArtifactKind,
    AstroToolName,
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
    lifecycle_tools = {
        AstroToolName.VERIFY_MISSION_LIFECYCLE_RESULT,
        AstroToolName.REVIEW_MISSION_LIFECYCLE,
    }
    if plan.plan_id != "mission-lifecycle-review" and any(
        step.tool in lifecycle_tools for step in plan.steps
    ):
        return VerificationResult(
            passed=False,
            diagnostics=[
                _diagnostic(
                    "protected_plan_id",
                    "lifecycle review tools require the fixed mission-lifecycle-review plan",
                )
            ],
        )
    if plan.plan_id == "mission-lifecycle-review":
        return _verify_mission_lifecycle_review_plan(plan)
    if plan.plan_id == "paired-assurance-review-comparison":
        return _verify_assurance_review_comparison_plan(plan)
    if plan.plan_id == "paired-assurance-review":
        return _verify_assurance_review_plan(plan)
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


def _verify_mission_lifecycle_review_plan(plan: AstroWorkflowPlan) -> VerificationResult:
    diagnostics: list[VerificationDiagnostic] = []
    if len(plan.steps) != 2:
        diagnostics.append(
            _diagnostic("unexpected_step_order", "lifecycle review requires two steps")
        )
        return VerificationResult(passed=False, diagnostics=diagnostics)
    verify_step, review_step = plan.steps
    if (
        verify_step.tool is not AstroToolName.VERIFY_MISSION_LIFECYCLE_RESULT
        or review_step.tool is not AstroToolName.REVIEW_MISSION_LIFECYCLE
    ):
        diagnostics.append(
            _diagnostic(
                "unexpected_step_order",
                "lifecycle review must verify evidence before writing review",
            )
        )
    for key in ("result_path", "scenario_path"):
        if verify_step.inputs.get(key) != review_step.inputs.get(key):
            diagnostics.append(
                _diagnostic("source_discontinuity", f"lifecycle {key} must remain fixed")
            )
    paths = [
        review_step.inputs.get("result_path"),
        review_step.inputs.get("scenario_path"),
        review_step.inputs.get("output"),
    ]
    summary_output = review_step.inputs.get("summary_output")
    if summary_output is not None:
        paths.append(summary_output)
    if not _paths_are_distinct(paths):
        diagnostics.append(
            _diagnostic("path_collision", "lifecycle review paths must all differ")
        )
    if (
        len(review_step.outputs) != 1
        or review_step.outputs[0].kind is not ArtifactKind.MISSION_LIFECYCLE_REVIEW
    ):
        diagnostics.append(
            _diagnostic(
                "unexpected_step_output",
                "lifecycle review must declare one lifecycle review artifact",
            )
        )
    elif review_step.inputs.get("output") != review_step.outputs[0].path:
        diagnostics.append(
            _diagnostic("output_discontinuity", "lifecycle review output must remain fixed")
        )
    return VerificationResult(passed=not diagnostics, diagnostics=diagnostics)


def _verify_assurance_review_comparison_plan(
    plan: AstroWorkflowPlan,
) -> VerificationResult:
    diagnostics: list[VerificationDiagnostic] = []
    if len(plan.steps) != 1 or plan.steps[0].tool is not AstroToolName.COMPARE_ASSURANCE_REVIEWS:
        diagnostics.append(
            _diagnostic("unexpected_step_order", "assurance comparison requires one compare step")
        )
        return VerificationResult(passed=False, diagnostics=diagnostics)
    step = plan.steps[0]
    comparison_paths = [
        step.inputs.get("baseline_path"),
        step.inputs.get("candidate_path"),
        step.inputs.get("output"),
    ]
    summary_output = step.inputs.get("summary_output")
    if summary_output is not None:
        comparison_paths.append(summary_output)
    if not _paths_are_distinct(comparison_paths):
        diagnostics.append(
            _diagnostic("path_collision", "assurance comparison paths must all differ")
        )
    if (
        len(step.outputs) != 1
        or step.outputs[0].kind is not ArtifactKind.ASSURANCE_REVIEW_COMPARISON
    ):
        diagnostics.append(
            _diagnostic("unexpected_step_output", "comparison must declare one comparison artifact")
        )
    elif step.inputs.get("output") != step.outputs[0].path:
        diagnostics.append(
            _diagnostic("output_discontinuity", "comparison output must match command input")
        )
    return VerificationResult(passed=not diagnostics, diagnostics=diagnostics)


def _paths_are_distinct(values: list[object]) -> bool:
    if not all(isinstance(value, str) and value for value in values):
        return False
    paths = [Path(value).resolve() for value in values if isinstance(value, str)]
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == other:
                return False
            if path.exists() and other.exists() and os.path.samefile(path, other):
                return False
    return True


def _verify_assurance_review_plan(plan: AstroWorkflowPlan) -> VerificationResult:
    diagnostics: list[VerificationDiagnostic] = []
    if len(plan.steps) != 2:
        diagnostics.append(
            _diagnostic("unexpected_step_order", "assurance review requires two steps")
        )
        return VerificationResult(passed=False, diagnostics=diagnostics)
    verify_step, review_step = plan.steps
    if (
        verify_step.tool is not AstroToolName.VERIFY_ASSURANCE_VALIDATION
        or review_step.tool is not AstroToolName.REVIEW_ASSURANCE_VALIDATION
    ):
        diagnostics.append(
            _diagnostic("unexpected_step_order", "assurance review must verify before review")
        )
    if verify_step.inputs.get("result_path") != review_step.inputs.get("result_path"):
        diagnostics.append(
            _diagnostic("source_discontinuity", "verify and review must use the same result path")
        )
    if (
        len(review_step.outputs) != 1
        or review_step.outputs[0].kind is not ArtifactKind.ASSURANCE_REVIEW
    ):
        diagnostics.append(
            _diagnostic(
                "unexpected_step_output",
                "assurance review must declare one review artifact",
            )
        )
    elif review_step.inputs.get("output") != review_step.outputs[0].path:
        diagnostics.append(
            _diagnostic("output_discontinuity", "declared review output must match command input")
        )
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
