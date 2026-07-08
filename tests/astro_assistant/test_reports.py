import json
from pathlib import Path

from astro_assistant.models import (
    ArtifactKind,
    StepExecutionResult,
    VerificationResult,
    WorkflowArtifact,
    WorkflowTrace,
)
from astro_assistant.planner import local_od_demo_plan
from astro_assistant.registry import build_command_spec
from astro_assistant.reports import build_workflow_report, format_workflow_report


def _trace_with_local_od_artifacts(tmp_path: Path) -> WorkflowTrace:
    artifact_dir = tmp_path / "astro-assistant" / "leo_two_station_angles"
    artifact_dir.mkdir(parents=True)
    measurements_path = artifact_dir / "measurements.json"
    tdm_path = artifact_dir / "measurements.tdm"
    estimate_path = artifact_dir / "estimate.json"

    measurements_path.write_text(
        json.dumps({"measurements": [{"value": 1.0}, {"value": 2.0}]}),
        encoding="utf-8",
    )
    tdm_path.write_text("CCSDS_TDM_VERS = 2.0\nMETA_START\nMETA_STOP\n", encoding="utf-8")
    estimate_path.write_text(
        json.dumps(
            {
                "converged": True,
                "iterations": 5,
                "rms": 0.7045,
                "metadata": {
                    "jacobian_rank": 6,
                    "residual_count": 44,
                    "max_abs_residual": 1.25,
                },
            }
        ),
        encoding="utf-8",
    )

    plan = local_od_demo_plan(
        "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml"
    )
    remapped_steps = []
    for step in plan.steps:
        inputs = dict(step.inputs)
        outputs = []
        for artifact in step.outputs:
            path = {
                "/tmp/astro-assistant/leo_two_station_angles/measurements.json": str(
                    measurements_path
                ),
                "/tmp/astro-assistant/leo_two_station_angles/measurements.tdm": str(tdm_path),
                "/tmp/astro-assistant/leo_two_station_angles/estimate.json": str(estimate_path),
            }.get(artifact.path, artifact.path)
            outputs.append(artifact.model_copy(update={"path": path}))
        if inputs.get("output") == "/tmp/astro-assistant/leo_two_station_angles/measurements.json":
            inputs["output"] = str(measurements_path)
        if inputs.get("output") == "/tmp/astro-assistant/leo_two_station_angles/measurements.tdm":
            inputs["output"] = str(tdm_path)
        if inputs.get("measurements_path") == (
            "/tmp/astro-assistant/leo_two_station_angles/measurements.json"
        ):
            inputs["measurements_path"] = str(measurements_path)
        if inputs.get("output") == "/tmp/astro-assistant/leo_two_station_angles/estimate.json":
            inputs["output"] = str(estimate_path)
        remapped_steps.append(step.model_copy(update={"inputs": inputs, "outputs": outputs}))
    plan = plan.model_copy(update={"steps": remapped_steps})

    return WorkflowTrace(
        plan=plan,
        dry_run=False,
        command_specs=[build_command_spec(step, cwd=str(tmp_path)) for step in plan.steps],
        verification=VerificationResult(passed=True),
        results=[
            StepExecutionResult(
                step_id=step.step_id,
                returncode=0,
                artifacts=step.outputs,
                validation_passed=True,
            )
            for step in plan.steps
        ],
    )


def test_build_workflow_report_summarizes_local_od_artifacts(tmp_path: Path) -> None:
    trace = _trace_with_local_od_artifacts(tmp_path)

    report = build_workflow_report(trace)

    assert report.workflow_id == "local-od-demo"
    assert report.scenario_path == "examples/scenarios/leo_two_station_angles.yaml"
    assert report.verification_passed is True
    assert report.executed_step_count == 4
    assert report.measurement_count == 2
    assert report.tdm_line_count == 3
    assert report.estimate_converged is True
    assert report.estimate_iterations == 5
    assert report.estimate_rms == 0.7045
    assert report.jacobian_rank == 6
    assert report.residual_count == 44
    assert report.max_abs_residual == 1.25
    assert [
        (artifact.kind, artifact.validation_passed) for artifact in report.artifacts
    ] == [
        (ArtifactKind.SCENARIO, True),
        (ArtifactKind.MEASUREMENTS_JSON, True),
        (ArtifactKind.MEASUREMENTS_TDM, True),
        (ArtifactKind.ESTIMATE_JSON, True),
    ]


def test_format_workflow_report_renders_concise_human_view(tmp_path: Path) -> None:
    report = build_workflow_report(_trace_with_local_od_artifacts(tmp_path))

    assert format_workflow_report(report).splitlines() == [
        "Workflow: local-od-demo",
        "Title: Local Orbit Determination Demo: leo-two-station-angles",
        "Scenario: examples/scenarios/leo_two_station_angles.yaml",
        "Mode: executed",
        "Verification: passed",
        "Steps: 4/4 executed",
        "Artifacts: 4 declared, 4 validated",
        "Measurements: 2",
        "TDM lines: 3",
        "Estimate: converged, iterations=5, rms=0.7045",
        "Jacobian rank: 6",
        "Residuals: count=44, max_abs=1.25",
        "Warnings: none",
    ]


def test_build_workflow_report_marks_missing_artifact_metrics_unavailable() -> None:
    plan = local_od_demo_plan("Run the local OD demo")
    missing_artifact = WorkflowArtifact(
        path="/tmp/astro-assistant/leo_two_station_od/missing.json",
        kind=ArtifactKind.MEASUREMENTS_JSON,
    )
    step = plan.steps[1].model_copy(update={"outputs": [missing_artifact]})
    plan = plan.model_copy(update={"steps": [plan.steps[0], step, *plan.steps[2:]]})
    trace = WorkflowTrace(
        plan=plan,
        dry_run=False,
        command_specs=[build_command_spec(step, cwd=None) for step in plan.steps],
        results=[
            StepExecutionResult(
                step_id=step.step_id,
                returncode=0,
                artifacts=step.outputs,
                validation_passed=True,
            )
            for step in plan.steps
        ],
    )

    report = build_workflow_report(trace)

    assert report.measurement_count is None
    assert any("missing artifact" in warning for warning in report.warnings)


def test_build_workflow_report_does_not_read_artifacts_for_dry_run(tmp_path: Path) -> None:
    measurements_path = tmp_path / "measurements.json"
    measurements_path.write_text(json.dumps({"measurements": [{"value": 1.0}]}), encoding="utf-8")
    plan = local_od_demo_plan("Run the local OD demo")
    step = plan.steps[1].model_copy(
        update={
            "outputs": [
                WorkflowArtifact(
                    path=str(measurements_path),
                    kind=ArtifactKind.MEASUREMENTS_JSON,
                )
            ]
        }
    )
    plan = plan.model_copy(update={"steps": [plan.steps[0], step, *plan.steps[2:]]})
    trace = WorkflowTrace(
        plan=plan,
        dry_run=True,
        command_specs=[build_command_spec(step, cwd=None) for step in plan.steps],
    )

    report = build_workflow_report(trace)

    assert report.measurement_count is None
    assert report.executed_step_count == 0
