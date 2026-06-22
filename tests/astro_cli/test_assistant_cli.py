import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from astro_assistant.models import (
    AstroWorkflowPlan,
    VerificationDiagnostic,
    VerificationResult,
    WorkflowTrace,
)
from astro_assistant.policy import evaluate_plan
from astro_assistant.registry import build_command_spec
from astro_cli.main import app

runner = CliRunner(mix_stderr=False)


@dataclass(frozen=True)
class ExecutorCall:
    dry_run: bool
    approved: bool
    cwd: str | None


@pytest.fixture(autouse=True)
def fake_executor(monkeypatch: pytest.MonkeyPatch) -> list[ExecutorCall]:
    calls: list[ExecutorCall] = []

    class FakeWorkflowExecutor:
        def run(
            self,
            plan: AstroWorkflowPlan,
            *,
            dry_run: bool,
            approved: bool,
            cwd: str | None,
        ) -> WorkflowTrace:
            calls.append(ExecutorCall(dry_run=dry_run, approved=approved, cwd=cwd))
            policy = evaluate_plan(plan, dry_run=dry_run, approved=approved)
            return WorkflowTrace(
                plan=plan,
                dry_run=dry_run,
                command_specs=[build_command_spec(step, cwd=cwd) for step in plan.steps],
                warnings=policy.warnings,
            )

    monkeypatch.setattr("astro_cli.main.WorkflowExecutor", FakeWorkflowExecutor)
    return calls


def test_assistant_ask_dry_run_prints_plan() -> None:
    result = runner.invoke(app, ["ask", "Run the local OD demo", "--dry-run"])

    assert result.exit_code == 0
    assert "local-od-demo" in result.stdout
    assert "estimate-measurements" in result.stdout


def test_assistant_ask_dry_run_prints_requested_scenario_plan() -> None:
    result = runner.invoke(
        app,
        [
            "ask",
            "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "examples/scenarios/leo_two_station_angles.yaml" in result.stdout
    assert "/tmp/astro-assistant/leo_two_station_angles/measurements.json" in result.stdout


def test_assistant_ask_requires_approval_for_execution() -> None:
    result = runner.invoke(app, ["ask", "Run the local OD demo", "--execute"])

    assert result.exit_code == 2
    assert (
        "execution requires approval" in result.stdout
        or "execution requires approval" in result.stderr
    )


def test_assistant_ask_execute_wins_over_dry_run_flag() -> None:
    result = runner.invoke(
        app,
        ["ask", "Run the local OD demo", "--dry-run", "--execute"],
    )

    assert result.exit_code == 2
    assert (
        "execution requires approval" in result.stdout
        or "execution requires approval" in result.stderr
    )


def test_assistant_ask_passes_approved_execution_to_executor(
    fake_executor: list[ExecutorCall],
) -> None:
    result = runner.invoke(app, ["ask", "Run the local OD demo", "--execute", "--approved"])

    assert result.exit_code == 0
    assert fake_executor == [ExecutorCall(dry_run=False, approved=True, cwd=str(Path.cwd()))]


def test_assistant_ask_writes_trace_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"

    result = runner.invoke(
        app,
        [
            "ask",
            "Run the local OD demo",
            "--dry-run",
            "--trace-output",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    assert trace_path.exists()
    assert json.loads(trace_path.read_text(encoding="utf-8")) == json.loads(result.stdout)


def test_assistant_ask_creates_trace_output_parent_directories(tmp_path: Path) -> None:
    trace_path = tmp_path / "nested" / "trace.json"

    result = runner.invoke(
        app,
        [
            "ask",
            "Run the local OD demo",
            "--dry-run",
            "--trace-output",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    assert trace_path.exists()


def test_assistant_ask_exits_when_executor_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingVerificationExecutor:
        def run(
            self,
            plan: AstroWorkflowPlan,
            *,
            dry_run: bool,
            approved: bool,
            cwd: str | None,
        ) -> WorkflowTrace:
            return WorkflowTrace(
                plan=plan,
                dry_run=dry_run,
                command_specs=[],
                verification=VerificationResult(
                    passed=False,
                    diagnostics=[
                        VerificationDiagnostic(
                            code="bad_plan",
                            message="verification rejected the plan",
                        )
                    ],
                ),
            )

    monkeypatch.setattr("astro_cli.main.WorkflowExecutor", FailingVerificationExecutor)

    result = runner.invoke(app, ["ask", "Run the local OD demo", "--dry-run"])

    assert result.exit_code == 2
    assert "verification rejected the plan" in result.stderr


def test_assistant_ask_unsupported_prompt_exits_with_planner_error() -> None:
    result = runner.invoke(app, ["ask", "Tune a launch vehicle", "--dry-run"])

    assert result.exit_code == 2
    assert "deterministic planner currently supports the local OD demo only" in result.stderr


def test_assistant_ask_unsupported_scenario_exits_with_resolver_error() -> None:
    result = runner.invoke(
        app,
        [
            "ask",
            "Run local orbit determination on examples/scenarios/leo_orekit_high_fidelity.yaml",
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "could not resolve a supported local OD scenario" in result.stderr


def test_verify_assistant_reports_supported_scenario() -> None:
    result = runner.invoke(
        app,
        ["verify-assistant", "Run local OD on leo_two_station_topocentric.yaml"],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["supported"] is True
    assert report["plan_id"] == "local-od-demo"
    assert report["scenario_path"] == "examples/scenarios/leo_two_station_topocentric.yaml"
    assert report["artifact_dir"] == "/tmp/astro-assistant/leo_two_station_topocentric"
    assert report["verification"]["passed"] is True
    assert report["verification"]["diagnostics"] == []


def test_verify_assistant_reports_unsupported_scenario() -> None:
    result = runner.invoke(
        app,
        [
            "verify-assistant",
            "Run local orbit determination on examples/scenarios/leo_orekit_high_fidelity.yaml",
        ],
    )

    assert result.exit_code == 2
    report = json.loads(result.stdout)
    assert report["supported"] is False
    assert report["scenario_path"] is None
    assert report["verification"]["passed"] is False
    assert "could not resolve a supported local OD scenario" in (
        report["verification"]["diagnostics"][0]["message"]
    )
