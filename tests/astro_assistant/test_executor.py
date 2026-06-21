from collections.abc import Sequence

import pytest

from astro_assistant.executor import WorkflowExecutor
from astro_assistant.models import AstroToolName, AstroWorkflowPlan, RiskLevel, WorkflowStep
from astro_assistant.planner import local_od_demo_plan


class FakeRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[tuple[Sequence[str], str | None]] = []

    def __call__(self, argv: Sequence[str], cwd: str | None) -> tuple[int, str, str]:
        self.calls.append((argv, cwd))
        return self.returncode, "ok", ""


class FailingOnSecondCallRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[str], str | None]] = []

    def __call__(self, argv: Sequence[str], cwd: str | None) -> tuple[int, str, str]:
        self.calls.append((argv, cwd))
        if len(self.calls) == 2:
            return 1, "", "failed"
        return 0, "ok", ""


class UnexpectedRunner:
    def __call__(self, argv: Sequence[str], cwd: str | None) -> tuple[int, str, str]:
        raise AssertionError(f"runner should not have been called with {argv!r} in {cwd!r}")


def _read_only_plan(step_count: int = 1) -> AstroWorkflowPlan:
    return AstroWorkflowPlan(
        plan_id="read-only-demo",
        title="Read-only Demo",
        user_intent="Validate scenario",
        steps=[
            WorkflowStep(
                step_id=f"validate_scenario_{index}",
                tool=AstroToolName.VALIDATE_SCENARIO,
                description="Validate scenario.",
                inputs={"scenario_path": "examples/scenarios/leo_two_station_od.yaml"},
                risk=RiskLevel.READ_ONLY,
            )
            for index in range(step_count)
        ],
    )


def test_dry_run_returns_command_specs_without_results() -> None:
    plan = local_od_demo_plan("Run OD")
    executor = WorkflowExecutor(command_runner=FakeRunner())

    trace = executor.run(plan, dry_run=True, approved=False, cwd="/workspace")

    assert trace.dry_run is True
    assert len(trace.command_specs) == 4
    assert trace.results == []


def test_execute_blocks_without_approval() -> None:
    plan = local_od_demo_plan("Run OD")
    executor = WorkflowExecutor(command_runner=FakeRunner())

    trace = executor.run(plan, dry_run=False, approved=False, cwd="/workspace")

    assert trace.results == []
    assert trace.warnings == ["execution requires approval because the plan writes artifacts"]


def test_approved_execution_records_step_results() -> None:
    plan = _read_only_plan()
    runner = FakeRunner()
    executor = WorkflowExecutor(command_runner=runner)

    trace = executor.run(plan, dry_run=False, approved=True, cwd="/workspace")

    assert len(runner.calls) == 1
    assert runner.calls[0][0] == ["astro", "validate", "examples/scenarios/leo_two_station_od.yaml"]
    assert runner.calls[0][1] == "/workspace"
    assert len(trace.results) == 1
    assert trace.results[0].step_id == "validate_scenario_0"
    assert trace.results[0].returncode == 0
    assert trace.results[0].stdout == "ok"
    assert trace.results[0].stderr == ""
    assert trace.results[0].artifacts == []
    assert trace.results[0].validation_passed is True


def test_execution_stops_after_nonzero_return_code() -> None:
    plan = _read_only_plan(step_count=3)
    runner = FailingOnSecondCallRunner()
    executor = WorkflowExecutor(command_runner=runner)

    trace = executor.run(plan, dry_run=False, approved=True, cwd="/workspace")

    assert len(runner.calls) == 2
    assert [result.returncode for result in trace.results] == [0, 1]
    assert trace.results[-1].step_id == "validate_scenario_1"
    assert trace.results[-1].stderr == "failed"
    assert trace.results[-1].validation_passed is False


@pytest.mark.parametrize(
    ("dry_run", "approved"),
    [
        (True, False),
        (False, False),
    ],
)
def test_command_runner_is_not_called_when_execution_does_not_run(
    dry_run: bool, approved: bool
) -> None:
    plan = local_od_demo_plan("Run OD")
    executor = WorkflowExecutor(command_runner=UnexpectedRunner())

    trace = executor.run(plan, dry_run=dry_run, approved=approved, cwd="/workspace")

    assert trace.results == []
