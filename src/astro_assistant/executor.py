import subprocess
from collections.abc import Callable, Sequence

from astro_assistant.models import AstroWorkflowPlan, StepExecutionResult, WorkflowTrace
from astro_assistant.policy import evaluate_plan
from astro_assistant.registry import build_command_spec
from astro_assistant.validators import validate_artifact

CommandRunner = Callable[[Sequence[str], str | None], tuple[int, str, str]]


def subprocess_runner(argv: Sequence[str], cwd: str | None) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


class WorkflowExecutor:
    def __init__(self, command_runner: CommandRunner = subprocess_runner) -> None:
        self._command_runner = command_runner

    def run(
        self,
        plan: AstroWorkflowPlan,
        *,
        dry_run: bool,
        approved: bool,
        cwd: str | None,
    ) -> WorkflowTrace:
        command_specs = [build_command_spec(step, cwd=cwd) for step in plan.steps]
        policy = evaluate_plan(plan, dry_run=dry_run, approved=approved)
        trace = WorkflowTrace(
            plan=plan,
            dry_run=dry_run,
            command_specs=command_specs,
            warnings=policy.warnings,
        )
        if dry_run or not policy.allowed:
            return trace

        for step, command_spec in zip(plan.steps, command_specs, strict=True):
            returncode, stdout, stderr = self._command_runner(
                command_spec.argv, command_spec.cwd
            )
            validation_passed = returncode == 0 and all(
                validate_artifact(artifact) for artifact in step.outputs
            )
            trace.results.append(
                StepExecutionResult(
                    step_id=step.step_id,
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                    artifacts=step.outputs,
                    validation_passed=validation_passed,
                )
            )
            if returncode != 0 or not validation_passed:
                break
        return trace
