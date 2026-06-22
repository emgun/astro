import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from astro_assistant.models import (
    AstroWorkflowPlan,
    CommandSpec,
    StepExecutionResult,
    WorkflowArtifact,
    WorkflowTrace,
)
from astro_assistant.policy import evaluate_plan
from astro_assistant.registry import build_command_spec
from astro_assistant.validators import validate_artifact
from astro_assistant.verification import verify_plan

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


def _artifact_for_validation(
    artifact: WorkflowArtifact, cwd: str | None
) -> WorkflowArtifact:
    artifact_path = Path(artifact.path)
    if cwd is None or artifact_path.is_absolute():
        return artifact
    return artifact.model_copy(update={"path": str(Path(cwd) / artifact_path)})


def _command_write_path(write_path: str, cwd: str | None) -> Path:
    path = Path(write_path)
    if cwd is not None and not path.is_absolute():
        return Path(cwd) / path
    return path


def _prepare_write_paths(command_spec: CommandSpec) -> None:
    for write_path in command_spec.writes:
        _command_write_path(write_path, command_spec.cwd).parent.mkdir(
            parents=True,
            exist_ok=True,
        )


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
        verification = verify_plan(plan)
        policy = evaluate_plan(plan, dry_run=dry_run, approved=approved)
        trace = WorkflowTrace(
            plan=plan,
            dry_run=dry_run,
            command_specs=command_specs,
            verification=verification,
            warnings=policy.warnings,
        )
        if dry_run or not verification.passed or not policy.allowed:
            return trace

        for step, command_spec in zip(plan.steps, command_specs, strict=True):
            _prepare_write_paths(command_spec)
            returncode, stdout, stderr = self._command_runner(
                command_spec.argv, command_spec.cwd
            )
            validation_passed = returncode == 0 and all(
                validate_artifact(_artifact_for_validation(artifact, command_spec.cwd))
                for artifact in step.outputs
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
