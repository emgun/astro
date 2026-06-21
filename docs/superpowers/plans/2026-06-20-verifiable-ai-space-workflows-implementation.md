# Verifiable AI Space Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verifiable AI-assisted workflow layer that compiles natural-language mission-analysis intent into typed, reviewable, deterministic Astro workflow plans with audit traces.

**Architecture:** Add a small `astro_assistant` package that owns plan schemas, policy checks, command-spec generation, execution traces, and deterministic planner adapters. The first workflow pack is a local orbit-determination workflow over existing Astro CLI/product boundaries; MCP remains an interface over the same registry after the CLI path is proven. The LLM/planner proposes plans only; Astro commands compute and validators decide whether artifacts are valid.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, pytest, existing Astro CLI commands, JSON audit traces, deterministic fake planner for tests.

---

## Context

Branch target: create from `main`.

Research artifact: `docs/research/2026-06-20-verifiable-ai-space-workflows.md`

North star:

```text
natural language intent
-> typed workflow plan
-> allow-listed Astro command specs
-> dry-run review
-> deterministic execution
-> artifact validation
-> replayable audit trace
```

First public slice:

```text
astro ask "run the local OD demo"
-> validate scenario
-> synthesize measurements
-> export TDM
-> estimate initial state
-> validate output files
-> write trace JSON
```

Non-goals for this branch:

- No arbitrary shell execution.
- No autonomous maneuver planning.
- No operational flight-readiness claims.
- No required real LLM provider or API key in default tests.
- No MCP runtime server until the shared registry and CLI trace are green.

## File Map

- Create `src/astro_assistant/__init__.py`: public exports for plan, policy, registry, and execution helpers.
- Create `src/astro_assistant/models.py`: Pydantic plan, step, artifact, policy, and trace models.
- Create `src/astro_assistant/registry.py`: allow-listed Astro tool definitions and command-spec builders.
- Create `src/astro_assistant/policy.py`: risk classification and execution gating.
- Create `src/astro_assistant/planner.py`: deterministic planner interface and golden prompt mapping.
- Create `src/astro_assistant/executor.py`: dry-run and execute workflow plans through a command runner.
- Create `src/astro_assistant/validators.py`: artifact validators for file existence, JSON parseability, and estimate metadata.
- Modify `src/astro_cli/main.py`: add the `astro ask` command.
- Modify `pyproject.toml`: include `src/astro_assistant` in the wheel packages.
- Create `tests/astro_assistant/test_models.py`: plan schema tests.
- Create `tests/astro_assistant/test_registry.py`: command-spec and allow-list tests.
- Create `tests/astro_assistant/test_policy.py`: approval/risk tests.
- Create `tests/astro_assistant/test_planner.py`: deterministic prompt-to-plan tests.
- Create `tests/astro_assistant/test_executor.py`: dry-run, trace, and fake-runner tests.
- Create `tests/astro_cli/test_assistant_cli.py`: CLI smoke tests with deterministic planner and dry-run output.
- Create `examples/assistant/od_workflow_prompt.txt`: public demo prompt.
- Create `docs/assistant-workflows.md`: public usage and safety-boundary documentation.
- Modify `docs/validation-matrix.md`: add required local gates for assistant schema/CLI dry-run tests.

## Plan Schema

The implementation should keep the model small. Use discriminated values and explicit enums so plan
files are stable under tests.

```python
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AstroToolName(StrEnum):
    VALIDATE_SCENARIO = "validate_scenario"
    SYNTH_MEASUREMENTS = "synth_measurements"
    EXPORT_MEASUREMENTS = "export_measurements"
    ESTIMATE_MEASUREMENTS = "estimate_measurements"


class RiskLevel(StrEnum):
    READ_ONLY = "read_only"
    WRITES_ARTIFACTS = "writes_artifacts"
    OPTIONAL_BACKEND = "optional_backend"


class ArtifactKind(StrEnum):
    SCENARIO = "scenario"
    MEASUREMENTS_JSON = "measurements_json"
    MEASUREMENTS_TDM = "measurements_tdm"
    ESTIMATE_JSON = "estimate_json"
    TRACE_JSON = "trace_json"


class WorkflowArtifact(BaseModel):
    path: str
    kind: ArtifactKind
    required: bool = True


class WorkflowStep(BaseModel):
    step_id: str = Field(pattern=r"^[a-z0-9_]+$")
    tool: AstroToolName
    description: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[WorkflowArtifact] = Field(default_factory=list)
    risk: RiskLevel


class AstroWorkflowPlan(BaseModel):
    plan_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    title: str
    user_intent: str
    requires_approval: bool = True
    steps: list[WorkflowStep]


class CommandSpec(BaseModel):
    step_id: str
    argv: list[str]
    cwd: str | None = None
    writes: list[str] = Field(default_factory=list)


class StepExecutionResult(BaseModel):
    step_id: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    artifacts: list[WorkflowArtifact] = Field(default_factory=list)
    validation_passed: bool


class WorkflowTrace(BaseModel):
    plan: AstroWorkflowPlan
    dry_run: bool
    command_specs: list[CommandSpec]
    results: list[StepExecutionResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

## Task 1: Package Scaffold

**Files:**
- Create: `src/astro_assistant/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/astro_assistant/test_imports.py`

- [ ] **Step 1: Write failing import test**

```python
def test_astro_assistant_imports() -> None:
    import astro_assistant

    assert astro_assistant.__all__ == [
        "AstroWorkflowPlan",
        "WorkflowStep",
        "WorkflowTrace",
    ]
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/astro_assistant/test_imports.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'astro_assistant'`.

- [ ] **Step 3: Create package export**

Create `src/astro_assistant/__init__.py`:

```python
"""Verifiable AI-assisted workflow planning for Astro Suite."""

from astro_assistant.models import AstroWorkflowPlan, WorkflowStep, WorkflowTrace

__all__ = [
    "AstroWorkflowPlan",
    "WorkflowStep",
    "WorkflowTrace",
]
```

- [ ] **Step 4: Add package to wheel**

Modify `pyproject.toml` package list:

```toml
packages = [
  "src/astro_core",
  "src/astro_dynamics",
  "src/astro_launch",
  "src/astro_od",
  "src/astro_backends",
  "src/astro_cli",
  "src/astro_assistant",
]
```

- [ ] **Step 5: Run test**

Run: `python -m pytest tests/astro_assistant/test_imports.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/astro_assistant/__init__.py tests/astro_assistant/test_imports.py
git commit -m "feat: add assistant package scaffold"
```

## Task 2: Workflow Models

**Files:**
- Create: `src/astro_assistant/models.py`
- Test: `tests/astro_assistant/test_models.py`

- [ ] **Step 1: Write model tests**

```python
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
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/astro_assistant/test_models.py -q`

Expected: FAIL because `astro_assistant.models` does not exist.

- [ ] **Step 3: Implement models**

Create `src/astro_assistant/models.py` using the exact schema from the "Plan Schema" section.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_assistant/test_models.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_assistant/models.py tests/astro_assistant/test_models.py
git commit -m "feat: define assistant workflow models"
```

## Task 3: Tool Registry

**Files:**
- Create: `src/astro_assistant/registry.py`
- Test: `tests/astro_assistant/test_registry.py`

- [ ] **Step 1: Write registry tests**

```python
from astro_assistant.models import AstroToolName, WorkflowStep
from astro_assistant.registry import build_command_spec


def test_validate_scenario_command_spec() -> None:
    step = WorkflowStep(
        step_id="validate_scenario",
        tool=AstroToolName.VALIDATE_SCENARIO,
        description="Validate scenario.",
        inputs={"scenario_path": "examples/scenarios/leo_two_station_od.yaml"},
        risk="read_only",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "validate",
        "examples/scenarios/leo_two_station_od.yaml",
    ]
    assert command.cwd == "/workspace"
    assert command.writes == []


def test_estimate_measurements_command_spec_records_output_write() -> None:
    step = WorkflowStep(
        step_id="estimate",
        tool=AstroToolName.ESTIMATE_MEASUREMENTS,
        description="Estimate initial state.",
        inputs={
            "scenario_path": "examples/scenarios/leo_two_station_od.yaml",
            "measurements_path": "/tmp/astro-assistant/measurements.json",
            "backend": "local",
            "output": "/tmp/astro-assistant/estimate.json",
        },
        risk="writes_artifacts",
    )

    command = build_command_spec(step, cwd="/workspace")

    assert command.argv == [
        "astro",
        "estimate-measurements",
        "examples/scenarios/leo_two_station_od.yaml",
        "/tmp/astro-assistant/measurements.json",
        "--backend",
        "local",
        "--output",
        "/tmp/astro-assistant/estimate.json",
    ]
    assert command.writes == ["/tmp/astro-assistant/estimate.json"]
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/astro_assistant/test_registry.py -q`

Expected: FAIL because `astro_assistant.registry` does not exist.

- [ ] **Step 3: Implement command-spec builders**

Create `src/astro_assistant/registry.py`:

```python
from collections.abc import Callable

from astro_assistant.models import AstroToolName, CommandSpec, WorkflowStep


def _required_str(step: WorkflowStep, key: str) -> str:
    value = step.inputs.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{step.step_id} requires string input {key!r}")
    return value


def _optional_str(step: WorkflowStep, key: str, default: str) -> str:
    value = step.inputs.get(key, default)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{step.step_id} requires string input {key!r}")
    return value


def _validate_scenario(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    scenario_path = _required_str(step, "scenario_path")
    return CommandSpec(
        step_id=step.step_id,
        argv=["astro", "validate", scenario_path],
        cwd=cwd,
    )


def _synth_measurements(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    scenario_path = _required_str(step, "scenario_path")
    output = _required_str(step, "output")
    backend = _optional_str(step, "backend", "local")
    return CommandSpec(
        step_id=step.step_id,
        argv=[
            "astro",
            "synth-measurements",
            scenario_path,
            "--backend",
            backend,
            "--output",
            output,
        ],
        cwd=cwd,
        writes=[output],
    )


def _export_measurements(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    measurements_path = _required_str(step, "measurements_path")
    output = _required_str(step, "output")
    measurement_format = _optional_str(step, "format", "tdm")
    return CommandSpec(
        step_id=step.step_id,
        argv=[
            "astro",
            "export-measurements",
            measurements_path,
            "--format",
            measurement_format,
            "--output",
            output,
        ],
        cwd=cwd,
        writes=[output],
    )


def _estimate_measurements(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    scenario_path = _required_str(step, "scenario_path")
    measurements_path = _required_str(step, "measurements_path")
    output = _required_str(step, "output")
    backend = _optional_str(step, "backend", "local")
    return CommandSpec(
        step_id=step.step_id,
        argv=[
            "astro",
            "estimate-measurements",
            scenario_path,
            measurements_path,
            "--backend",
            backend,
            "--output",
            output,
        ],
        cwd=cwd,
        writes=[output],
    )


_BUILDERS: dict[AstroToolName, Callable[[WorkflowStep, str | None], CommandSpec]] = {
    AstroToolName.VALIDATE_SCENARIO: _validate_scenario,
    AstroToolName.SYNTH_MEASUREMENTS: _synth_measurements,
    AstroToolName.EXPORT_MEASUREMENTS: _export_measurements,
    AstroToolName.ESTIMATE_MEASUREMENTS: _estimate_measurements,
}


def build_command_spec(step: WorkflowStep, cwd: str | None = None) -> CommandSpec:
    return _BUILDERS[step.tool](step, cwd)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_assistant/test_registry.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_assistant/registry.py tests/astro_assistant/test_registry.py
git commit -m "feat: add assistant command registry"
```

## Task 4: Policy Gates

**Files:**
- Create: `src/astro_assistant/policy.py`
- Test: `tests/astro_assistant/test_policy.py`

- [ ] **Step 1: Write policy tests**

```python
import pytest

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
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/astro_assistant/test_policy.py -q`

Expected: FAIL because `astro_assistant.policy` does not exist.

- [ ] **Step 3: Implement policy evaluator**

Create `src/astro_assistant/policy.py`:

```python
from pydantic import BaseModel, Field

from astro_assistant.models import AstroWorkflowPlan, RiskLevel


class PolicyDecision(BaseModel):
    allowed: bool
    warnings: list[str] = Field(default_factory=list)


def evaluate_plan(plan: AstroWorkflowPlan, *, dry_run: bool, approved: bool) -> PolicyDecision:
    warnings: list[str] = []
    risks = {step.risk for step in plan.steps}

    if dry_run:
        return PolicyDecision(allowed=True)

    if RiskLevel.OPTIONAL_BACKEND in risks:
        warnings.append("optional backend execution is not enabled in the first assistant slice")

    if plan.requires_approval and RiskLevel.WRITES_ARTIFACTS in risks and not approved:
        warnings.append("execution requires approval because the plan writes artifacts")

    return PolicyDecision(allowed=len(warnings) == 0, warnings=warnings)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_assistant/test_policy.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_assistant/policy.py tests/astro_assistant/test_policy.py
git commit -m "feat: add assistant execution policy"
```

## Task 5: Deterministic Planner

**Files:**
- Create: `src/astro_assistant/planner.py`
- Create: `examples/assistant/od_workflow_prompt.txt`
- Test: `tests/astro_assistant/test_planner.py`

- [ ] **Step 1: Write planner tests**

```python
from astro_assistant.models import AstroToolName
from astro_assistant.planner import DeterministicPlanner


def test_deterministic_planner_builds_local_od_workflow() -> None:
    planner = DeterministicPlanner()

    plan = planner.plan("Run the local OD demo and export TDM")

    assert plan.plan_id == "local-od-demo"
    assert [step.tool for step in plan.steps] == [
        AstroToolName.VALIDATE_SCENARIO,
        AstroToolName.SYNTH_MEASUREMENTS,
        AstroToolName.EXPORT_MEASUREMENTS,
        AstroToolName.ESTIMATE_MEASUREMENTS,
    ]
    assert plan.steps[-1].inputs["output"] == "/tmp/astro-assistant/estimate.json"
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/astro_assistant/test_planner.py -q`

Expected: FAIL because `astro_assistant.planner` does not exist.

- [ ] **Step 3: Implement deterministic planner**

Create `src/astro_assistant/planner.py`:

```python
from astro_assistant.models import (
    ArtifactKind,
    AstroToolName,
    AstroWorkflowPlan,
    RiskLevel,
    WorkflowArtifact,
    WorkflowStep,
)


class DeterministicPlanner:
    def plan(self, prompt: str) -> AstroWorkflowPlan:
        normalized = prompt.lower()
        if "od" not in normalized and "orbit determination" not in normalized:
            raise ValueError("deterministic planner currently supports the local OD demo only")
        return local_od_demo_plan(prompt)


def local_od_demo_plan(user_intent: str) -> AstroWorkflowPlan:
    scenario_path = "examples/scenarios/leo_two_station_od.yaml"
    measurements_json = "/tmp/astro-assistant/measurements.json"
    measurements_tdm = "/tmp/astro-assistant/measurements.tdm"
    estimate_json = "/tmp/astro-assistant/estimate.json"

    return AstroWorkflowPlan(
        plan_id="local-od-demo",
        title="Local Orbit Determination Demo",
        user_intent=user_intent,
        steps=[
            WorkflowStep(
                step_id="validate_scenario",
                tool=AstroToolName.VALIDATE_SCENARIO,
                description="Validate the checked-in two-station OD scenario.",
                inputs={"scenario_path": scenario_path},
                outputs=[WorkflowArtifact(path=scenario_path, kind=ArtifactKind.SCENARIO)],
                risk=RiskLevel.READ_ONLY,
            ),
            WorkflowStep(
                step_id="synth_measurements",
                tool=AstroToolName.SYNTH_MEASUREMENTS,
                description="Generate local synthetic range/range-rate measurements.",
                inputs={"scenario_path": scenario_path, "backend": "local", "output": measurements_json},
                outputs=[
                    WorkflowArtifact(path=measurements_json, kind=ArtifactKind.MEASUREMENTS_JSON)
                ],
                risk=RiskLevel.WRITES_ARTIFACTS,
            ),
            WorkflowStep(
                step_id="export_measurements_tdm",
                tool=AstroToolName.EXPORT_MEASUREMENTS,
                description="Export generated measurements to CCSDS TDM-style KVN.",
                inputs={"measurements_path": measurements_json, "format": "tdm", "output": measurements_tdm},
                outputs=[WorkflowArtifact(path=measurements_tdm, kind=ArtifactKind.MEASUREMENTS_TDM)],
                risk=RiskLevel.WRITES_ARTIFACTS,
            ),
            WorkflowStep(
                step_id="estimate_state",
                tool=AstroToolName.ESTIMATE_MEASUREMENTS,
                description="Estimate the initial state from the generated measurements.",
                inputs={
                    "scenario_path": scenario_path,
                    "measurements_path": measurements_json,
                    "backend": "local",
                    "output": estimate_json,
                },
                outputs=[WorkflowArtifact(path=estimate_json, kind=ArtifactKind.ESTIMATE_JSON)],
                risk=RiskLevel.WRITES_ARTIFACTS,
            ),
        ],
    )
```

- [ ] **Step 4: Add public prompt example**

Create `examples/assistant/od_workflow_prompt.txt`:

```text
Run the local orbit-determination demo, export the generated measurements as TDM, estimate the initial state, and write a trace.
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/astro_assistant/test_planner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/astro_assistant/planner.py tests/astro_assistant/test_planner.py examples/assistant/od_workflow_prompt.txt
git commit -m "feat: add deterministic od workflow planner"
```

## Task 6: Artifact Validators

**Files:**
- Create: `src/astro_assistant/validators.py`
- Test: `tests/astro_assistant/test_validators.py`

- [ ] **Step 1: Write validator tests**

```python
import json

from astro_assistant.models import ArtifactKind, WorkflowArtifact
from astro_assistant.validators import validate_artifact


def test_validate_json_artifact_requires_parseable_json(tmp_path) -> None:
    path = tmp_path / "estimate.json"
    path.write_text(json.dumps({"converged": True}), encoding="utf-8")

    artifact = WorkflowArtifact(path=str(path), kind=ArtifactKind.ESTIMATE_JSON)

    assert validate_artifact(artifact) is True


def test_validate_required_artifact_fails_when_missing(tmp_path) -> None:
    artifact = WorkflowArtifact(path=str(tmp_path / "missing.json"), kind=ArtifactKind.ESTIMATE_JSON)

    assert validate_artifact(artifact) is False
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/astro_assistant/test_validators.py -q`

Expected: FAIL because `astro_assistant.validators` does not exist.

- [ ] **Step 3: Implement validators**

Create `src/astro_assistant/validators.py`:

```python
import json
from pathlib import Path

from astro_assistant.models import ArtifactKind, WorkflowArtifact


_JSON_KINDS = {
    ArtifactKind.MEASUREMENTS_JSON,
    ArtifactKind.ESTIMATE_JSON,
    ArtifactKind.TRACE_JSON,
}


def validate_artifact(artifact: WorkflowArtifact) -> bool:
    path = Path(artifact.path)
    if not path.exists():
        return not artifact.required
    if artifact.kind in _JSON_KINDS:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
    return True
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_assistant/test_validators.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_assistant/validators.py tests/astro_assistant/test_validators.py
git commit -m "feat: add assistant artifact validators"
```

## Task 7: Executor And Trace Writer

**Files:**
- Create: `src/astro_assistant/executor.py`
- Test: `tests/astro_assistant/test_executor.py`

- [ ] **Step 1: Write executor tests**

```python
from collections.abc import Sequence

from astro_assistant.executor import WorkflowExecutor
from astro_assistant.planner import local_od_demo_plan


class FakeRunner:
    def __call__(self, argv: Sequence[str], cwd: str | None) -> tuple[int, str, str]:
        return 0, "ok", ""


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
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/astro_assistant/test_executor.py -q`

Expected: FAIL because `astro_assistant.executor` does not exist.

- [ ] **Step 3: Implement executor**

Create `src/astro_assistant/executor.py`:

```python
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
            returncode, stdout, stderr = self._command_runner(command_spec.argv, command_spec.cwd)
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_assistant/test_executor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_assistant/executor.py tests/astro_assistant/test_executor.py
git commit -m "feat: add assistant workflow executor"
```

## Task 8: CLI Dry-Run And Execution Commands

**Files:**
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_cli/test_assistant_cli.py`

- [ ] **Step 1: Write CLI tests**

```python
from typer.testing import CliRunner

from astro_cli.main import app


def test_assistant_ask_dry_run_prints_plan() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["ask", "Run the local OD demo", "--dry-run"])

    assert result.exit_code == 0
    assert "local-od-demo" in result.stdout
    assert "estimate-measurements" in result.stdout


def test_assistant_ask_requires_approval_for_execution() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["ask", "Run the local OD demo", "--execute"])

    assert result.exit_code == 2
    assert "execution requires approval" in result.stdout or "execution requires approval" in result.stderr
```

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/astro_cli/test_assistant_cli.py -q`

Expected: FAIL because `ask` is not registered.

- [ ] **Step 3: Add CLI command imports**

Add near the top of `src/astro_cli/main.py`:

```python
from astro_assistant.executor import WorkflowExecutor
from astro_assistant.planner import DeterministicPlanner
```

- [ ] **Step 4: Add `ask` command**

Append this command near other public CLI commands in `src/astro_cli/main.py`:

```python
@app.command("ask")
def ask_assistant(
    prompt: Annotated[str, typer.Argument()],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = True,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    approved: Annotated[bool, typer.Option("--approved")] = False,
) -> None:
    """Compile natural language into a typed Astro workflow plan."""
    planner = DeterministicPlanner()
    try:
        plan = planner.plan(prompt)
        trace = WorkflowExecutor().run(
            plan,
            dry_run=dry_run or not execute,
            approved=approved,
            cwd=None,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(trace.model_dump_json(indent=2))
    if trace.warnings and execute and not approved:
        raise typer.Exit(code=2)
```

- [ ] **Step 5: Run CLI tests**

Run: `python -m pytest tests/astro_cli/test_assistant_cli.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/astro_cli/main.py tests/astro_cli/test_assistant_cli.py
git commit -m "feat: expose assistant workflow cli"
```

## Task 9: Trace File Output

**Files:**
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_cli/test_assistant_cli.py`

- [ ] **Step 1: Add trace output test**

```python
def test_assistant_ask_writes_trace_file(tmp_path) -> None:
    runner = CliRunner()
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
    assert "local-od-demo" in trace_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run failing test**

Run: `python -m pytest tests/astro_cli/test_assistant_cli.py::test_assistant_ask_writes_trace_file -q`

Expected: FAIL because `--trace-output` is not registered.

- [ ] **Step 3: Add CLI trace output parameter**

Modify `ask_assistant` signature:

```python
    trace_output: Annotated[Path | None, typer.Option("--trace-output")] = None,
```

Add after trace creation:

```python
    payload = trace.model_dump_json(indent=2)
    if trace_output is not None:
        _write_text_or_exit(trace_output, payload, "assistant trace")
    typer.echo(payload)
```

Remove the earlier direct `typer.echo(trace.model_dump_json(indent=2))` line.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/astro_cli/test_assistant_cli.py::test_assistant_ask_writes_trace_file -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_cli/main.py tests/astro_cli/test_assistant_cli.py
git commit -m "feat: write assistant trace output"
```

## Task 10: Public Documentation

**Files:**
- Create: `docs/assistant-workflows.md`
- Modify: `README.md`
- Modify: `docs/validation-matrix.md`

- [ ] **Step 1: Create assistant workflow docs**

Create `docs/assistant-workflows.md`:

```markdown
# Assistant Workflows

Astro Suite's assistant layer compiles natural-language mission-analysis requests into typed,
reviewable workflow plans. The assistant does not perform flight dynamics itself. Astro Suite CLI
commands generate and validate the artifacts.

## First Supported Workflow

The first workflow is the local orbit-determination demo:

```bash
astro ask "Run the local OD demo" --dry-run
astro ask "Run the local OD demo" --execute --approved --trace-output /tmp/astro-assistant/trace.json
```

The generated plan validates `examples/scenarios/leo_two_station_od.yaml`, synthesizes local
measurements, exports TDM, estimates the initial state, and records a trace.

## Safety Boundaries

- Plans are typed Pydantic models.
- Commands are generated from an allow-listed registry.
- Execution defaults to dry-run.
- Artifact-writing execution requires `--approved`.
- Optional backends are blocked in this first assistant slice.
- Arbitrary shell commands are not supported.
```
```

- [ ] **Step 2: Add README link**

Add under README verification or documentation links:

```markdown
- [Assistant workflows](docs/assistant-workflows.md)
```

- [ ] **Step 3: Add validation matrix row**

Add a required local gate row:

```markdown
| Assistant dry-run | `astro ask "Run the local OD demo" --dry-run` | Writes a typed dry-run trace to stdout with four allow-listed command specs and no tool execution. |
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/assistant-workflows.md docs/validation-matrix.md
git commit -m "docs: document assistant workflow boundary"
```

## Task 11: Focused Verification

**Files:**
- No code edits.

- [ ] **Step 1: Run assistant tests**

Run:

```bash
python -m pytest tests/astro_assistant tests/astro_cli/test_assistant_cli.py -q
```

Expected: all assistant tests pass.

- [ ] **Step 2: Run static checks**

Run:

```bash
python -m ruff check src/astro_assistant tests/astro_assistant tests/astro_cli/test_assistant_cli.py
python -m mypy
```

Expected: no lint findings and strict typing passes.

- [ ] **Step 3: Run full non-live tests**

Run:

```bash
python -m pytest -q
```

Expected: all non-live tests pass; optional live backend tests remain skipped unless their env gates are enabled.

- [ ] **Step 4: Commit verification docs if needed**

If validation matrix wording changes during verification, commit only those doc changes:

```bash
git add docs/validation-matrix.md
git commit -m "docs: update assistant validation gate"
```

## Task 12: MCP Follow-On Contract

**Files:**
- Create: `docs/assistant-mcp-contract.md`

- [ ] **Step 1: Document MCP boundary**

Create `docs/assistant-mcp-contract.md`:

```markdown
# Assistant MCP Contract

The MCP server is a later interface over the same assistant registry used by `astro ask`.

## Tools

- `astro_plan_workflow`: accepts natural-language intent and returns `AstroWorkflowPlan`.
- `astro_dry_run_workflow`: accepts an `AstroWorkflowPlan` and returns command specs plus warnings.
- `astro_execute_workflow`: accepts an approved plan and returns `WorkflowTrace`.

## Resources

- `astro://examples/assistant/od_workflow_prompt`
- `astro://schemas/assistant/workflow-plan`
- `astro://schemas/assistant/workflow-trace`

## Policy

The MCP layer must not expose arbitrary shell execution. It must call the same registry, policy, and
executor modules used by the CLI. Write-producing tools must require explicit approval from the MCP
client before execution.
```

- [ ] **Step 2: Commit MCP contract**

```bash
git add docs/assistant-mcp-contract.md
git commit -m "docs: define assistant mcp contract"
```

## Self-Review Checklist

- Spec coverage: the plan implements the core workflow kernel, OD workflow pack, CLI dry-run/execute path, artifact validation, audit trace, public docs, and MCP contract boundary.
- Placeholder scan: no task relies on unspecified code or unspecified tests.
- Type consistency: `AstroWorkflowPlan`, `WorkflowStep`, `WorkflowTrace`, `CommandSpec`, and `WorkflowArtifact` are defined before use and reused consistently.
- Risk boundary: write-producing execution requires approval; optional backends are blocked in the first slice; no arbitrary shell execution is introduced.
- Public credibility: the first demo uses the existing local OD validation surface and avoids optional backend install friction.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-20-verifiable-ai-space-workflows-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
