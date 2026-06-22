from enum import StrEnum
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


class VerificationDiagnostic(BaseModel):
    code: str
    message: str
    severity: str = "error"


class VerificationResult(BaseModel):
    passed: bool
    diagnostics: list[VerificationDiagnostic] = Field(default_factory=list)


class WorkflowTrace(BaseModel):
    plan: AstroWorkflowPlan
    dry_run: bool
    command_specs: list[CommandSpec]
    verification: VerificationResult = Field(
        default_factory=lambda: VerificationResult(passed=True)
    )
    results: list[StepExecutionResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
