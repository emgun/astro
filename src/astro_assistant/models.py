from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator


class AstroToolName(StrEnum):
    VALIDATE_SCENARIO = "validate_scenario"
    SYNTH_MEASUREMENTS = "synth_measurements"
    EXPORT_MEASUREMENTS = "export_measurements"
    ESTIMATE_MEASUREMENTS = "estimate_measurements"
    VALIDATE_CAMPAIGN = "validate_campaign"
    RUN_CAMPAIGN = "run_campaign"
    SUMMARIZE_CAMPAIGN = "summarize_campaign"
    VERIFY_ASSURANCE_VALIDATION = "verify_assurance_validation"
    REVIEW_ASSURANCE_VALIDATION = "review_assurance_validation"
    COMPARE_ASSURANCE_REVIEWS = "compare_assurance_reviews"


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
    CAMPAIGN_DEFINITION = "campaign_definition"
    CAMPAIGN_RESULT = "campaign_result"
    CAMPAIGN_SUMMARY = "campaign_summary"
    ASSURANCE_VALIDATION_RESULT = "assurance_validation_result"
    ASSURANCE_REVIEW = "assurance_review"
    ASSURANCE_REVIEW_COMPARISON = "assurance_review_comparison"


class _CampaignInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _reject_option_like_paths(cls, value: object) -> object:
        if isinstance(value, str) and (value == "" or value.startswith("-") or "\x00" in value):
            raise ValueError("path inputs must be non-empty and may not look like command options")
        return value


class ValidateCampaignInputs(_CampaignInputs):
    definition_path: str


class RunCampaignInputs(_CampaignInputs):
    definition_path: str
    output_dir: str
    resume: StrictBool = False


class SummarizeCampaignInputs(_CampaignInputs):
    output_dir: str


class VerifyAssuranceValidationInputs(_CampaignInputs):
    result_path: str


class ReviewAssuranceValidationInputs(_CampaignInputs):
    result_path: str
    output: str
    summary_output: str | None = None


class CompareAssuranceReviewsInputs(_CampaignInputs):
    baseline_path: str
    candidate_path: str
    output: str
    summary_output: str | None = None


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
