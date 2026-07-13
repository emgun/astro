from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

SchemaVersion = Literal["1.0"]
Digest = str
Identifier = str


class SurrogateModel(BaseModel):
    """Strict, immutable base for persisted surrogate lifecycle records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BOUNDARY_CHALLENGE = "boundary_challenge"
    OOD_CHALLENGE = "ood_challenge"


class PromotionStatus(StrEnum):
    REJECTED = "rejected"
    EXPERIMENTAL = "experimental"
    CAMPAIGN_SCREENING = "campaign_screening"


class EvaluatorIdentity(SurrogateModel):
    evaluator_id: Identifier = Field(min_length=1)
    implementation: str = Field(min_length=1)
    version: str = Field(min_length=1)
    configuration_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = Field(min_length=1)


class FieldSchema(SurrogateModel):
    name: Identifier = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    unit: str = Field(min_length=1)
    frame: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SimulationDatasetSpec(SurrogateModel):
    schema_version: SchemaVersion = "1.0"
    dataset_id: Identifier = Field(min_length=1)
    scenario_family: str = Field(min_length=1)
    teacher: EvaluatorIdentity
    baseline: EvaluatorIdentity
    parameter_domain: dict[str, tuple[FiniteFloat, FiniteFloat]]
    features: tuple[FieldSchema, ...] = Field(min_length=1)
    targets: tuple[FieldSchema, ...] = Field(min_length=1)
    time_coordinate: FieldSchema
    split_policy: str = Field(min_length=1)
    seed: int = Field(ge=0)
    retention_policy: str = Field(min_length=1)
    software_versions: dict[str, str] = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_domain_and_names(self) -> SimulationDatasetSpec:
        if any(low >= high for low, high in self.parameter_domain.values()):
            raise ValueError("parameter domain bounds must satisfy low < high")
        names = [field.name for field in (*self.features, *self.targets)]
        if len(names) != len(set(names)):
            raise ValueError("feature and target names must be unique")
        return self


class SimulationEpisodeRef(SurrogateModel):
    episode_id: Identifier = Field(min_length=1)
    split: DatasetSplit
    scenario_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    arrays: tuple[str, ...] = Field(min_length=1)
    sample_count: int = Field(gt=0)
    metadata: dict[str, str] = Field(default_factory=dict)


class DatasetManifest(SurrogateModel):
    schema_version: SchemaVersion = "1.0"
    dataset_id: Identifier = Field(min_length=1)
    spec_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    arrays_path: str = Field(min_length=1)
    arrays_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    episodes: tuple[SimulationEpisodeRef, ...] = Field(min_length=1)
    created_at: datetime
    software_versions: dict[str, str] = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information")
        return value

    @model_validator(mode="after")
    def episode_ids_must_be_unique(self) -> DatasetManifest:
        ids = [episode.episode_id for episode in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("episode ids must be unique")
        return self


class SurrogateDomain(SurrogateModel):
    domain_id: Identifier = Field(min_length=1)
    teacher: EvaluatorIdentity
    baseline: EvaluatorIdentity
    scenario_family: str = Field(min_length=1)
    parameter_bounds: dict[str, tuple[FiniteFloat, FiniteFloat]] = Field(min_length=1)
    feature_schema_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    target_schema_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    time_horizon_s: FiniteFloat = Field(gt=0.0)
    frame: str = Field(min_length=1)
    ood_policy: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def bounds_must_increase(self) -> SurrogateDomain:
        if any(low >= high for low, high in self.parameter_bounds.values()):
            raise ValueError("parameter bounds must satisfy low < high")
        return self


class SurrogateTrainingRun(SurrogateModel):
    schema_version: SchemaVersion = "1.0"
    run_id: Identifier = Field(min_length=1)
    dataset_manifest_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    domain_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    model_kind: str = Field(min_length=1)
    seed: int = Field(ge=0)
    hyperparameters: dict[str, Any]
    metrics: dict[str, FiniteFloat]
    wall_time_s: FiniteFloat = Field(ge=0.0)
    software_versions: dict[str, str] = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    numerical_warnings: tuple[str, ...] = ()


class SurrogateModelArtifact(SurrogateModel):
    schema_version: SchemaVersion = "1.0"
    model_id: Identifier = Field(min_length=1)
    training_run_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    domain_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    model_kind: str = Field(min_length=1)
    arrays_path: str = Field(min_length=1)
    arrays_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    target_schema_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")


class ValidationMetric(SurrogateModel):
    name: str = Field(min_length=1)
    value: FiniteFloat
    threshold: FiniteFloat
    passed: bool


class SurrogateValidationReport(SurrogateModel):
    schema_version: SchemaVersion = "1.0"
    report_id: Identifier = Field(min_length=1)
    model_artifact_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    validation_spec_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: tuple[ValidationMetric, ...] = Field(min_length=1)
    failure_challenge_false_negatives: int = Field(ge=0)
    ood_fallback_passed: bool
    deterministic_retraining_passed: bool
    overall_passed: bool
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def overall_result_must_match_evidence(self) -> SurrogateValidationReport:
        evidence_passed = (
            all(metric.passed for metric in self.metrics)
            and self.failure_challenge_false_negatives == 0
            and self.ood_fallback_passed
            and self.deterministic_retraining_passed
        )
        if self.overall_passed != evidence_passed:
            raise ValueError("overall_passed must match validation evidence")
        return self


class SurrogatePromotionDecision(SurrogateModel):
    schema_version: SchemaVersion = "1.0"
    decision_id: Identifier = Field(min_length=1)
    model_artifact_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    validation_report_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    status: PromotionStatus
    decided_at: datetime
    decided_by: str = Field(min_length=1)
    reasons: tuple[str, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @field_validator("decided_at")
    @classmethod
    def decided_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must include timezone information")
        return value
