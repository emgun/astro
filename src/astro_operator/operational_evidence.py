"""Concrete, provider-neutral evidence tools for operational review workflows."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, FiniteFloat, field_validator, model_validator

from astro_core.models import AstroModel
from astro_operator.evidence_tools import EvidenceToolRegistry
from astro_operator.models import (
    AcquisitionStatus,
    EpistemicKind,
    EvidenceAcquisitionResult,
    EvidenceAssertion,
    EvidenceReference,
    EvidenceRequest,
    EvidenceToolSpec,
    WorldState,
)


class OperationalEvidenceKind(StrEnum):
    SIMULATED_TELEMETRY = "simulated_telemetry"
    ORBIT_ESTIMATE = "orbit_estimate"
    PROCEDURE = "procedure"


class OperationalEvidenceSource(AstroModel):
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    kind: OperationalEvidenceKind
    path: str = Field(min_length=1)


class SimulatedTelemetrySnapshot(AstroModel):
    snapshot_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    observed_at: datetime
    decision_time: datetime
    operating_mode: str = Field(min_length=1)
    source_simulation: str = Field(min_length=1)

    @field_validator("observed_at", "decision_time")
    @classmethod
    def observed_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("telemetry timestamps must include timezone information")
        return value

    @model_validator(mode="after")
    def decision_must_not_precede_observation(self) -> SimulatedTelemetrySnapshot:
        if self.decision_time < self.observed_at:
            raise ValueError("telemetry decision_time cannot precede observed_at")
        return self


class OrbitEstimateSnapshot(AstroModel):
    estimate_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    telemetry_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimated_at: datetime
    estimator: str = Field(min_length=1)
    converged: bool
    measurement_count: int = Field(ge=0)
    position_sigma_km: FiniteFloat = Field(ge=0.0)
    normalized_residual_rms: FiniteFloat = Field(ge=0.0)

    @field_validator("estimated_at")
    @classmethod
    def estimated_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("estimate estimated_at must include timezone information")
        return value

    @field_validator("measurement_count", mode="before")
    @classmethod
    def measurement_count_must_be_integer(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("estimate measurement_count must be an integer")
        return value


class ProcedureSnapshot(AstroModel):
    procedure_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    applicability_scope: str = Field(min_length=1)
    applicable_configuration_id: str = Field(min_length=1)
    effective_from: datetime
    expires_at: datetime
    maximum_position_sigma_km: FiniteFloat = Field(gt=0.0)
    maximum_age_s: FiniteFloat = Field(ge=0.0)
    required_operating_mode: str = Field(min_length=1)
    manual_review_required: Literal[True] = True

    @model_validator(mode="after")
    def validity_must_be_aware_and_ordered(self) -> ProcedureSnapshot:
        for value in (self.effective_from, self.expires_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("procedure validity timestamps must include timezone information")
        if self.expires_at <= self.effective_from:
            raise ValueError("procedure expiry must follow its effective time")
        return self


class _SourceRequest(AstroModel):
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _schema_digest(model: type[AstroModel]) -> str:
    payload = json.dumps(
        model.model_json_schema(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class _CapturedSourceTool:
    source_kind: OperationalEvidenceKind
    artifact_kind: str
    epistemic_kind: EpistemicKind
    model: type[AstroModel]
    spec: EvidenceToolSpec

    def __init__(
        self,
        sources: tuple[OperationalEvidenceSource, ...],
        *,
        source_root: Path,
        output_root: Path,
    ) -> None:
        self._sources = {
            source.source_id: source
            for source in sources
            if source.kind == self.source_kind
        }
        self._source_root = source_root
        self._output_root = output_root

    def acquire(
        self, request: EvidenceRequest, world_state: WorldState
    ) -> EvidenceAcquisitionResult:
        del world_state
        parameters = _SourceRequest.model_validate(request.parameters)
        source = self._sources.get(parameters.source_id)
        if source is None:
            return EvidenceAcquisitionResult(
                request=request,
                tool=self.spec,
                status=AcquisitionStatus.FAILED,
                message=f"unknown {self.source_kind.value} source {parameters.source_id!r}",
            )
        source_path = Path(source.path)
        if source_path.is_absolute() or ".." in source_path.parts:
            return EvidenceAcquisitionResult(
                request=request,
                tool=self.spec,
                status=AcquisitionStatus.FAILED,
                message=f"{self.source_kind.value} source path escapes its catalog root",
            )
        if self._source_root.resolve() != self._source_root.absolute():
            return EvidenceAcquisitionResult(
                request=request,
                tool=self.spec,
                status=AcquisitionStatus.FAILED,
                message=f"{self.source_kind.value} catalog root contains a symbolic link",
            )
        source_path = self._source_root / source_path
        source_root = self._source_root.resolve()
        if (
            any(part.is_symlink() for part in _path_prefixes(self._source_root, Path(source.path)))
            or not source_path.resolve().is_relative_to(source_root)
            or not source_path.is_file()
        ):
            return EvidenceAcquisitionResult(
                request=request,
                tool=self.spec,
                status=AcquisitionStatus.FAILED,
                message=f"{self.source_kind.value} source is not a regular file",
            )
        try:
            raw_bytes = source_path.read_bytes()
            raw = yaml.safe_load(raw_bytes)
            product = self.model.model_validate(raw)
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            return EvidenceAcquisitionResult(
                request=request,
                tool=self.spec,
                status=AcquisitionStatus.FAILED,
                message=f"{self.source_kind.value} source is invalid: {exc}",
            )
        suffix = source_path.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            suffix = ".yaml"
        captured_path = (
            self._output_root
            / "evidence"
            / f"{self.source_kind.value}-{source.source_id}{suffix}"
        )
        captured_path.parent.mkdir(parents=True, exist_ok=True)
        captured_path.write_bytes(raw_bytes)
        evidence = EvidenceReference(
            evidence_id=f"{request.request_id}.source",
            kind=self.artifact_kind,
            epistemic_kind=self.epistemic_kind,
            claim_scope=self._claim_scope(product),
            path=captured_path.relative_to(self._output_root).as_posix(),
            sha256=sha256(raw_bytes).hexdigest(),
            metadata={
                "source_id": source.source_id,
                "tool_id": self.spec.tool_id,
                "tool_version": self.spec.version,
            },
        )
        return EvidenceAcquisitionResult(
            request=request,
            tool=self.spec,
            status=AcquisitionStatus.SUCCEEDED,
            evidence=(evidence,),
            assertions=self._assertions(request, evidence, product),
        )

    def _claim_scope(self, product: AstroModel) -> str:
        raise NotImplementedError

    def _assertions(
        self,
        request: EvidenceRequest,
        evidence: EvidenceReference,
        product: AstroModel,
    ) -> tuple[EvidenceAssertion, ...]:
        raise NotImplementedError

    def _assertion(
        self,
        request: EvidenceRequest,
        evidence: EvidenceReference,
        *,
        suffix: str,
        subject: str,
        predicate: str,
        value: Any,
        scope: str,
        valid_at: datetime | None,
    ) -> EvidenceAssertion:
        return EvidenceAssertion(
            assertion_id=f"{request.request_id}.{suffix}",
            subject=subject,
            predicate=predicate,
            value=value,
            epistemic_kind=self.epistemic_kind,
            scope=scope,
            source_evidence_ids=(evidence.evidence_id,),
            producer_tool_id=self.spec.tool_id,
            producer_tool_version=self.spec.version,
            valid_at=valid_at,
        )


class SimulatedTelemetryTool(_CapturedSourceTool):
    source_kind = OperationalEvidenceKind.SIMULATED_TELEMETRY
    artifact_kind = "simulated_telemetry_snapshot"
    epistemic_kind = EpistemicKind.SIMULATED
    model = SimulatedTelemetrySnapshot
    spec = EvidenceToolSpec(
        tool_id="astro.simulated_telemetry_snapshot",
        version="1.0",
        request_kind="read_simulated_telemetry",
        parameter_schema_sha256=_schema_digest(_SourceRequest),
        output_assertion_kinds=(
            "asset_configuration_id",
            "operating_mode",
            "telemetry_source_sha256",
            "simulated_decision_time",
        ),
    )

    def _claim_scope(self, product: AstroModel) -> str:
        snapshot = SimulatedTelemetrySnapshot.model_validate(product)
        return f"simulated telemetry {snapshot.snapshot_id} for {snapshot.asset_id}"

    def _assertions(
        self,
        request: EvidenceRequest,
        evidence: EvidenceReference,
        product: AstroModel,
    ) -> tuple[EvidenceAssertion, ...]:
        snapshot = SimulatedTelemetrySnapshot.model_validate(product)
        scope = self._claim_scope(snapshot)
        return (
            self._assertion(
                request,
                evidence,
                suffix="configuration",
                subject=snapshot.asset_id,
                predicate="asset_configuration_id",
                value=snapshot.configuration_id,
                scope=scope,
                valid_at=snapshot.observed_at,
            ),
            self._assertion(
                request,
                evidence,
                suffix="mode",
                subject=snapshot.asset_id,
                predicate="operating_mode",
                value=snapshot.operating_mode,
                scope=scope,
                valid_at=snapshot.observed_at,
            ),
            self._assertion(
                request,
                evidence,
                suffix="source-sha256",
                subject=snapshot.asset_id,
                predicate="telemetry_source_sha256",
                value=evidence.sha256,
                scope=scope,
                valid_at=snapshot.observed_at,
            ),
            self._assertion(
                request,
                evidence,
                suffix="decision-time",
                subject=snapshot.asset_id,
                predicate="simulated_decision_time",
                value=snapshot.decision_time.isoformat(),
                scope=scope,
                valid_at=snapshot.decision_time,
            ),
        )


class OrbitEstimateTool(_CapturedSourceTool):
    source_kind = OperationalEvidenceKind.ORBIT_ESTIMATE
    artifact_kind = "orbit_estimate_snapshot"
    epistemic_kind = EpistemicKind.ESTIMATED
    model = OrbitEstimateSnapshot
    spec = EvidenceToolSpec(
        tool_id="astro.orbit_estimate",
        version="1.0",
        request_kind="read_orbit_estimate",
        parameter_schema_sha256=_schema_digest(_SourceRequest),
        output_assertion_kinds=(
            "estimate_configuration_id",
            "telemetry_source_sha256",
            "estimate_converged",
            "measurement_count",
            "position_sigma_km",
            "normalized_residual_rms",
        ),
    )

    def _claim_scope(self, product: AstroModel) -> str:
        estimate = OrbitEstimateSnapshot.model_validate(product)
        return f"orbit estimate {estimate.estimate_id} for {estimate.asset_id}"

    def _assertions(
        self,
        request: EvidenceRequest,
        evidence: EvidenceReference,
        product: AstroModel,
    ) -> tuple[EvidenceAssertion, ...]:
        estimate = OrbitEstimateSnapshot.model_validate(product)
        scope = self._claim_scope(estimate)
        values = (
            ("configuration", "estimate_configuration_id", estimate.configuration_id),
            (
                "telemetry-source-sha256",
                "telemetry_source_sha256",
                estimate.telemetry_source_sha256,
            ),
            ("converged", "estimate_converged", estimate.converged),
            ("measurement-count", "measurement_count", estimate.measurement_count),
            ("position-sigma", "position_sigma_km", estimate.position_sigma_km),
            (
                "normalized-rms",
                "normalized_residual_rms",
                estimate.normalized_residual_rms,
            ),
        )
        return tuple(
            self._assertion(
                request,
                evidence,
                suffix=suffix,
                subject=estimate.asset_id,
                predicate=predicate,
                value=value,
                scope=scope,
                valid_at=estimate.estimated_at,
            )
            for suffix, predicate, value in values
        )


class ProcedureTool(_CapturedSourceTool):
    source_kind = OperationalEvidenceKind.PROCEDURE
    artifact_kind = "declared_operational_procedure"
    epistemic_kind = EpistemicKind.DECLARED
    model = ProcedureSnapshot
    spec = EvidenceToolSpec(
        tool_id="astro.procedure_snapshot",
        version="1.0",
        request_kind="read_procedure",
        parameter_schema_sha256=_schema_digest(_SourceRequest),
        output_assertion_kinds=(
            "applicable_configuration_id",
            "maximum_position_sigma_km",
            "maximum_age_s",
            "procedure_validity_s",
            "required_operating_mode",
            "manual_review_required",
        ),
    )

    def _claim_scope(self, product: AstroModel) -> str:
        procedure = ProcedureSnapshot.model_validate(product)
        return procedure.applicability_scope

    def _assertions(
        self,
        request: EvidenceRequest,
        evidence: EvidenceReference,
        product: AstroModel,
    ) -> tuple[EvidenceAssertion, ...]:
        procedure = ProcedureSnapshot.model_validate(product)
        values = (
            (
                "configuration",
                "applicable_configuration_id",
                procedure.applicable_configuration_id,
            ),
            (
                "position-sigma-limit",
                "maximum_position_sigma_km",
                procedure.maximum_position_sigma_km,
            ),
            ("maximum-age", "maximum_age_s", procedure.maximum_age_s),
            (
                "validity",
                "procedure_validity_s",
                (procedure.expires_at - procedure.effective_from).total_seconds(),
            ),
            ("required-mode", "required_operating_mode", procedure.required_operating_mode),
            (
                "manual-review",
                "manual_review_required",
                procedure.manual_review_required,
            ),
        )
        return tuple(
            self._assertion(
                request,
                evidence,
                suffix=suffix,
                subject=procedure.asset_id,
                predicate=predicate,
                value=value,
                scope=procedure.applicability_scope,
                valid_at=procedure.effective_from,
            )
            for suffix, predicate, value in values
        )


def build_operational_evidence_registry(
    sources: tuple[OperationalEvidenceSource, ...],
    *,
    source_root: Path,
    output_root: Path,
) -> EvidenceToolRegistry:
    source_ids = [source.source_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("operational evidence source IDs must be unique")
    tools: list[_CapturedSourceTool] = []
    for tool_type in (SimulatedTelemetryTool, OrbitEstimateTool, ProcedureTool):
        if any(source.kind == tool_type.source_kind for source in sources):
            tools.append(tool_type(sources, source_root=source_root, output_root=output_root))
    return EvidenceToolRegistry(tools)


def verify_operational_acquisition(
    result: EvidenceAcquisitionResult, artifact_path: Path
) -> None:
    """Re-derive concrete tool assertions from the captured source bytes."""

    tool_types = {
        tool_type.spec.tool_id: tool_type
        for tool_type in (SimulatedTelemetryTool, OrbitEstimateTool, ProcedureTool)
    }
    tool_type = tool_types.get(result.tool.tool_id)
    if tool_type is None:
        return
    if result.status == AcquisitionStatus.FAILED:
        if result.evidence:
            raise ValueError(
                "failed operational evidence acquisition cannot retain captured sources"
            )
        return
    if len(result.evidence) != 1:
        raise ValueError("operational evidence acquisition must contain one captured source")
    evidence = result.evidence[0]
    if (
        evidence.kind != tool_type.artifact_kind
        or evidence.epistemic_kind != tool_type.epistemic_kind
    ):
        raise ValueError("operational evidence reference does not match its tool contract")
    try:
        raw = yaml.safe_load(artifact_path.read_bytes())
        product = tool_type.model.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError("operational evidence artifact does not satisfy its tool schema") from exc
    tool = tool_type((), source_root=artifact_path.parent, output_root=artifact_path.parent)
    if evidence.claim_scope != tool._claim_scope(product):
        raise ValueError("operational evidence claim scope does not match captured content")
    from astro_operator.world_state import reduce_world_state

    expected = reduce_world_state(
        tool._assertions(result.request, evidence, product)
    ).assertions
    if result.assertions != expected:
        raise ValueError("operational evidence assertions do not match captured content")


def _path_prefixes(root: Path, relative: Path) -> tuple[Path, ...]:
    prefixes: list[Path] = []
    current = root
    for part in relative.parts:
        current = current / part
        prefixes.append(current)
    return tuple(prefixes)
