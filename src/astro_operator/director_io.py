"""Loading, publication inventory, and offline verification for mission design runs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_mission.io import load_mission_lifecycle_result, load_mission_lifecycle_scenario
from astro_mission.models import MissionLifecycleInputOverrides, MissionLifecycleScenario
from astro_operator.director import (
    MissionDesignDirectorSpec,
    MissionDesignRun,
    build_mission_design_run,
)
from astro_operator.io import verify_operator_run
from astro_operator.models import (
    EpistemicKind,
    EvidenceReference,
    ObservedMetric,
    OperatorRun,
)


class DirectorArtifact(AstroModel):
    path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class MissionDesignManifest(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["mission_design_director_v1"] = "mission_design_director_v1"
    artifacts: tuple[DirectorArtifact, ...] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def artifacts_must_be_unique(self) -> MissionDesignManifest:
        paths = [item.path for item in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("director artifact paths must be unique")
        if paths != sorted(paths):
            raise ValueError("director artifact inventory must be path-sorted")
        return self


def load_mission_design_director_spec(
    path: Path | str,
) -> MissionDesignDirectorSpec:
    spec_path = Path(path)
    try:
        raw: Any = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        return MissionDesignDirectorSpec.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load mission design director spec {spec_path}: {exc}"
        ) from exc


def write_mission_design_run(path: Path, run: MissionDesignRun) -> None:
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")


def capture_resolved_base_scenario(
    scenario: MissionLifecycleScenario, operator_root: Path
) -> EvidenceReference:
    path = operator_root / "inputs" / "resolved-base-scenario.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scenario.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return EvidenceReference(
        evidence_id="resolved-base-scenario",
        kind="resolved_mission_lifecycle_scenario",
        epistemic_kind=EpistemicKind.DECLARED,
        claim_scope="resolved declared input to mission design lifecycle screening",
        path=path.relative_to(operator_root).as_posix(),
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def write_mission_design_manifest(root: Path) -> MissionDesignManifest:
    artifacts = tuple(
        DirectorArtifact(
            path=path.relative_to(root).as_posix(),
            role=_artifact_role(path.relative_to(root).as_posix()),
            sha256=sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "director-manifest.json"
    )
    payload = {
        "schema_version": "1.0",
        "workflow": "mission_design_director_v1",
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
    }
    manifest = MissionDesignManifest.model_validate(
        {**payload, "manifest_sha256": _canonical_digest(payload)}
    )
    (root / "director-manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_mission_design_director(root: Path | str) -> MissionDesignRun:
    run_root = Path(root)
    if run_root.is_symlink():
        raise InvalidScenarioError("Mission design root must not be a symbolic link")
    manifest_path = run_root / "director-manifest.json"
    run_path = run_root / "mission-design-run.json"
    spec_path = run_root / "inputs" / "design-spec.yaml"
    for path in (manifest_path, run_path, spec_path):
        if path.is_symlink():
            raise InvalidScenarioError(
                f"Mission design artifact must not be a symbolic link: {path.name}"
            )
    try:
        manifest = MissionDesignManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load mission design manifest {manifest_path}: {exc}"
        ) from exc
    manifest_payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if manifest.manifest_sha256 != _canonical_digest(manifest_payload):
        raise InvalidScenarioError("Mission design manifest digest mismatch")
    expected_paths = {item.path for item in manifest.artifacts}
    actual_paths: set[str] = set()
    for path in run_root.rglob("*"):
        if path.is_symlink():
            raise InvalidScenarioError(
                f"Mission design bundle contains a symbolic link: {path.relative_to(run_root)}"
            )
        if path.is_file() and path != manifest_path:
            actual_paths.add(path.relative_to(run_root).as_posix())
    if actual_paths != expected_paths:
        raise InvalidScenarioError("Mission design artifact inventory does not match the bundle")
    for artifact in manifest.artifacts:
        relative = Path(artifact.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise InvalidScenarioError("Mission design artifact path escapes the bundle")
        path = run_root / relative
        if artifact.role != _artifact_role(artifact.path):
            raise InvalidScenarioError(
                f"Mission design artifact role mismatch: {artifact.path}"
            )
        if path.stat().st_size != artifact.size_bytes:
            raise InvalidScenarioError(
                f"Mission design artifact size mismatch: {artifact.path}"
            )
        if sha256(path.read_bytes()).hexdigest() != artifact.sha256:
            raise InvalidScenarioError(
                f"Mission design artifact digest mismatch: {artifact.path}"
            )
    try:
        stored = MissionDesignRun.model_validate_json(run_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(f"Could not load mission design run {run_path}: {exc}") from exc
    run_payload = stored.model_dump(mode="json", exclude={"run_sha256"})
    if stored.run_sha256 != _canonical_digest(run_payload):
        raise InvalidScenarioError("Mission design run digest mismatch")
    spec = load_mission_design_director_spec(spec_path)
    operator_run = verify_operator_run(run_root / "operator")
    _verify_lifecycle_observations(run_root / "operator", operator_run)
    expected = build_mission_design_run(
        spec=spec,
        operator_run=operator_run,
        spec_sha256=sha256(spec_path.read_bytes()).hexdigest(),
        operator_run_sha256=sha256(
            (run_root / stored.operator_run_path).read_bytes()
        ).hexdigest(),
    )
    if stored != expected:
        raise InvalidScenarioError(
            "Mission design derived decision does not match the verified operator evidence"
        )
    return stored


def _verify_lifecycle_observations(operator_root: Path, run: OperatorRun) -> None:
    variables = {item.variable_id: item for item in run.objective.design_variables}
    base_evidence = {item.evidence_id: item for item in run.objective.base_evidence}
    source_reference = base_evidence.get("base-scenario")
    resolved_reference = base_evidence.get("resolved-base-scenario")
    if source_reference is None or resolved_reference is None:
        raise InvalidScenarioError("Director operator lacks captured base scenario evidence")
    declared_base = load_mission_lifecycle_scenario(operator_root / source_reference.path)
    try:
        resolved_base = MissionLifecycleScenario.model_validate_json(
            (operator_root / resolved_reference.path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not reconstruct resolved base lifecycle scenario: {exc}"
        ) from exc
    expected_resolved_base = declared_base.model_copy(
        update={
            "launch_scenario": resolved_base.launch_scenario,
            "twin_scenario": resolved_base.twin_scenario,
            "reentry_scenario": resolved_base.reentry_scenario,
        }
    )
    if resolved_base != expected_resolved_base:
        raise InvalidScenarioError(
            "Resolved base scenario differs from its declaration beyond reference resolution"
        )
    for step in run.steps:
        observation = step.observation
        if observation is None:
            continue
        evidence_by_kind = {item.kind: item for item in observation.evidence}
        if len(evidence_by_kind) != len(observation.evidence):
            raise InvalidScenarioError("Candidate observation evidence kinds must be unique")
        scenario_reference = evidence_by_kind.get("mission_lifecycle_scenario")
        if scenario_reference is None:
            raise InvalidScenarioError("Candidate observation lacks its lifecycle scenario")
        scenario_path = operator_root / scenario_reference.path
        try:
            scenario = MissionLifecycleScenario.model_validate_json(
                scenario_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError) as exc:
            raise InvalidScenarioError(
                f"Could not reconstruct candidate lifecycle scenario: {exc}"
            ) from exc
        candidate = observation.candidate
        override_payload = (
            resolved_base.input_overrides.model_dump(mode="python")
            if resolved_base.input_overrides is not None
            else {}
        )
        for variable_id, value in candidate.assignments.items():
            variable = variables.get(variable_id)
            if variable is None:
                raise InvalidScenarioError("Candidate assignment lacks a typed lifecycle override")
            override_payload[variable.target] = value
        expected_scenario = resolved_base.model_copy(
            update={
                "scenario_id": f"{resolved_base.scenario_id}--{candidate.candidate_id}",
                "input_overrides": MissionLifecycleInputOverrides.model_validate(
                    override_payload
                ),
                "metadata": {
                    **resolved_base.metadata,
                    "operator_candidate_id": candidate.candidate_id,
                },
            }
        )
        if scenario != expected_scenario:
            raise InvalidScenarioError(
                "Candidate scenario is not the exact base scenario plus journaled assignments"
            )
        if scenario.input_overrides is None:
            raise InvalidScenarioError("Candidate scenario lacks resolved input overrides")
        resolved_overrides = scenario.input_overrides.model_dump(
            mode="json", exclude_none=True, exclude_defaults=True
        )
        expected_metadata = {
            "claim_boundary": scenario.metadata.get("claim_boundary"),
        }
        if observation.evaluation_status == "evaluated":
            result_reference = evidence_by_kind.get("mission_lifecycle_result")
            if result_reference is None or len(observation.evidence) != 2:
                raise InvalidScenarioError("Evaluated candidate lacks exact result evidence")
            result = load_mission_lifecycle_result(operator_root / result_reference.path)
            if result.scenario_id != scenario.scenario_id:
                raise InvalidScenarioError("Candidate result does not match its scenario artifact")
            if (
                result.metadata.get("resolved_input_overrides") != resolved_overrides
                or result.metadata.get("operator_scenario_sha256")
                != scenario_reference.sha256
            ):
                raise InvalidScenarioError(
                    "Candidate result provenance does not bind its exact scenario inputs"
                )
            expected_metrics = tuple(
                ObservedMetric(
                    metric_id=f"margin:{margin.phase}:{margin.name}",
                    value=margin.margin,
                    unit=margin.unit,
                    status=margin.status.value,
                )
                for margin in result.margin_report.margins
            )
            expected_metadata.update(
                {
                    "workflow": result.workflow,
                    "overall_status": result.margin_report.overall_status.value,
                    "continuity_all_passed": result.continuity_report.all_passed,
                }
            )
            if (
                observation.metrics != expected_metrics
                or observation.passed != result.passed
                or observation.warnings != tuple(result.warnings)
                or observation.metadata != expected_metadata
            ):
                raise InvalidScenarioError(
                    "Candidate observation does not match its lifecycle result artifact"
                )
            continue
        if observation.evaluation_status != "evaluation_failed":
            raise InvalidScenarioError("Candidate observation has an unsupported status")
        error_reference = evidence_by_kind.get("mission_lifecycle_evaluation_error")
        if error_reference is None or len(observation.evidence) != 2:
            raise InvalidScenarioError("Failed candidate lacks exact error evidence")
        try:
            error_payload: Any = json.loads(
                (operator_root / error_reference.path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidScenarioError(
                f"Could not reconstruct candidate evaluation failure: {exc}"
            ) from exc
        if (
            not isinstance(error_payload, dict)
            or error_payload.get("candidate_id") != candidate.candidate_id
            or error_payload.get("operator_scenario_sha256")
            != scenario_reference.sha256
            or error_payload.get("resolved_input_overrides") != resolved_overrides
            or observation.passed
            or observation.metrics
            or observation.warnings != (error_payload.get("message"),)
            or observation.metadata != expected_metadata
        ):
            raise InvalidScenarioError(
                "Candidate failure observation does not match its error artifact"
            )


def _artifact_role(path: str) -> str:
    if path == "inputs/design-spec.yaml":
        return "declared_design_contract"
    if path == "mission-design-run.json":
        return "derived_design_decision"
    if path == "operator/operator-run.json":
        return "adaptive_operator_journal"
    if path.startswith("operator/candidates/"):
        return "candidate_analysis_evidence"
    if path.startswith("operator/inputs/"):
        return "captured_analysis_input"
    return "supporting_artifact"


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "capture_resolved_base_scenario",
    "DirectorArtifact",
    "MissionDesignManifest",
    "load_mission_design_director_spec",
    "verify_mission_design_director",
    "write_mission_design_manifest",
    "write_mission_design_run",
]
