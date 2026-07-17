from __future__ import annotations

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from astro_assurance.lifecycle_review import (
    review_mission_lifecycle,
    verify_mission_lifecycle_review,
)
from astro_assurance.lifecycle_review_io import (
    format_mission_lifecycle_review,
    load_mission_lifecycle_review,
    write_mission_lifecycle_review,
    write_mission_lifecycle_review_summary,
)
from astro_assurance.lifecycle_review_models import (
    LifecycleReviewInputReference,
    LifecycleReviewInputRole,
    MissionLifecycleReview,
)
from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_mission.io import (
    format_mission_lifecycle_summary,
    load_mission_lifecycle_result,
    load_mission_lifecycle_scenario,
    write_mission_artifact_bundle,
    write_mission_lifecycle_result,
)
from astro_mission.runner import run_mission_lifecycle
from astro_uq.cli import (
    SOFTWARE_COMPATIBILITY,
    _bind_resolved_dependencies,
    _runtime,
)
from astro_uq.io import CampaignArtifactStore, CampaignIOError, load_campaign_definition
from astro_uq.models import CampaignDefinition, CampaignState
from astro_uq.runner import run_campaign

_MANIFEST = "manifest.json"
_CLAIM_BOUNDARIES = {
    "lifecycle": "deterministic_design_screening_not_operational_authority",
    "assurance": "deterministic_lifecycle_review_not_causal_probabilistic_or_operational_authority",
}


class MissionEvidencePackSpec(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["mission_evidence_pack_v1"] = "mission_evidence_pack_v1"
    pack_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    lifecycle_scenario: str = Field(min_length=1)
    uncertainty_campaign: str = Field(min_length=1)


class EvidenceArtifact(AstroModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MissionEvidenceManifest(AstroModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    workflow: Literal["mission_evidence_pack_v1"] = "mission_evidence_pack_v1"
    pack_id: str
    location_bound: bool = False
    pack_root: str = Field(min_length=1)
    lifecycle_scenario_id: str
    deterministic_disposition: str
    uncertainty_state: CampaignState
    uncertainty_requested_samples: int = Field(gt=0)
    uncertainty_completed_samples: int = Field(ge=0)
    uncertainty_requirement_fractions: dict[str, float]
    claim_boundaries: dict[str, str]
    artifacts: tuple[EvidenceArtifact, ...]

    @model_validator(mode="after")
    def location_contract_must_match_schema(self) -> MissionEvidenceManifest:
        expected = self.schema_version == "1.0"
        if self.location_bound != expected:
            raise ValueError("mission evidence location binding must match schema version")
        return self


def load_mission_evidence_pack_spec(path: str | Path) -> MissionEvidencePackSpec:
    source = Path(path)
    try:
        payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
        return MissionEvidencePackSpec.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(f"could not load mission evidence pack {source}: {exc}") from exc


def run_mission_evidence_pack(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 1,
) -> MissionEvidenceManifest:
    source = Path(spec_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise InvalidScenarioError("mission evidence output directory must be new or empty")
    spec = load_mission_evidence_pack_spec(source)
    lifecycle_source = _resolve(source, spec.lifecycle_scenario)
    campaign_source = _resolve(source, spec.uncertainty_campaign)
    lifecycle_scenario = load_mission_lifecycle_scenario(lifecycle_source)
    campaign = _load_bound_campaign(campaign_source)
    if campaign.workflow.kind != "mission_lifecycle":
        raise InvalidScenarioError(
            "mission evidence uncertainty campaign must use mission_lifecycle"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        captured = _capture_inputs(
            staging, destination, source, lifecycle_source, campaign_source, lifecycle_scenario
        )
        result = run_mission_lifecycle(lifecycle_scenario)
        result_path = staging / "lifecycle/result.json"
        result_path.parent.mkdir(parents=True)
        write_mission_lifecycle_result(result_path, result)
        (staging / "lifecycle/summary.txt").write_text(
            format_mission_lifecycle_summary(result) + "\n", encoding="utf-8"
        )
        write_mission_artifact_bundle(staging / "lifecycle/artifacts", result)

        source_review = review_mission_lifecycle(result_path, lifecycle_source)
        review = _portable_review(
            source_review,
            destination,
            captured,
            sha256(
                (staging / f"inputs/lifecycle{lifecycle_source.suffix}").read_bytes()
            ).hexdigest(),
        )
        review_path = staging / "assurance/review.json"
        review_path.parent.mkdir(parents=True)
        write_mission_lifecycle_review(review_path, review)
        write_mission_lifecycle_review_summary(
            staging / "assurance/summary.txt", format_mission_lifecycle_review(review)
        )

        runtime = _runtime(campaign, campaign_source)
        campaign_result = run_campaign(
            campaign,
            runtime,
            output_dir=staging / "uncertainty",
            software_compatibility=SOFTWARE_COMPATIBILITY,
            workers=workers,
            runtime_factory=(lambda: _runtime(campaign, campaign_source)) if workers > 1 else None,
        )
        manifest = _build_manifest(
            staging,
            destination,
            spec,
            result.scenario_id,
            review.disposition.value,
            campaign_result.state,
            campaign_result.statistics.requested_samples,
            campaign_result.statistics.completed_samples,
            campaign_result.statistics.requirement_probabilities,
            campaign.evaluator.claim_boundary,
        )
        (staging / _MANIFEST).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_mission_evidence_pack(output_dir: str | Path) -> MissionEvidenceManifest:
    root = Path(output_dir).resolve()
    try:
        manifest = MissionEvidenceManifest.model_validate_json(
            (root / _MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"mission evidence manifest is missing or invalid: {exc}"
        ) from exc
    if manifest.location_bound and Path(manifest.pack_root) != root:
        raise InvalidScenarioError("mission evidence pack moved from its bound location")
    pack_paths = list(root.rglob("*"))
    for path in pack_paths:
        if path.is_symlink():
            raise InvalidScenarioError(
                f"mission evidence pack may not contain symbolic links: "
                f"{path.relative_to(root).as_posix()}"
            )
        if path.exists() and not path.resolve().is_relative_to(root):
            raise InvalidScenarioError(
                f"mission evidence artifact resolves outside pack: "
                f"{path.relative_to(root).as_posix()}"
            )
    declared = {artifact.path for artifact in manifest.artifacts}
    actual = {
        path.relative_to(root).as_posix()
        for path in pack_paths
        if path.is_file() and path.name != _MANIFEST
    }
    if actual != declared:
        raise InvalidScenarioError("mission evidence artifact inventory does not match manifest")
    for artifact in manifest.artifacts:
        if sha256((root / artifact.path).read_bytes()).hexdigest() != artifact.sha256:
            raise InvalidScenarioError(
                f"mission evidence artifact digest mismatch: {artifact.path}"
            )
    if manifest.schema_version == "1.0":
        review = load_mission_lifecycle_review(root / "assurance/review.json")
        if verify_mission_lifecycle_review(root / "assurance/review.json") != review:
            raise InvalidScenarioError("mission evidence lifecycle review verification failed")
    else:
        _verify_relocated_lifecycle_review(root, manifest)
    campaign_manifest = json.loads((root / "uncertainty/campaign.json").read_text(encoding="utf-8"))
    definition = CampaignDefinition.model_validate(campaign_manifest["definition"])
    with CampaignArtifactStore(root / "uncertainty") as store:
        state = store.resume(
            definition,
            software_compatibility=campaign_manifest["software_compatibility"],
        )
    if state.state is not CampaignState.COMPLETED:
        raise CampaignIOError("mission evidence campaign is not completed")
    return manifest


def _resolve(owner: Path, configured: str) -> Path:
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate.resolve()
    for parent in (owner.parent, *owner.parents):
        resolved = parent / candidate
        if resolved.exists():
            return resolved.resolve()
    raise InvalidScenarioError(f"mission evidence input does not exist: {configured}")


def _verify_relocated_lifecycle_review(
    root: Path,
    manifest: MissionEvidenceManifest,
) -> None:
    review = load_mission_lifecycle_review(root / "assurance/review.json")
    _require_pack_path(
        review.result_path,
        manifest.pack_root,
        "lifecycle/result.json",
        label="review result",
    )
    lifecycle_artifact = _artifact_for_prefix(manifest, "inputs/lifecycle")
    _require_pack_path(
        review.scenario_path,
        manifest.pack_root,
        lifecycle_artifact.path,
        label="review scenario",
    )
    scenario_path = root / lifecycle_artifact.path
    scenario_bytes = scenario_path.read_bytes()
    if sha256(scenario_bytes).hexdigest() != review.scenario_digest:
        raise InvalidScenarioError("mission evidence review scenario digest mismatch")
    scenario = load_mission_lifecycle_scenario(scenario_path)
    roles: tuple[LifecycleReviewInputRole, ...] = (
        "launch_scenario",
        "twin_scenario",
        "reentry_scenario",
    )
    role_artifacts = {
        role: _artifact_for_prefix(manifest, f"inputs/{role}")
        for role in roles
    }
    reference_by_role = {reference.role: reference for reference in review.referenced_inputs}
    for role, artifact in role_artifacts.items():
        reference = reference_by_role[role]
        _require_pack_path(
            reference.path,
            manifest.pack_root,
            artifact.path,
            label=f"review {role}",
        )
        if sha256((root / artifact.path).read_bytes()).hexdigest() != reference.digest:
            raise InvalidScenarioError(f"mission evidence review {role} digest mismatch")
        _require_pack_path(
            str(getattr(scenario, role)),
            manifest.pack_root,
            artifact.path,
            label=f"captured lifecycle {role}",
        )

    rebound = scenario.model_copy(
        update={role: str(root / artifact.path) for role, artifact in role_artifacts.items()}
    )
    result_path = root / "lifecycle/result.json"
    stored_result = load_mission_lifecycle_result(result_path)
    reproduced = run_mission_lifecycle(rebound)
    if stored_result.model_dump(mode="json") != reproduced.model_dump(mode="json"):
        raise InvalidScenarioError(
            "mission evidence lifecycle result does not match relocated captured inputs"
        )

    with tempfile.TemporaryDirectory(prefix="astro-mission-evidence-review-") as directory:
        rebound_path = Path(directory) / "lifecycle.yaml"
        rebound_path.write_text(
            yaml.safe_dump(rebound.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        expected = review_mission_lifecycle(result_path, rebound_path)
    normalized_references = tuple(
        reference.model_copy(update={"path": reference_by_role[reference.role].path})
        for reference in expected.referenced_inputs
    )
    normalized = expected.model_copy(
        update={
            "result_path": review.result_path,
            "scenario_path": review.scenario_path,
            "scenario_digest": review.scenario_digest,
            "referenced_inputs": normalized_references,
        }
    )
    if normalized != review:
        raise InvalidScenarioError("mission evidence relocated lifecycle review does not match")


def _artifact_for_prefix(
    manifest: MissionEvidenceManifest,
    prefix: str,
) -> EvidenceArtifact:
    matches = [
        artifact
        for artifact in manifest.artifacts
        if artifact.path.startswith(prefix + ".")
    ]
    if len(matches) != 1:
        raise InvalidScenarioError(f"mission evidence requires exactly one {prefix} artifact")
    return matches[0]


def _require_pack_path(
    configured: str,
    creation_root: str,
    expected_relative: str,
    *,
    label: str,
) -> None:
    configured_path = Path(configured)
    creation_path = Path(creation_root)
    if not configured_path.is_absolute() or not creation_path.is_absolute():
        raise InvalidScenarioError(f"{label} must preserve an absolute creation path")
    try:
        relative = configured_path.relative_to(creation_path)
    except ValueError as exc:
        raise InvalidScenarioError(f"{label} escapes the recorded creation root") from exc
    if relative.as_posix() != expected_relative:
        raise InvalidScenarioError(f"{label} does not match the fixed pack layout")


def _load_bound_campaign(path: Path) -> CampaignDefinition:
    return _bind_resolved_dependencies(load_campaign_definition(path), path)


def _capture_inputs(
    staging: Path,
    destination: Path,
    spec_source: Path,
    lifecycle_source: Path,
    campaign_source: Path,
    scenario: Any,
) -> dict[str, Path]:
    inputs = staging / "inputs"
    inputs.mkdir(parents=True)
    sources = {
        "pack": spec_source,
        "lifecycle": lifecycle_source,
        "campaign": campaign_source,
        "launch_scenario": Path(scenario.launch_scenario).resolve(),
        "twin_scenario": Path(scenario.twin_scenario).resolve(),
        "reentry_scenario": Path(scenario.reentry_scenario).resolve(),
    }
    captured: dict[str, Path] = {}
    for role, source in sources.items():
        target = inputs / f"{role}{source.suffix}"
        shutil.copyfile(source, target)
        captured[role] = destination / target.relative_to(staging)
    portable = scenario.model_copy(
        update={
            role: str(captured[role])
            for role in ("launch_scenario", "twin_scenario", "reentry_scenario")
        }
    )
    (inputs / f"lifecycle{lifecycle_source.suffix}").write_text(
        yaml.safe_dump(portable.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return captured


def _portable_review(
    review: MissionLifecycleReview,
    destination: Path,
    captured: dict[str, Path],
    scenario_digest: str,
) -> MissionLifecycleReview:
    scenario_path = captured["lifecycle"]
    references = tuple(
        LifecycleReviewInputReference(
            role=reference.role,
            path=str(captured[reference.role]),
            digest=reference.digest,
        )
        for reference in review.referenced_inputs
    )
    return review.model_copy(
        update={
            "result_path": str(destination / "lifecycle/result.json"),
            "scenario_path": str(scenario_path),
            "scenario_digest": scenario_digest,
            "referenced_inputs": references,
        }
    )


def _build_manifest(
    staging: Path,
    destination: Path,
    spec: MissionEvidencePackSpec,
    scenario_id: str,
    disposition: str,
    campaign_state: CampaignState,
    requested_samples: int,
    completed_samples: int,
    requirement_fractions: dict[str, float],
    uncertainty_claim: str,
) -> MissionEvidenceManifest:
    artifacts = tuple(
        EvidenceArtifact(
            path=path.relative_to(staging).as_posix(),
            sha256=sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != _MANIFEST
    )
    return MissionEvidenceManifest(
        pack_id=spec.pack_id,
        pack_root=str(destination),
        lifecycle_scenario_id=scenario_id,
        deterministic_disposition=disposition,
        uncertainty_state=campaign_state,
        uncertainty_requested_samples=requested_samples,
        uncertainty_completed_samples=completed_samples,
        uncertainty_requirement_fractions=requirement_fractions,
        claim_boundaries={**_CLAIM_BOUNDARIES, "uncertainty": uncertainty_claim},
        artifacts=artifacts,
    )
