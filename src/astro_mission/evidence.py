from __future__ import annotations

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError

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
    MissionLifecycleReview,
)
from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_mission.io import (
    format_mission_lifecycle_summary,
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
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["mission_evidence_pack_v1"] = "mission_evidence_pack_v1"
    pack_id: str
    location_bound: Literal[True] = True
    pack_root: str = Field(min_length=1)
    lifecycle_scenario_id: str
    deterministic_disposition: str
    uncertainty_state: CampaignState
    uncertainty_requested_samples: int = Field(gt=0)
    uncertainty_completed_samples: int = Field(ge=0)
    uncertainty_requirement_fractions: dict[str, float]
    claim_boundaries: dict[str, str]
    artifacts: tuple[EvidenceArtifact, ...]


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
    if Path(manifest.pack_root) != root:
        raise InvalidScenarioError("mission evidence pack moved from its bound location")
    declared = {artifact.path for artifact in manifest.artifacts}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != _MANIFEST
    }
    if actual != declared:
        raise InvalidScenarioError("mission evidence artifact inventory does not match manifest")
    for artifact in manifest.artifacts:
        if sha256((root / artifact.path).read_bytes()).hexdigest() != artifact.sha256:
            raise InvalidScenarioError(
                f"mission evidence artifact digest mismatch: {artifact.path}"
            )
    review = load_mission_lifecycle_review(root / "assurance/review.json")
    if verify_mission_lifecycle_review(root / "assurance/review.json") != review:
        raise InvalidScenarioError("mission evidence lifecycle review verification failed")
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
