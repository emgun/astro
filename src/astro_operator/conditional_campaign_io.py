"""Execution, publication, and offline verification for conditional campaigns."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.runner import resolve_lifecycle_twin_scenario
from astro_operator.conditional_campaign import (
    ConditionalCampaignBinding,
    ConditionalCampaignExecutionSpec,
    ConditionalCampaignOutcome,
    build_conditional_campaign_outcome,
    canonical_digest,
    select_conditional_decision,
    validate_campaign_contract,
)
from astro_operator.director import MissionDesignRun
from astro_operator.director_io import verify_mission_design_director
from astro_twin.io import load_twin_scenario
from astro_uq.cli import SOFTWARE_COMPATIBILITY, _bind_resolved_dependencies, _runtime
from astro_uq.io import (
    CAMPAIGN_FILE,
    LOCK_FILE,
    TRANSACTION_FILE,
    CampaignArtifactStore,
    CampaignIOError,
    atomic_write_json,
    canonical_hash,
    load_campaign_definition,
)
from astro_uq.metrics import evaluate_requirements
from astro_uq.models import (
    CampaignDefinition,
    CampaignState,
    CampaignStatistics,
    CaseObservation,
    ParameterRealization,
    WorkflowSpec,
)
from astro_uq.parameters import model_digest
from astro_uq.runner import plan_campaign_samples, run_campaign
from astro_uq.statistics import summarize_campaign

_MANIFEST = "conditional-campaign-manifest.json"
_OUTCOME = "conditional-campaign-outcome.json"
_BINDING = "inputs/binding.json"
_BOUND_DEFINITION = "inputs/bound-campaign-definition.json"
_CAPTURED_SPEC = "inputs/execution-spec.yaml"
_CAPTURED_TEMPLATE = "inputs/campaign-template.yaml"
_SOURCE_DIRECTOR = "source/director"
_CAMPAIGN_ARTIFACTS = {
    "campaign/campaign.json",
    "campaign/cases.jsonl",
    "campaign/samples.jsonl",
    "campaign/statistics.json",
    "campaign/summary.txt",
}
_FIXED_INPUT_ARTIFACTS = {
    _BINDING,
    _BOUND_DEFINITION,
    _CAPTURED_SPEC,
    _CAPTURED_TEMPLATE,
    "inputs/execution-scenario.json",
    "inputs/execution-twin-scenario.json",
    "inputs/source-candidate-scenario.json",
}
_CAPTURED_INPUT_PREFIXES = (
    "inputs/launch_scenario.",
    "inputs/orbit-scenario.",
    "inputs/reentry_scenario.",
    "inputs/source-twin-scenario.",
)


@dataclass(frozen=True)
class _VerifiedCampaignEvidence:
    campaign_state: CampaignState
    samples: tuple[ParameterRealization, ...]
    cases: tuple[CaseObservation, ...]
    statistics: CampaignStatistics
    samples_sha256: str
    cases_sha256: str
    statistics_sha256: str


class ConditionalCampaignArtifact(AstroModel):
    path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ConditionalCampaignManifest(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["conditional_mission_design_campaign_v1"] = (
        "conditional_mission_design_campaign_v1"
    )
    execution_id: str = Field(min_length=1)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["retain", "revise", "abstain"]
    artifacts: tuple[ConditionalCampaignArtifact, ...] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def inventory_must_be_canonical(self) -> ConditionalCampaignManifest:
        paths = [item.path for item in self.artifacts]
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise ValueError("conditional campaign inventory must be unique and sorted")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != canonical_digest(payload):
            raise ValueError("conditional campaign manifest digest mismatch")
        return self


def load_conditional_campaign_spec(
    path: str | Path,
) -> ConditionalCampaignExecutionSpec:
    source = Path(path)
    try:
        payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
        return ConditionalCampaignExecutionSpec.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"could not load conditional campaign spec {source}: {exc}"
        ) from exc


def run_conditional_campaign(
    *,
    director_root: str | Path,
    spec_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
    workers: int = 1,
) -> ConditionalCampaignOutcome:
    source_root = _checked_path(director_root, label="Director root")
    source_spec = _checked_path(spec_path, label="execution spec")
    root = _checked_path(output_dir, label="conditional campaign output")
    if workers < 1:
        raise InvalidScenarioError("conditional campaign workers must be at least one")
    if resume:
        if not root.is_dir() or (root / _MANIFEST).exists():
            raise InvalidScenarioError(
                "resume requires an incomplete conditional campaign directory"
            )
        verify_mission_design_director(source_root)
        if (
            source_spec.read_bytes() != (root / _CAPTURED_SPEC).read_bytes()
            or (source_root / "director-manifest.json").read_bytes()
            != (root / _SOURCE_DIRECTOR / "director-manifest.json").read_bytes()
        ):
            raise InvalidScenarioError(
                "resume inputs do not match the captured conditional campaign"
            )
        spec = load_conditional_campaign_spec(root / _CAPTURED_SPEC)
        director = verify_mission_design_director(root / _SOURCE_DIRECTOR)
        definition = CampaignDefinition.model_validate_json(
            (root / _BOUND_DEFINITION).read_text(encoding="utf-8")
        )
        if definition != _expected_bound_definition(root, spec):
            raise InvalidScenarioError(
                "captured campaign definition does not derive from its template"
            )
        binding = ConditionalCampaignBinding.model_validate_json(
            (root / _BINDING).read_text(encoding="utf-8")
        )
        expected = _build_binding(root, spec, director, definition)
        if binding != expected:
            raise InvalidScenarioError("conditional campaign resume binding mismatch")
    else:
        if root.exists():
            raise InvalidScenarioError(
                "conditional campaign output directory must not already exist"
            )
        verify_mission_design_director(source_root)
        spec = load_conditional_campaign_spec(source_spec)
        root.mkdir(parents=True)
        shutil.copytree(source_root, root / _SOURCE_DIRECTOR)
        inputs = root / "inputs"
        inputs.mkdir()
        shutil.copyfile(source_spec, root / _CAPTURED_SPEC)
        template_source = _resolve(source_spec, spec.campaign_template_path)
        shutil.copyfile(template_source, root / _CAPTURED_TEMPLATE)
        director = verify_mission_design_director(root / _SOURCE_DIRECTOR)
        _capture_execution_scenario(root, director)
        template = load_campaign_definition(root / _CAPTURED_TEMPLATE)
        preliminary = CampaignDefinition.model_validate(
            template.model_copy(
                update={
                    "campaign_id": f"{spec.execution_id}-campaign",
                    "workflow": WorkflowSpec(
                        kind="mission_lifecycle",
                        scenario="execution-scenario.json",
                    ),
                }
            ).model_dump(mode="python")
        )
        atomic_write_json(root / _BOUND_DEFINITION, preliminary)
        definition = _bind_resolved_dependencies(
            preliminary,
            root / _BOUND_DEFINITION,
        )
        if definition != _expected_bound_definition(root, spec):
            raise InvalidScenarioError(
                "bound campaign definition does not match captured dependencies"
            )
        atomic_write_json(root / _BOUND_DEFINITION, definition)
        validate_campaign_contract(director, spec, definition)
        binding = _build_binding(root, spec, director, definition)
        atomic_write_json(root / _BINDING, binding)

    validate_campaign_contract(director, spec, definition)
    _verify_captured_execution_scenario(root, director)
    runtime = _runtime(definition, root / _BOUND_DEFINITION)
    run_campaign(
        definition,
        runtime,
        output_dir=root / "campaign",
        software_compatibility=SOFTWARE_COMPATIBILITY,
        resume=resume,
        workers=workers,
        runtime_factory=(
            partial(_runtime, definition, root / _BOUND_DEFINITION) if workers > 1 else None
        ),
    )
    (root / "campaign" / LOCK_FILE).unlink(missing_ok=True)
    evidence = _verify_campaign_evidence(root / "campaign", definition)
    outcome = build_conditional_campaign_outcome(
        spec=spec,
        binding=binding,
        definition=definition,
        campaign_state=evidence.campaign_state,
        samples=evidence.samples,
        cases=evidence.cases,
        statistics=evidence.statistics,
        samples_sha256=evidence.samples_sha256,
        cases_sha256=evidence.cases_sha256,
        statistics_sha256=evidence.statistics_sha256,
    )
    atomic_write_json(root / _OUTCOME, outcome)
    _write_manifest(root, spec, binding, outcome)
    return outcome


def verify_conditional_campaign(
    output_dir: str | Path,
) -> ConditionalCampaignOutcome:
    root = _checked_path(output_dir, label="conditional campaign root")
    try:
        manifest = ConditionalCampaignManifest.model_validate_json(
            (root / _MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"conditional campaign manifest is missing or invalid: {exc}"
        ) from exc
    _verify_inventory(root, manifest)
    spec = load_conditional_campaign_spec(root / _CAPTURED_SPEC)
    director = verify_mission_design_director(root / _SOURCE_DIRECTOR)
    definition = CampaignDefinition.model_validate_json(
        (root / _BOUND_DEFINITION).read_text(encoding="utf-8")
    )
    if definition != _expected_bound_definition(root, spec):
        raise InvalidScenarioError("captured campaign definition does not derive from its template")
    validate_campaign_contract(director, spec, definition)
    _verify_captured_execution_scenario(root, director)
    stored_binding = ConditionalCampaignBinding.model_validate_json(
        (root / _BINDING).read_text(encoding="utf-8")
    )
    expected_binding = _build_binding(root, spec, director, definition)
    if stored_binding != expected_binding:
        raise InvalidScenarioError("conditional campaign binding mismatch")
    evidence = _verify_campaign_evidence(root / "campaign", definition)
    expected = build_conditional_campaign_outcome(
        spec=spec,
        binding=stored_binding,
        definition=definition,
        campaign_state=evidence.campaign_state,
        samples=evidence.samples,
        cases=evidence.cases,
        statistics=evidence.statistics,
        samples_sha256=evidence.samples_sha256,
        cases_sha256=evidence.cases_sha256,
        statistics_sha256=evidence.statistics_sha256,
    )
    stored = ConditionalCampaignOutcome.model_validate_json(
        (root / _OUTCOME).read_text(encoding="utf-8")
    )
    if stored != expected:
        raise InvalidScenarioError(
            "conditional campaign outcome does not match verified campaign evidence"
        )
    if (
        manifest.execution_id != spec.execution_id
        or manifest.binding_sha256 != stored_binding.binding_sha256
        or manifest.outcome_sha256 != stored.outcome_sha256
        or manifest.disposition != stored.disposition
    ):
        raise InvalidScenarioError("conditional campaign manifest summary mismatch")
    return stored


def _build_binding(
    root: Path,
    spec: ConditionalCampaignExecutionSpec,
    director: MissionDesignRun,
    definition: CampaignDefinition,
) -> ConditionalCampaignBinding:
    decision = select_conditional_decision(director, spec)
    baseline = director.baseline
    assert baseline is not None
    payload = {
        "schema_version": "1.0",
        "execution_id": spec.execution_id,
        "execution_spec_sha256": sha256((root / _CAPTURED_SPEC).read_bytes()).hexdigest(),
        "director_manifest_sha256": sha256(
            (root / _SOURCE_DIRECTOR / "director-manifest.json").read_bytes()
        ).hexdigest(),
        "director_run_sha256": director.run_sha256,
        "baseline_id": baseline.baseline_id,
        "baseline_version": baseline.version,
        "baseline_sha256": baseline.baseline_sha256,
        "candidate_id": baseline.candidate_id,
        "rule_id": decision.rule_id,
        "conditional_decision_sha256": decision.decision_sha256,
        "capability_id": decision.capability_id,
        "campaign_template_sha256": sha256((root / _CAPTURED_TEMPLATE).read_bytes()).hexdigest(),
        "source_candidate_scenario_sha256": sha256(
            (root / "inputs/source-candidate-scenario.json").read_bytes()
        ).hexdigest(),
        "campaign_definition_digest": canonical_hash(definition),
    }
    return ConditionalCampaignBinding.model_validate(
        {**payload, "binding_sha256": canonical_digest(payload)}
    )


def _expected_bound_definition(
    root: Path,
    spec: ConditionalCampaignExecutionSpec,
) -> CampaignDefinition:
    template = load_campaign_definition(root / _CAPTURED_TEMPLATE)
    preliminary = CampaignDefinition.model_validate(
        template.model_copy(
            update={
                "campaign_id": f"{spec.execution_id}-campaign",
                "workflow": WorkflowSpec(
                    kind="mission_lifecycle",
                    scenario="execution-scenario.json",
                ),
            }
        ).model_dump(mode="python")
    )
    scenario = load_mission_lifecycle_scenario(root / "inputs/execution-scenario.json")
    twin = load_twin_scenario(root / "inputs/execution-twin-scenario.json")
    resolved_twin = resolve_lifecycle_twin_scenario(twin, scenario)
    metadata = {
        **preliminary.metadata,
        "resolved_dependencies": {
            "twin_scenario": scenario.twin_scenario,
            "twin_template_digest": model_digest(twin),
            "resolved_twin_scenario_digest": model_digest(resolved_twin),
        },
    }
    return CampaignDefinition.model_validate(
        preliminary.model_copy(update={"metadata": metadata}).model_dump(mode="python")
    )


def _capture_execution_scenario(root: Path, director: MissionDesignRun) -> None:
    baseline = director.baseline
    assert baseline is not None
    source = (
        root
        / _SOURCE_DIRECTOR
        / "operator"
        / "candidates"
        / baseline.candidate_id
        / "scenario.json"
    )
    inputs = root / "inputs"
    shutil.copyfile(source, inputs / "source-candidate-scenario.json")
    scenario = load_mission_lifecycle_scenario(source)
    for role in ("launch_scenario", "reentry_scenario"):
        role_source = Path(str(getattr(scenario, role))).resolve()
        shutil.copyfile(role_source, inputs / f"{role}{role_source.suffix}")
    twin_source = Path(scenario.twin_scenario).resolve()
    shutil.copyfile(twin_source, inputs / f"source-twin-scenario{twin_source.suffix}")
    twin = load_twin_scenario(twin_source)
    orbit_source = _resolve(twin_source, twin.orbit_scenario)
    shutil.copyfile(orbit_source, inputs / f"orbit-scenario{orbit_source.suffix}")
    execution_twin = twin.model_copy(
        update={"orbit_scenario": str(root / "inputs" / f"orbit-scenario{orbit_source.suffix}")}
    )
    atomic_write_json(inputs / "execution-twin-scenario.json", execution_twin)
    execution_scenario = scenario.model_copy(
        update={
            "launch_scenario": str(
                inputs / f"launch_scenario{Path(scenario.launch_scenario).suffix}"
            ),
            "twin_scenario": str(inputs / "execution-twin-scenario.json"),
            "reentry_scenario": str(
                inputs / f"reentry_scenario{Path(scenario.reentry_scenario).suffix}"
            ),
        }
    )
    atomic_write_json(inputs / "execution-scenario.json", execution_scenario)


def _verify_captured_execution_scenario(
    root: Path,
    director: MissionDesignRun,
) -> None:
    baseline = director.baseline
    assert baseline is not None
    director_source = (
        root
        / _SOURCE_DIRECTOR
        / "operator"
        / "candidates"
        / baseline.candidate_id
        / "scenario.json"
    )
    captured_source = root / "inputs/source-candidate-scenario.json"
    if director_source.read_bytes() != captured_source.read_bytes():
        raise InvalidScenarioError(
            "captured candidate scenario does not match the verified Director"
        )
    source = load_mission_lifecycle_scenario(captured_source)
    execution = load_mission_lifecycle_scenario(root / "inputs/execution-scenario.json")
    expected = source.model_copy(
        update={
            "launch_scenario": execution.launch_scenario,
            "twin_scenario": execution.twin_scenario,
            "reentry_scenario": execution.reentry_scenario,
        }
    )
    if execution != expected:
        raise InvalidScenarioError(
            "execution scenario differs from the candidate beyond captured paths"
        )
    source_twin_path = next((root / "inputs").glob("source-twin-scenario.*"))
    source_twin = load_twin_scenario(source_twin_path)
    execution_twin = load_twin_scenario(root / "inputs/execution-twin-scenario.json")
    if execution_twin != source_twin.model_copy(
        update={"orbit_scenario": execution_twin.orbit_scenario}
    ):
        raise InvalidScenarioError(
            "execution twin differs from its source beyond the captured orbit path"
        )


def _verify_campaign_evidence(
    campaign_root: Path,
    definition: CampaignDefinition,
) -> _VerifiedCampaignEvidence:
    if (campaign_root / TRANSACTION_FILE).exists():
        raise CampaignIOError("conditional campaign has an unresolved transaction")
    manifest = json.loads((campaign_root / CAMPAIGN_FILE).read_text(encoding="utf-8"))
    stored_definition = CampaignDefinition.model_validate(manifest["definition"])
    if stored_definition != definition:
        raise CampaignIOError("campaign manifest definition does not match its binding")
    with CampaignArtifactStore(campaign_root) as store:
        state = store.resume(
            definition,
            software_compatibility=manifest["software_compatibility"],
        )
    samples = tuple(ParameterRealization.model_validate(item) for item in state.samples)
    cases = tuple(CaseObservation.model_validate(item) for item in state.cases)
    for case in cases:
        if case.outcome_status.value == "success":
            expected_requirements = evaluate_requirements(
                case.metric_values,
                definition.requirements,
            )
            if case.requirements != expected_requirements:
                raise CampaignIOError("campaign requirements do not match captured metric values")
        elif case.metric_values or case.requirements:
            raise CampaignIOError("failed campaign cases must not contain metrics or requirements")
    statistics = CampaignStatistics.model_validate(state.statistics)
    _batch, planned = plan_campaign_samples(definition)
    if samples != planned:
        raise CampaignIOError("campaign samples do not match the deterministic sample plan")
    if tuple(item.sample_id for item in cases) != tuple(item.sample_id for item in planned):
        raise CampaignIOError("campaign cases do not match the planned sample order")
    expected_statistics = summarize_campaign(
        requested_samples=definition.sampler.samples,
        observations=cases,
        weights={item.sample_id: float(item.weight) for item in samples},
        requirement_ids=tuple(item.requirement_id for item in definition.requirements),
        convergence_history=(
            *(
                {
                    "completed_samples": index,
                    "reason": ("fixed_count_reached" if index == len(cases) else "continue"),
                }
                for index in range(1, len(cases) + 1)
            ),
        ),
    )
    if statistics != expected_statistics:
        raise CampaignIOError("campaign statistics do not match the captured cases")
    return _VerifiedCampaignEvidence(
        campaign_state=state.state,
        samples=samples,
        cases=cases,
        statistics=statistics,
        samples_sha256=str(manifest["samples_digest"]),
        cases_sha256=str(manifest["cases_digest"]),
        statistics_sha256=str(manifest["statistics_digest"]),
    )


def _write_manifest(
    root: Path,
    spec: ConditionalCampaignExecutionSpec,
    binding: ConditionalCampaignBinding,
    outcome: ConditionalCampaignOutcome,
) -> ConditionalCampaignManifest:
    artifacts = tuple(
        ConditionalCampaignArtifact(
            path=path.relative_to(root).as_posix(),
            role=_artifact_role(path.relative_to(root).as_posix()),
            sha256=sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / _MANIFEST and path != root / "campaign" / LOCK_FILE
    )
    payload = {
        "schema_version": "1.0",
        "workflow": "conditional_mission_design_campaign_v1",
        "execution_id": spec.execution_id,
        "binding_sha256": binding.binding_sha256,
        "outcome_sha256": outcome.outcome_sha256,
        "disposition": outcome.disposition,
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
    }
    manifest = ConditionalCampaignManifest.model_validate(
        {**payload, "manifest_sha256": canonical_digest(payload)}
    )
    atomic_write_json(root / _MANIFEST, manifest)
    return manifest


def _verify_inventory(
    root: Path,
    manifest: ConditionalCampaignManifest,
) -> None:
    declared = {item.path for item in manifest.artifacts}
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise InvalidScenarioError("conditional campaign bundle may not contain symbolic links")
        if path.is_file() and path != root / _MANIFEST and path != root / "campaign" / LOCK_FILE:
            actual.add(path.relative_to(root).as_posix())
    if actual != declared:
        raise InvalidScenarioError("conditional campaign artifact inventory does not match")
    for prefix in _CAPTURED_INPUT_PREFIXES:
        if sum(path.startswith(prefix) for path in declared) != 1:
            raise InvalidScenarioError(
                f"conditional campaign requires exactly one {prefix} artifact"
            )
    for artifact in manifest.artifacts:
        relative = Path(artifact.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise InvalidScenarioError("conditional campaign artifact path escapes the bundle")
        path = root / artifact.path
        if (
            _artifact_role(artifact.path) != artifact.role
            or path.stat().st_size != artifact.size_bytes
            or sha256(path.read_bytes()).hexdigest() != artifact.sha256
        ):
            raise InvalidScenarioError(f"conditional campaign artifact mismatch: {artifact.path}")


def _artifact_role(path: str) -> str:
    if path == _OUTCOME:
        return "derived_campaign_disposition"
    if path == _BINDING:
        return "director_campaign_binding"
    if path == _CAPTURED_SPEC:
        return "declared_execution_contract"
    if path == _CAPTURED_TEMPLATE:
        return "declared_campaign_template"
    if path.startswith(_SOURCE_DIRECTOR + "/"):
        return "verified_director_source"
    if path in _CAMPAIGN_ARTIFACTS:
        return "campaign_evidence"
    if path in _FIXED_INPUT_ARTIFACTS or path.startswith(_CAPTURED_INPUT_PREFIXES):
        return "captured_campaign_input"
    raise InvalidScenarioError(f"unrecognized conditional campaign artifact: {path}")


def _resolve(owner: Path, configured: str) -> Path:
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate.resolve()
    for parent in (owner.parent, *owner.parents):
        resolved = parent / candidate
        if resolved.exists():
            return resolved.resolve()
    raise InvalidScenarioError(f"conditional campaign input does not exist: {configured}")


def _checked_path(path: str | Path, *, label: str) -> Path:
    absolute = Path(path).absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise InvalidScenarioError(f"{label} must not traverse a symbolic link")
    return absolute.resolve()


__all__ = [
    "ConditionalCampaignArtifact",
    "ConditionalCampaignManifest",
    "load_conditional_campaign_spec",
    "run_conditional_campaign",
    "verify_conditional_campaign",
]
