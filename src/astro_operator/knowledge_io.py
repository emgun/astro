"""Publication and offline verification for mission knowledge graph bundles."""

from __future__ import annotations

import json
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_operator.conditional_campaign import ConditionalCampaignOutcome
from astro_operator.conditional_campaign_io import verify_conditional_campaign
from astro_operator.director import MissionDesignRun
from astro_operator.director_io import verify_mission_design_director
from astro_operator.io import verify_operator_run
from astro_operator.knowledge import (
    KnowledgeSourceKind,
    MissionKnowledgeGraph,
    MissionKnowledgeGraphSpec,
    VerifiedKnowledgeSource,
    build_mission_knowledge_graph,
)
from astro_operator.models import OperatorRun


class KnowledgeArtifact(AstroModel):
    path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class MissionKnowledgeManifest(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["mission_knowledge_graph_v1"] = "mission_knowledge_graph_v1"
    artifacts: tuple[KnowledgeArtifact, ...] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def artifacts_must_be_canonical(self) -> MissionKnowledgeManifest:
        paths = [item.path for item in self.artifacts]
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise ValueError("knowledge artifacts must be unique and path-sorted")
        return self


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_mission_knowledge_graph_spec(
    path: Path | str,
) -> MissionKnowledgeGraphSpec:
    spec_path = Path(path)
    try:
        raw: Any = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        return MissionKnowledgeGraphSpec.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load mission knowledge graph spec {spec_path}: {exc}"
        ) from exc


def publish_mission_knowledge_graph(
    spec_path: Path | str,
    output_dir: Path | str,
) -> MissionKnowledgeGraph:
    """Capture verified source bundles and publish their deterministic read model."""

    declared_spec_path = Path(spec_path)
    output = Path(output_dir)
    if output.exists():
        raise InvalidScenarioError("mission knowledge output directory must not already exist")
    spec = load_mission_knowledge_graph_spec(declared_spec_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        inputs = partial / "inputs"
        inputs.mkdir()
        (inputs / "graph-spec.yaml").write_bytes(declared_spec_path.read_bytes())
        for source in spec.sources:
            declared = Path(source.path)
            source_root = (
                declared if declared.is_absolute() else declared_spec_path.parent / declared
            )
            if source_root.is_symlink():
                raise InvalidScenarioError(
                    "Mission knowledge source root must not be a symbolic link"
                )
            source_root = source_root.resolve()
            _validate_source_tree(source_root)
            shutil.copytree(source_root, partial / "sources" / source.source_id)
        verified = _verify_captured_sources(partial, spec)
        graph = build_mission_knowledge_graph(spec, verified)
        (partial / "mission-knowledge-graph.json").write_text(
            graph.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        write_mission_knowledge_manifest(partial)
        partial.replace(output)
        return graph
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def write_mission_knowledge_manifest(root: Path) -> MissionKnowledgeManifest:
    artifacts = tuple(
        KnowledgeArtifact(
            path=path.relative_to(root).as_posix(),
            role=_artifact_role(path.relative_to(root).as_posix()),
            sha256=sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "knowledge-manifest.json"
    )
    payload = {
        "schema_version": "1.0",
        "workflow": "mission_knowledge_graph_v1",
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
    }
    manifest = MissionKnowledgeManifest.model_validate(
        {**payload, "manifest_sha256": _canonical_digest(payload)}
    )
    (root / "knowledge-manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_mission_knowledge_graph(
    root: Path | str,
) -> MissionKnowledgeGraph:
    """Verify captured sources and reconstruct the stored graph from them."""

    bundle_root = Path(root)
    if bundle_root.is_symlink():
        raise InvalidScenarioError("Mission knowledge root must not be a symbolic link")
    manifest_path = bundle_root / "knowledge-manifest.json"
    graph_path = bundle_root / "mission-knowledge-graph.json"
    spec_path = bundle_root / "inputs" / "graph-spec.yaml"
    for path in (manifest_path, graph_path, spec_path):
        if path.is_symlink():
            raise InvalidScenarioError(
                f"Mission knowledge artifact must not be a symbolic link: {path.name}"
            )
    try:
        manifest = MissionKnowledgeManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load mission knowledge manifest {manifest_path}: {exc}"
        ) from exc
    payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if manifest.manifest_sha256 != _canonical_digest(payload):
        raise InvalidScenarioError("Mission knowledge manifest digest mismatch")
    expected_paths = {item.path for item in manifest.artifacts}
    actual_paths: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise InvalidScenarioError(
                "Mission knowledge bundle contains a symbolic link: "
                f"{path.relative_to(bundle_root)}"
            )
        if path.is_file():
            if path != manifest_path:
                actual_paths.add(path.relative_to(bundle_root).as_posix())
            continue
        if path.is_dir():
            continue
        raise InvalidScenarioError(
            "Mission knowledge bundle contains an unsupported entry: "
            f"{path.relative_to(bundle_root)}"
        )
    if actual_paths != expected_paths:
        raise InvalidScenarioError("Mission knowledge artifact inventory does not match the bundle")
    for artifact in manifest.artifacts:
        relative = Path(artifact.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise InvalidScenarioError("Mission knowledge artifact path escapes the bundle")
        path = bundle_root / relative
        if artifact.role != _artifact_role(artifact.path):
            raise InvalidScenarioError(f"Mission knowledge artifact role mismatch: {artifact.path}")
        if path.stat().st_size != artifact.size_bytes:
            raise InvalidScenarioError(f"Mission knowledge artifact size mismatch: {artifact.path}")
        if sha256(path.read_bytes()).hexdigest() != artifact.sha256:
            raise InvalidScenarioError(
                f"Mission knowledge artifact digest mismatch: {artifact.path}"
            )
    spec = load_mission_knowledge_graph_spec(spec_path)
    verified = _verify_captured_sources(bundle_root, spec)
    try:
        stored = MissionKnowledgeGraph.model_validate_json(graph_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load mission knowledge graph {graph_path}: {exc}"
        ) from exc
    graph_payload = stored.model_dump(mode="json", exclude={"graph_sha256"})
    if stored.graph_sha256 != _canonical_digest(graph_payload):
        raise InvalidScenarioError("Mission knowledge graph digest mismatch")
    expected = build_mission_knowledge_graph(
        spec,
        verified,
        schema_version=stored.schema_version,
    )
    if stored != expected:
        raise InvalidScenarioError(
            "Mission knowledge graph does not match its verified source bundles"
        )
    return stored


def _verify_captured_sources(
    root: Path,
    spec: MissionKnowledgeGraphSpec,
) -> dict[str, VerifiedKnowledgeSource]:
    verified: dict[str, VerifiedKnowledgeSource] = {}
    for source in spec.sources:
        source_root = root / "sources" / source.source_id
        _validate_source_tree(source_root)
        tree_sha256 = _source_tree_digest(source_root)
        try:
            if source.kind == KnowledgeSourceKind.MISSION_DESIGN_DIRECTOR:
                primary: MissionDesignRun | OperatorRun | ConditionalCampaignOutcome = (
                    verify_mission_design_director(source_root)
                )
                operator = verify_operator_run(source_root / "operator")
            elif source.kind == KnowledgeSourceKind.OPERATOR_RUN:
                primary = verify_operator_run(source_root)
                operator = None
            else:
                primary = verify_conditional_campaign(source_root)
                operator = None
        except (InvalidScenarioError, OSError, ValueError) as exc:
            raise InvalidScenarioError(
                f"Mission knowledge source {source.source_id!r} failed verification: {exc}"
            ) from exc
        verified[source.source_id] = VerifiedKnowledgeSource(
            primary=primary,
            nested_operator=operator,
            source_tree_sha256=tree_sha256,
        )
    return verified


def _validate_source_tree(root: Path) -> None:
    if not root.is_dir():
        raise InvalidScenarioError(f"Mission knowledge source is not a directory: {root}")
    if root.is_symlink():
        raise InvalidScenarioError("Mission knowledge source root must not be a symbolic link")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise InvalidScenarioError(
                f"Mission knowledge source contains a symbolic link: {path.relative_to(root)}"
            )
        if not path.is_dir() and not path.is_file():
            raise InvalidScenarioError(
                f"Mission knowledge source contains an unsupported entry: {path}"
            )


def _source_tree_digest(root: Path) -> str:
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return _canonical_digest(inventory)


def _artifact_role(path: str) -> str:
    if path == "inputs/graph-spec.yaml":
        return "declared_graph_contract"
    if path == "mission-knowledge-graph.json":
        return "derived_knowledge_read_model"
    if path.startswith("sources/"):
        return "captured_verified_source"
    return "supporting_artifact"


__all__ = [
    "KnowledgeArtifact",
    "MissionKnowledgeManifest",
    "load_mission_knowledge_graph_spec",
    "publish_mission_knowledge_graph",
    "verify_mission_knowledge_graph",
    "write_mission_knowledge_manifest",
]
