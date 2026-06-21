import json
from pathlib import Path

from astro_assistant.models import ArtifactKind, WorkflowArtifact
from astro_assistant.validators import validate_artifact


def test_validate_json_artifact_requires_parseable_json(tmp_path: Path) -> None:
    path = tmp_path / "estimate.json"
    path.write_text(json.dumps({"converged": True}), encoding="utf-8")

    artifact = WorkflowArtifact(path=str(path), kind=ArtifactKind.ESTIMATE_JSON)

    assert validate_artifact(artifact) is True


def test_validate_required_artifact_fails_when_missing(tmp_path: Path) -> None:
    artifact = WorkflowArtifact(
        path=str(tmp_path / "missing.json"),
        kind=ArtifactKind.ESTIMATE_JSON,
    )

    assert validate_artifact(artifact) is False


def test_validate_optional_artifact_passes_when_missing(tmp_path: Path) -> None:
    artifact = WorkflowArtifact(
        path=str(tmp_path / "missing.json"),
        kind=ArtifactKind.ESTIMATE_JSON,
        required=False,
    )

    assert validate_artifact(artifact) is True


def test_validate_json_artifact_fails_when_json_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "estimate.json"
    path.write_text("{invalid", encoding="utf-8")

    artifact = WorkflowArtifact(path=str(path), kind=ArtifactKind.ESTIMATE_JSON)

    assert validate_artifact(artifact) is False


def test_validate_non_json_artifact_requires_only_existence(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text("not: [json", encoding="utf-8")

    artifact = WorkflowArtifact(path=str(path), kind=ArtifactKind.SCENARIO)

    assert validate_artifact(artifact) is True
