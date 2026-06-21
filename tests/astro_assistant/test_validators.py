import json

from astro_assistant.models import ArtifactKind, WorkflowArtifact
from astro_assistant.validators import validate_artifact


def test_validate_json_artifact_requires_parseable_json(tmp_path) -> None:
    path = tmp_path / "estimate.json"
    path.write_text(json.dumps({"converged": True}), encoding="utf-8")

    artifact = WorkflowArtifact(path=str(path), kind=ArtifactKind.ESTIMATE_JSON)

    assert validate_artifact(artifact) is True


def test_validate_required_artifact_fails_when_missing(tmp_path) -> None:
    artifact = WorkflowArtifact(path=str(tmp_path / "missing.json"), kind=ArtifactKind.ESTIMATE_JSON)

    assert validate_artifact(artifact) is False
