import json
from pathlib import Path

from astro_assistant.models import ArtifactKind, WorkflowArtifact
from astro_assurance.review_comparison import verify_assurance_review_comparison
from astro_core.errors import InvalidScenarioError

_JSON_KINDS = {
    ArtifactKind.MEASUREMENTS_JSON,
    ArtifactKind.ESTIMATE_JSON,
    ArtifactKind.TRACE_JSON,
    ArtifactKind.ASSURANCE_REVIEW,
    ArtifactKind.ASSURANCE_REVIEW_COMPARISON,
}


def validate_artifact(artifact: WorkflowArtifact) -> bool:
    path = Path(artifact.path)
    if not path.exists():
        return not artifact.required
    if artifact.kind in _JSON_KINDS:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
    if artifact.kind is ArtifactKind.ASSURANCE_REVIEW_COMPARISON:
        try:
            verify_assurance_review_comparison(path)
        except (InvalidScenarioError, OSError, ValueError):
            return False
    return True
