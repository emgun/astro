import json
from pathlib import Path

from astro_assistant.models import ArtifactKind, WorkflowArtifact


_JSON_KINDS = {
    ArtifactKind.MEASUREMENTS_JSON,
    ArtifactKind.ESTIMATE_JSON,
    ArtifactKind.TRACE_JSON,
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
    return True
