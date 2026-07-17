from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from astro_core.errors import InvalidScenarioError
from astro_mission.evidence import (
    run_mission_evidence_pack,
    verify_mission_evidence_pack,
)


def _one_case_spec(tmp_path: Path) -> Path:
    campaign = yaml.safe_load(
        Path("examples/campaigns/leo_lifecycle_robustness.yaml").read_text(encoding="utf-8")
    )
    campaign["sampler"]["samples"] = 1
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
    spec = {
        "schema_version": "1.0",
        "workflow": "mission_evidence_pack_v1",
        "pack_id": "test-mission-evidence-v1",
        "lifecycle_scenario": str(Path("examples/lifecycle/leo_round_trip.yaml").resolve()),
        "uncertainty_campaign": str(campaign_path),
    }
    spec_path = tmp_path / "pack.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec_path


def test_run_and_verify_mission_evidence_pack(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    manifest = run_mission_evidence_pack(_one_case_spec(tmp_path), output)

    assert manifest.uncertainty_state.value == "completed"
    assert manifest.uncertainty_requested_samples == 1
    assert manifest.uncertainty_completed_samples == 1
    assert manifest.uncertainty_requirement_fractions
    assert (output / "lifecycle/result.json").exists()
    assert (output / "assurance/review.json").exists()
    assert (output / "uncertainty/campaign.json").exists()
    assert verify_mission_evidence_pack(output) == manifest

    relocated = tmp_path / "relocated" / "evidence"
    relocated.parent.mkdir()
    shutil.copytree(output, relocated)
    before = _pack_bytes(relocated)
    assert verify_mission_evidence_pack(relocated) == manifest
    assert _pack_bytes(relocated) == before


def test_mission_evidence_verifier_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    run_mission_evidence_pack(_one_case_spec(tmp_path), output)
    result = output / "lifecycle/result.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["warnings"].append("tampered")
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="digest mismatch"):
        verify_mission_evidence_pack(output)


def test_mission_evidence_output_must_be_new_or_empty(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "owned.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="new or empty"):
        run_mission_evidence_pack(_one_case_spec(tmp_path), output)

    assert (output / "owned.txt").read_text(encoding="utf-8") == "keep"


def test_mission_evidence_relocation_rejects_recorded_path_escape(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    run_mission_evidence_pack(_one_case_spec(tmp_path), output)
    review_path = output / "assurance/review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["result_path"] = str(tmp_path / "outside.json")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    _refresh_manifest_digest(output, "assurance/review.json")

    with pytest.raises(InvalidScenarioError, match="escapes the recorded creation root"):
        verify_mission_evidence_pack(output)


def test_legacy_mission_evidence_pack_remains_location_bound(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    run_mission_evidence_pack(_one_case_spec(tmp_path), output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.0"
    manifest["location_bound"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_mission_evidence_pack(output).schema_version == "1.0"

    relocated = tmp_path / "legacy-relocated"
    output.rename(relocated)
    with pytest.raises(InvalidScenarioError, match="moved from its bound location"):
        verify_mission_evidence_pack(relocated)


@pytest.mark.parametrize(
    "artifact_path",
    [
        "lifecycle/result.json",
        "inputs/lifecycle.yaml",
        "inputs/launch_scenario.yaml",
        "inputs/twin_scenario.yaml",
        "inputs/reentry_scenario.yaml",
    ],
)
def test_mission_evidence_rejects_symlinked_artifacts(
    tmp_path: Path,
    artifact_path: str,
) -> None:
    output = tmp_path / "evidence"
    run_mission_evidence_pack(_one_case_spec(tmp_path), output)
    artifact = output / artifact_path
    external = tmp_path / f"external-{artifact.name}"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)

    with pytest.raises(InvalidScenarioError, match="may not contain symbolic links"):
        verify_mission_evidence_pack(output)


def _refresh_manifest_digest(output: Path, artifact_path: str) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256((output / artifact_path).read_bytes()).hexdigest()
    for artifact in manifest["artifacts"]:
        if artifact["path"] == artifact_path:
            artifact["sha256"] = digest
            break
    else:
        raise AssertionError(f"missing manifest artifact {artifact_path}")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _pack_bytes(output: Path) -> dict[str, bytes]:
    return {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
