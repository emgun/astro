from __future__ import annotations

import json
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
