from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from astro_cli.main import app
from astro_uq.io import load_campaign_definition

MANIFEST_PATH = Path("examples/campaigns/benchmark-manifest.yaml")
EXPECTED_WORKFLOWS = {
    "orbit",
    "digital_twin",
    "constellation",
    "reentry",
    "mission_lifecycle",
}
REQUIRED_MEASUREMENTS = {
    "repeated_wall_time_s",
    "evaluator_time_s",
    "serialization_time_s",
    "metric_extraction_time_s",
    "peak_rss_bytes",
    "artifact_bytes",
}


def test_campaign_benchmark_manifest_contract() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0"
    assert manifest["warmup_runs"] >= 1
    assert set(manifest["measurements"]) == REQUIRED_MEASUREMENTS
    assert manifest["machine_metadata"]
    assert manifest["runtime_metadata"]

    entries = manifest["entries"]
    assert {entry["workflow"] for entry in entries} == EXPECTED_WORKFLOWS
    for entry in entries:
        assert entry["workflow"]
        assert isinstance(entry["command"], list) and entry["command"][0] == "astro"
        assert entry["repetitions"] >= 3
        assert entry["expected_artifact"]
        assert entry["claim_boundary"].strip()

        example_paths = [Path(token) for token in entry["command"] if token.startswith("examples/")]
        assert example_paths, f"{entry['workflow']} must use a checked example"
        assert all(path.is_file() for path in example_paths)

    teacher_policy = manifest["optional_teacher_policy"]
    assert set(teacher_policy["backends"]) == {"orekit", "tudat"}
    assert teacher_policy["evidence_scope"] == "machine_scoped"

    selection = manifest["surrogate_selection"]
    assert selection["first_candidate"]
    assert selection["challenger"]
    assert selection["kill_decision"]
    assert len(selection["proceed_only_if"]) >= 3

    gate = manifest["profiling_gate"]
    assert gate["warmup_runs"] >= 1
    assert gate["minimum_measured_cases"] >= 5
    assert 0.0 < gate["minimum_evaluation_share_of_instrumented_time"] <= 1.0
    assert gate["minimum_median_evaluation_time_s"] > 0.0
    assert gate["evidence_scope"] == "machine_scoped"
    assert Path(gate["candidate_campaign"]).is_file()
    assert Path(gate["challenger_campaign"]).is_file()
    assert len(gate["proceed_only_if"]) >= 3
    candidate = load_campaign_definition(gate["candidate_campaign"])
    challenger = load_campaign_definition(gate["challenger_campaign"])
    assert candidate.sampler.samples >= gate["minimum_measured_cases"]
    assert challenger.sampler.samples >= gate["minimum_measured_cases"]
    assert candidate.metadata["purpose"] == "surrogate_gate_timing_candidate"
    assert challenger.metadata["purpose"] == "surrogate_gate_timing_challenger"
    assert "deterministic_timing" not in candidate.evaluator.claim_boundary
    assert "deterministic_timing" not in challenger.evaluator.claim_boundary


def test_profiling_campaigns_validate_outside_repository_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    campaign_paths = (
        Path("examples/campaigns/profile_orbit_j2.yaml").resolve(),
        Path("examples/campaigns/profile_digital_twin.yaml").resolve(),
    )
    monkeypatch.chdir(tmp_path)

    for campaign_path in campaign_paths:
        result = CliRunner().invoke(app, ["validate-campaign", str(campaign_path)])
        assert result.exit_code == 0, result.output
