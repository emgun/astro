from __future__ import annotations

from pathlib import Path

import yaml

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
