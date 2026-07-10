from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from astro_cli.main import app


def test_run_mission_lifecycle_writes_result_summary_and_artifacts(tmp_path: Path) -> None:
    directory = tmp_path
    output = directory / "lifecycle.json"
    summary = directory / "summary.txt"
    artifacts = directory / "artifacts"

    result = CliRunner().invoke(
        app,
        [
            "run-mission-lifecycle",
            "examples/lifecycle/leo_round_trip.yaml",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--artifacts-dir",
            str(artifacts),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert "Status: pass" in summary.read_text(encoding="utf-8")
    assert (artifacts / "manifest.json").exists()
