import json
from pathlib import Path

from typer.testing import CliRunner

from astro_cli.main import app

runner = CliRunner()


def test_validate_command_accepts_example_scenario() -> None:
    result = runner.invoke(app, ["validate", "examples/scenarios/leo_two_body.yaml"])

    assert result.exit_code == 0
    assert "leo-two-body" in result.stdout


def test_propagate_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "local",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-two-body"
    assert payload["backend"] == "local"
    assert len(payload["samples"]) == 11
