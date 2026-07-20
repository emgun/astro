from __future__ import annotations

import json
import sqlite3
import stat
from pathlib import Path

from tests.astro_cli.helpers import make_cli_runner

SPEC = Path("examples/operator/supervised_simulated_burn.yaml")
REPLAY = Path("examples/operator/supervised_simulated_burn_replay.yaml")


def test_public_supervised_simulation_command_runs_and_verifies(tmp_path: Path) -> None:
    from astro_cli.main import app

    output = tmp_path / "supervised-simulation"
    command_store = tmp_path / "mission-command-ledger.sqlite3"
    run = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            str(SPEC),
            "--reasoner-replay",
            str(REPLAY),
            "--output-dir",
            str(output),
            "--command-store",
            str(command_store),
        ],
    )
    assert run.exit_code == 0, run.output

    verified = make_cli_runner().invoke(app, ["verify-mission-operator", str(output)])
    assert verified.exit_code == 0, verified.output
    payload = json.loads((output / "operator-run.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.2"
    assert payload["steps"][1]["command_execution_record"]["terminal"]["status"] == (
        "committed"
    )
    assert command_store.is_file()
    assert stat.S_IMODE(command_store.stat().st_mode) == 0o600

    second_output = tmp_path / "second-publication"
    replay = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            str(SPEC),
            "--reasoner-replay",
            str(REPLAY),
            "--output-dir",
            str(second_output),
            "--command-store",
            str(command_store),
        ],
    )
    assert replay.exit_code == 0, replay.output
    with sqlite3.connect(command_store) as connection:
        count = connection.execute("SELECT COUNT(*) FROM command_executions").fetchone()
    assert count == (1,)


def test_command_capable_cli_requires_cross_run_store(tmp_path: Path) -> None:
    from astro_cli.main import app

    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            str(SPEC),
            "--reasoner-replay",
            str(REPLAY),
            "--output-dir",
            str(tmp_path / "missing-ledger"),
        ],
    )

    assert result.exit_code == 2
    assert "durable --command-store" in result.stderr
