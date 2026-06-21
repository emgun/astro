from pathlib import Path

from typer.testing import CliRunner

from astro_cli.main import app

runner = CliRunner(mix_stderr=False)


def test_assistant_ask_dry_run_prints_plan() -> None:
    result = runner.invoke(app, ["ask", "Run the local OD demo", "--dry-run"])

    assert result.exit_code == 0
    assert "local-od-demo" in result.stdout
    assert "estimate-measurements" in result.stdout


def test_assistant_ask_requires_approval_for_execution() -> None:
    result = runner.invoke(app, ["ask", "Run the local OD demo", "--execute"])

    assert result.exit_code == 2
    assert (
        "execution requires approval" in result.stdout
        or "execution requires approval" in result.stderr
    )


def test_assistant_ask_execute_wins_over_dry_run_flag() -> None:
    result = runner.invoke(
        app,
        ["ask", "Run the local OD demo", "--dry-run", "--execute"],
    )

    assert result.exit_code == 2
    assert (
        "execution requires approval" in result.stdout
        or "execution requires approval" in result.stderr
    )


def test_assistant_ask_writes_trace_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"

    result = runner.invoke(
        app,
        [
            "ask",
            "Run the local OD demo",
            "--dry-run",
            "--trace-output",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    assert trace_path.exists()
    assert "local-od-demo" in trace_path.read_text(encoding="utf-8")


def test_assistant_ask_unsupported_prompt_exits_with_planner_error() -> None:
    result = runner.invoke(app, ["ask", "Tune a launch vehicle", "--dry-run"])

    assert result.exit_code == 2
    assert "deterministic planner currently supports the local OD demo only" in result.stderr
