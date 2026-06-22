from __future__ import annotations

import inspect

from typer.testing import CliRunner


def make_cli_runner() -> CliRunner:
    """Return a runner with separately captured stderr across Typer releases."""
    if "mix_stderr" in inspect.signature(CliRunner.__init__).parameters:
        return CliRunner(mix_stderr=False)
    return CliRunner()
