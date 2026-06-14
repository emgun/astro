from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_core.models import Scenario
from astro_dynamics.local import propagate_local

app = typer.Typer(help="Astro Suite flight dynamics workflows.")


def _load_scenario_or_exit(scenario_path: Path) -> Scenario:
    try:
        return load_scenario(scenario_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def validate(scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Validate a scenario file."""
    scenario = _load_scenario_or_exit(scenario_path)

    typer.echo(f"valid scenario: {scenario.scenario_id}")


@app.command()
def propagate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    backend: Annotated[str, typer.Option()] = "local",
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Propagate a scenario and write a trajectory product."""
    scenario = _load_scenario_or_exit(scenario_path)
    if backend != "local":
        typer.echo(f"unsupported propagation backend: {backend}", err=True)
        raise typer.Exit(code=2)

    trajectory = propagate_local(scenario)
    payload = trajectory.model_dump_json(indent=2)
    if output is None:
        typer.echo(payload)
    else:
        try:
            output.write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            typer.echo(f"could not write trajectory {output}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(f"wrote trajectory: {output}")
