from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from astro_core.errors import InvalidScenarioError, NumericalConvergenceError
from astro_core.io import load_scenario
from astro_core.models import CartesianState, GroundStation, Scenario
from astro_dynamics.local import propagate_local
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import generate_synthetic_measurements

app = typer.Typer(help="Astro Suite flight dynamics workflows.")


def _load_scenario_or_exit(scenario_path: Path) -> Scenario:
    try:
        return load_scenario(scenario_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _write_text_or_exit(output: Path, payload: str, product_name: str) -> None:
    try:
        output.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        typer.echo(f"could not write {product_name} {output}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _offset_cartesian_state(
    state: CartesianState,
    position_delta_km: tuple[float, float, float],
    velocity_delta_km_s: tuple[float, float, float],
) -> CartesianState:
    return CartesianState(
        position_km=(
            state.position_km[0] + position_delta_km[0],
            state.position_km[1] + position_delta_km[1],
            state.position_km[2] + position_delta_km[2],
        ),
        velocity_km_s=(
            state.velocity_km_s[0] + velocity_delta_km_s[0],
            state.velocity_km_s[1] + velocity_delta_km_s[1],
            state.velocity_km_s[2] + velocity_delta_km_s[2],
        ),
    )


def _with_estimation_demo_geometry(scenario: Scenario) -> Scenario:
    if len(scenario.ground_stations) >= 2:
        return scenario

    stations = list(scenario.ground_stations)
    station_names = {station.name for station in stations}

    if not stations and "equator-eci" not in station_names:
        stations.append(
            GroundStation(
                name="equator-eci",
                position_eci_km=(6378.1363, 0.0, 0.0),
                frame=scenario.initial_state.frame,
                elevation_mask_deg=0.0,
            )
        )
        station_names.add("equator-eci")

    if "north-eci" not in station_names:
        stations.append(
            GroundStation(
                name="north-eci",
                position_eci_km=(0.0, 6378.1363, 0.0),
                frame=scenario.initial_state.frame,
                elevation_mask_deg=0.0,
            )
        )

    return scenario.model_copy(update={"ground_stations": stations})


def _with_estimation_demo_initial_guess(scenario: Scenario) -> Scenario:
    perturbed_cartesian = _offset_cartesian_state(
        scenario.initial_state.cartesian,
        position_delta_km=(1.0, -0.8, 0.6),
        velocity_delta_km_s=(0.0005, -0.001, 0.0008),
    )
    perturbed_initial_state = scenario.initial_state.model_copy(
        update={"cartesian": perturbed_cartesian}
    )
    return scenario.model_copy(update={"initial_state": perturbed_initial_state})


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
        _write_text_or_exit(output, payload, "trajectory")
        typer.echo(f"wrote trajectory: {output}")


@app.command()
def synth_measurements(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Generate synthetic measurements for a locally propagated scenario."""
    scenario = _load_scenario_or_exit(scenario_path)
    trajectory = propagate_local(scenario)
    measurements = generate_synthetic_measurements(scenario, trajectory)
    payload = json.dumps(
        {
            "scenario_id": scenario.scenario_id,
            "measurements": [record.model_dump(mode="json") for record in measurements],
        },
        indent=2,
    )

    _write_text_or_exit(output, payload, "measurements")
    typer.echo(f"wrote measurements: {output}")


@app.command()
def estimate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Run a local synthetic orbit-determination workflow."""
    source_scenario = _load_scenario_or_exit(scenario_path)
    truth_scenario = _with_estimation_demo_geometry(source_scenario)
    truth_trajectory = propagate_local(truth_scenario)
    measurements = generate_synthetic_measurements(truth_scenario, truth_trajectory)
    estimate_scenario = _with_estimation_demo_initial_guess(truth_scenario)

    try:
        result = estimate_initial_state(estimate_scenario, measurements)
    except NumericalConvergenceError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    _write_text_or_exit(output, result.model_dump_json(indent=2), "estimate")
    typer.echo(f"wrote estimate: {output}")
