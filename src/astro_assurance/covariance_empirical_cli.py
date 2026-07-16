from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from astro_assurance.covariance_empirical import (
    run_empirical_covariance_campaign,
    write_empirical_covariance_artifact,
)
from astro_core.errors import InvalidScenarioError, UnsupportedBackendError


def run_empirical_covariance_campaign_command(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    predictor_trajectory_path: Annotated[
        Path, typer.Argument(exists=True, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    truth_backend: Annotated[str, typer.Option("--truth-backend")] = "tudat",
    samples: Annotated[int, typer.Option("--samples", min=2)] = 64,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 817,
) -> None:
    """Generate raw empirical truth-error covariance evidence."""
    try:
        artifact = run_empirical_covariance_campaign(
            scenario_path,
            predictor_trajectory_path,
            truth_backend=truth_backend,
            samples=samples,
            seed=seed,
        )
        write_empirical_covariance_artifact(output, artifact)
    except (InvalidScenarioError, UnsupportedBackendError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "samples": len(artifact.samples),
                "truth_backend": artifact.campaign_provenance.truth_backend,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
