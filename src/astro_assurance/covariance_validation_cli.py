from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from astro_assurance.covariance_validation_io import (
    format_covariance_validation_summary,
    load_covariance_validation_protocol,
    run_covariance_validation,
    verify_covariance_validation_result,
    write_covariance_validation_result,
)
from astro_assurance.covariance_validation_models import CovarianceValidationDisposition
from astro_core.errors import InvalidScenarioError


def validate_covariance_validation_command(
    protocol_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate covariance criteria and bound evidence inputs without assessing them."""
    protocol_id: str | None = None
    try:
        protocol = load_covariance_validation_protocol(protocol_path)
        protocol_id = protocol.protocol_id
        for path in (
            protocol.candidate_trajectory_path,
            protocol.reference_trajectory_path,
            protocol.empirical_evidence_path,
            protocol.independence_review_path,
        ):
            if path is not None and not Path(path).is_file():
                raise InvalidScenarioError(f"covariance evidence source does not exist: {path}")
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc), "protocol_id": protocol_id}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"protocol_id": protocol_id, "valid": True}, sort_keys=True))


def assess_covariance_validation_command(
    protocol_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Assess preregistered covariance comparison and empirical consistency criteria."""
    protocol_id: str | None = None
    try:
        protocol = load_covariance_validation_protocol(protocol_path)
        protocol_id = protocol.protocol_id
        result = run_covariance_validation(protocol)
        write_covariance_validation_result(output, result)
        summary = format_covariance_validation_summary(result)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc), "protocol_id": protocol_id}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)
    if result.disposition is not CovarianceValidationDisposition.CRITERIA_SATISFIED:
        raise typer.Exit(code=1)


def verify_covariance_validation_command(
    result_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Verify covariance evidence digests and exact deterministic reassessment."""
    try:
        result = verify_covariance_validation_result(result_path)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"protocol_id": result.protocol_id, "valid": True}, sort_keys=True))
