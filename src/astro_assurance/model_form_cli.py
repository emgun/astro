from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from astro_assurance.errors import MissionAssuranceError
from astro_assurance.model_form_io import (
    format_model_form_factorial_summary,
    load_model_form_factorial_protocol,
    verify_model_form_factorial_result,
    write_model_form_factorial_result,
)
from astro_assurance.model_form_runner import (
    run_model_form_factorial,
    validate_model_form_factorial_protocol,
)
from astro_assurance.validation_models import AssuranceValidationStatus
from astro_core.errors import InvalidScenarioError


def validate_model_form_matrix_command(
    protocol_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate a four-cell model-form factorial protocol without executing it."""
    protocol_id: str | None = None
    try:
        protocol = load_model_form_factorial_protocol(protocol_path)
        protocol_id = protocol.protocol_id
        validate_model_form_factorial_protocol(protocol)
    except (InvalidScenarioError, MissionAssuranceError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc), "protocol_id": protocol_id}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"protocol_id": protocol_id, "valid": True}, sort_keys=True))


def run_model_form_matrix_command(
    protocol_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    summary_output: Annotated[Path | None, typer.Option("--summary-output")] = None,
) -> None:
    """Run the four-cell model-form factorial assurance matrix."""
    protocol_id: str | None = None
    try:
        if summary_output is not None and output.resolve() == summary_output.resolve():
            raise InvalidScenarioError(
                "model-form matrix output and summary output must be different paths"
            )
        protocol = load_model_form_factorial_protocol(protocol_path)
        protocol_id = protocol.protocol_id
        result = run_model_form_factorial(protocol)
        write_model_form_factorial_result(output, result)
        summary = format_model_form_factorial_summary(result)
        if summary_output is not None:
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            summary_output.write_text(summary, encoding="utf-8")
    except (InvalidScenarioError, MissionAssuranceError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc), "protocol_id": protocol_id}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)
    if any(
        cell.profile_result.status is AssuranceValidationStatus.EXECUTION_FAILURE
        for realization in result.realizations
        for cell in realization.cells
    ):
        raise typer.Exit(code=1)


def verify_model_form_matrix_command(
    result_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Verify matrix provenance and exact deterministic local reexecution."""
    try:
        result = verify_model_form_factorial_result(result_path)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"protocol_id": result.protocol_id, "valid": True}, sort_keys=True))
