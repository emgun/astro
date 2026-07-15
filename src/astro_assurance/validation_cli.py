from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from astro_assurance.errors import MissionAssuranceError
from astro_assurance.validation_calibration_io import (
    inspect_assurance_validation_calibration,
    load_assurance_validation_calibration,
)
from astro_assurance.validation_io import (
    format_paired_assurance_validation_summary,
    load_paired_assurance_validation_protocol,
    verify_paired_assurance_validation_result,
    write_paired_assurance_validation_result,
)
from astro_assurance.validation_models import AssuranceValidationStatus
from astro_assurance.validation_runner import (
    run_paired_assurance_validation,
    validate_paired_assurance_protocol,
)
from astro_core.errors import InvalidScenarioError


def validate_assurance_validation_command(
    protocol_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate a paired mission-assurance protocol without executing it."""
    protocol_id: str | None = None
    try:
        protocol = load_paired_assurance_validation_protocol(protocol_path)
        protocol_id = protocol.protocol_id
        validate_paired_assurance_protocol(protocol)
    except (InvalidScenarioError, MissionAssuranceError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc), "protocol_id": protocol_id}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"protocol_id": protocol_id, "valid": True}, sort_keys=True))


def inspect_assurance_calibration_command(
    calibration_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    protocol_path: Annotated[
        Path | None, typer.Option("--protocol", exists=True, readable=True)
    ] = None,
) -> None:
    """Inspect typed evidence, authority coverage, and promotion blockers."""
    try:
        calibration = load_assurance_validation_calibration(calibration_path)
        protocol = (
            None
            if protocol_path is None
            else load_paired_assurance_validation_protocol(protocol_path)
        )
        report = inspect_assurance_validation_calibration(calibration, protocol)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


def run_assurance_validation_command(
    protocol_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    summary_output: Annotated[Path | None, typer.Option("--summary-output")] = None,
) -> None:
    """Run matched and force-model-mismatched assurance profiles in pairs."""
    protocol_id: str | None = None
    try:
        if summary_output is not None and output.resolve() == summary_output.resolve():
            raise InvalidScenarioError(
                "assurance validation output and summary output must be different paths"
            )
        protocol = load_paired_assurance_validation_protocol(protocol_path)
        protocol_id = protocol.protocol_id
        result = run_paired_assurance_validation(protocol)
        write_paired_assurance_validation_result(output, result)
        summary = format_paired_assurance_validation_summary(result)
        if summary_output is not None:
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            summary_output.write_text(summary, encoding="utf-8")
    except (InvalidScenarioError, MissionAssuranceError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc), "protocol_id": protocol_id}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)
    if any(
        profile.status is AssuranceValidationStatus.EXECUTION_FAILURE
        for pair in result.pairs
        for profile in (pair.matched, pair.mismatched)
    ):
        raise typer.Exit(code=1)


def verify_assurance_validation_command(
    result_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Verify paired evidence, embedded cases, manifests, and bound inputs."""
    try:
        result = verify_paired_assurance_validation_result(result_path)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"protocol_id": result.protocol_id, "valid": True}, sort_keys=True))
