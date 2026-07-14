from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from astro_assurance.lifecycle_review import (
    review_mission_lifecycle,
    verify_mission_lifecycle_result,
    verify_mission_lifecycle_review,
)
from astro_assurance.lifecycle_review_io import (
    format_mission_lifecycle_review,
    write_mission_lifecycle_review,
    write_mission_lifecycle_review_summary,
)
from astro_core.errors import InvalidScenarioError


def verify_mission_lifecycle_result_command(
    result_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Re-run a local lifecycle scenario and verify exact stored evidence."""
    try:
        result = verify_mission_lifecycle_result(result_path, scenario_path)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        json.dumps(
            {
                "scenario_id": result.scenario_id,
                "workflow": result.workflow,
                "verified": True,
                "claim_boundary": "exact_local_reexecution_not_operational_authority",
            },
            sort_keys=True,
        )
    )


def review_mission_lifecycle_command(
    result_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    summary_output: Annotated[Path | None, typer.Option("--summary-output")] = None,
) -> None:
    """Verify lifecycle evidence and publish deterministic findings and triage."""
    try:
        paths = [result_path, scenario_path, output]
        if summary_output is not None:
            paths.append(summary_output)
        _require_distinct_paths(paths)
        review = review_mission_lifecycle(result_path, scenario_path)
        write_mission_lifecycle_review(output, review)
        summary = format_mission_lifecycle_review(review)
        if summary_output is not None:
            write_mission_lifecycle_review_summary(summary_output, summary)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)


def verify_mission_lifecycle_review_command(
    review_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Re-verify a lifecycle review and its bound scenario and result."""
    try:
        review = verify_mission_lifecycle_review(review_path)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(format_mission_lifecycle_review(review), nl=False)


def _require_distinct_paths(paths: list[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    for index, path in enumerate(resolved):
        for other in resolved[index + 1 :]:
            same_existing_file = (
                path.exists() and other.exists() and os.path.samefile(path, other)
            )
            if path == other or same_existing_file:
                raise InvalidScenarioError("lifecycle review paths must all be different files")
