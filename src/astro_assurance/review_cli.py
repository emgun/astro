from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from astro_assurance.review import review_assurance_validation
from astro_assurance.review_comparison import (
    compare_assurance_validation_reviews,
    verify_assurance_review_comparison,
)
from astro_assurance.review_io import (
    format_assurance_review_comparison,
    format_assurance_validation_review,
    write_assurance_review_comparison,
    write_assurance_review_summary,
    write_assurance_validation_review,
)
from astro_core.errors import InvalidScenarioError


def review_assurance_validation_command(
    result_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    summary_output: Annotated[Path | None, typer.Option("--summary-output")] = None,
) -> None:
    """Verify paired assurance evidence and write deterministic review findings."""
    try:
        paths = [result_path, output]
        if summary_output is not None:
            paths.append(summary_output)
        _require_distinct_paths(
            paths, "assurance review source and outputs must be different paths or files"
        )
        review = review_assurance_validation(result_path)
        write_assurance_validation_review(output, review)
        summary = format_assurance_validation_review(review)
        if summary_output is not None:
            write_assurance_review_summary(summary_output, summary)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)


def compare_assurance_reviews_command(
    baseline_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    candidate_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    summary_output: Annotated[Path | None, typer.Option("--summary-output")] = None,
) -> None:
    """Re-verify and compare two deterministic assurance reviews."""
    try:
        paths = [baseline_path, candidate_path, output]
        if summary_output is not None:
            paths.append(summary_output)
        _require_distinct_paths(paths, "assurance comparison paths must all be different")
        comparison = compare_assurance_validation_reviews(baseline_path, candidate_path)
        write_assurance_review_comparison(output, comparison)
        summary = format_assurance_review_comparison(comparison)
        if summary_output is not None:
            write_assurance_review_summary(summary_output, summary)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)


def verify_assurance_review_comparison_command(
    comparison_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Re-verify a comparison and all review and paired-evidence inputs."""
    try:
        comparison = verify_assurance_review_comparison(comparison_path)
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(format_assurance_review_comparison(comparison), nl=False)


def _require_distinct_paths(paths: list[Path], message: str) -> None:
    resolved = [path.resolve() for path in paths]
    for index, path in enumerate(resolved):
        for other in resolved[index + 1 :]:
            same_existing_file = (
                path.exists() and other.exists() and os.path.samefile(path, other)
            )
            if path == other or same_existing_file:
                raise InvalidScenarioError(message)
