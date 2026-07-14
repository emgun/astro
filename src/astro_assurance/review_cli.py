from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from astro_assurance.review import review_assurance_validation
from astro_assurance.review_io import (
    format_assurance_validation_review,
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
        if result_path.resolve() == output.resolve():
            raise InvalidScenarioError("assurance review source and output must be different paths")
        if summary_output is not None and output.resolve() == summary_output.resolve():
            raise InvalidScenarioError(
                "assurance review output and summary must be different paths"
            )
        if summary_output is not None and result_path.resolve() == summary_output.resolve():
            raise InvalidScenarioError(
                "assurance review source and summary must be different paths"
            )
        review = review_assurance_validation(result_path)
        write_assurance_validation_review(output, review)
        summary = format_assurance_validation_review(review)
        if summary_output is not None:
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            summary_output.write_text(summary, encoding="utf-8")
    except (InvalidScenarioError, OSError, ValueError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)
