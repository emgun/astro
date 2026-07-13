from __future__ import annotations

from typing import Protocol

from astro_core.models import AstroModel
from astro_uq.models import CaseObservation, EvaluationOutcome
from astro_uq.parameters import ParameterRegistry


class WorkflowAdapter(Protocol):
    workflow: str

    def parameter_registry(self) -> ParameterRegistry: ...

    def evaluate(self, scenario: AstroModel) -> tuple[EvaluationOutcome, AstroModel | None]: ...

    def observe(self, result: AstroModel, outcome: EvaluationOutcome) -> CaseObservation: ...
