"""Canonical model dumps that preserve pre-schema-1.3 identities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from astro_operator.models import ConclusionClaim


def model_dump_for_digest(model: BaseModel) -> dict[str, Any]:
    """Omit only the additive empty field on actual conclusion-claim models."""

    data = model.model_dump(mode="json")
    _strip_model_defaults(model, data)
    return data


def _strip_model_defaults(model: object, data: object) -> None:
    if isinstance(model, ConclusionClaim):
        if not model.predicates and isinstance(data, dict):
            data.pop("predicates", None)
        return
    if isinstance(model, BaseModel) and isinstance(data, dict):
        for field_name in type(model).model_fields:
            if field_name in data:
                _strip_model_defaults(getattr(model, field_name), data[field_name])
        return
    if isinstance(model, tuple | list) and isinstance(data, list):
        for model_item, data_item in zip(model, data, strict=True):
            _strip_model_defaults(model_item, data_item)
