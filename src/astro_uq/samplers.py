from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from math import log2
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc  # type: ignore[import-untyped]

from astro_uq.models import (
    ModelVariant,
    ParameterRealization,
    SamplePlan,
    SamplerKind,
)


@dataclass(frozen=True)
class SampleBatch:
    samples: tuple[ParameterRealization, ...]
    metadata: dict[str, Any]
    warnings: tuple[str, ...] = ()


class Sampler(Protocol):
    def generate(
        self,
        plan: SamplePlan,
        parameter_ids: tuple[str, ...],
        model_variants: tuple[ModelVariant, ...] = (),
    ) -> SampleBatch: ...


def _validate_parameter_ids(parameter_ids: tuple[str, ...]) -> None:
    if len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("parameter ids must be unique")


def _sample_id(plan: SamplePlan, sample_index: int) -> str:
    identity = f"{plan.campaign_digest}:{plan.sampler.seed}:{sample_index}"
    return sha256(identity.encode("ascii")).hexdigest()


def _realizations(
    plan: SamplePlan,
    parameter_ids: tuple[str, ...],
    points: NDArray[np.float64],
) -> tuple[ParameterRealization, ...]:
    return tuple(
        ParameterRealization(
            sample_id=_sample_id(plan, index),
            sample_index=index,
            normalized_values={
                parameter_id: float(value)
                for parameter_id, value in zip(parameter_ids, point, strict=True)
            },
            weight=1.0 / len(points),
        )
        for index, point in enumerate(points)
    )


class PseudorandomSampler:
    def generate(
        self,
        plan: SamplePlan,
        parameter_ids: tuple[str, ...],
        model_variants: tuple[ModelVariant, ...] = (),
    ) -> SampleBatch:
        del model_variants
        _validate_parameter_ids(parameter_ids)
        points = np.empty((plan.sampler.samples, len(parameter_ids)))
        digest_words = np.frombuffer(bytes.fromhex(plan.campaign_digest), dtype=">u4")
        for index in range(plan.sampler.samples):
            seed = np.random.SeedSequence([plan.sampler.seed, index, *digest_words.tolist()])
            points[index] = np.random.default_rng(seed).random(len(parameter_ids))
        return SampleBatch(
            samples=_realizations(plan, parameter_ids, points),
            metadata={"engine": "numpy.random.PCG64", "seed": plan.sampler.seed},
        )


class LatinHypercubeSampler:
    def generate(
        self,
        plan: SamplePlan,
        parameter_ids: tuple[str, ...],
        model_variants: tuple[ModelVariant, ...] = (),
    ) -> SampleBatch:
        del model_variants
        _validate_parameter_ids(parameter_ids)
        engine = qmc.LatinHypercube(
            d=len(parameter_ids),
            scramble=plan.sampler.scramble,
            seed=plan.sampler.seed,
        )
        points = engine.random(n=plan.sampler.samples)
        return SampleBatch(
            samples=_realizations(plan, parameter_ids, points),
            metadata={
                "engine": "scipy.stats.qmc.LatinHypercube",
                "seed": plan.sampler.seed,
                "scramble": plan.sampler.scramble,
            },
        )


class SobolSampler:
    def generate(
        self,
        plan: SamplePlan,
        parameter_ids: tuple[str, ...],
        model_variants: tuple[ModelVariant, ...] = (),
    ) -> SampleBatch:
        del model_variants
        _validate_parameter_ids(parameter_ids)
        engine = qmc.Sobol(
            d=len(parameter_ids),
            scramble=plan.sampler.scramble,
            seed=plan.sampler.seed,
        )
        if plan.sampler.skip:
            engine.fast_forward(plan.sampler.skip)
        points = engine.random(n=plan.sampler.samples)
        warnings: tuple[str, ...] = ()
        if log2(plan.sampler.samples) % 1.0:
            warnings = ("Sobol balance properties require a power-of-two sample count.",)
        return SampleBatch(
            samples=_realizations(plan, parameter_ids, points),
            metadata={
                "engine": "scipy.stats.qmc.Sobol",
                "seed": plan.sampler.seed,
                "scramble": plan.sampler.scramble,
                "skip": plan.sampler.skip,
            },
            warnings=warnings,
        )


class SweepSampler:
    def generate(
        self,
        plan: SamplePlan,
        parameter_ids: tuple[str, ...],
        model_variants: tuple[ModelVariant, ...] = (),
    ) -> SampleBatch:
        del model_variants
        _validate_parameter_ids(parameter_ids)
        if set(plan.sampler.sweep_values) != set(parameter_ids):
            raise ValueError("sweep values must exactly match parameter ids")
        values = tuple(plan.sampler.sweep_values[parameter_id] for parameter_id in parameter_ids)
        if any(not dimension for dimension in values):
            raise ValueError("sweep dimensions must not be empty")
        points = np.asarray(tuple(product(*values)), dtype=float)
        if len(points) != plan.sampler.samples:
            raise ValueError("sweep Cartesian product must match requested sample count")
        if np.any((points < 0.0) | (points > 1.0)):
            raise ValueError("sweep normalized values must be in [0, 1]")
        return SampleBatch(
            samples=_realizations(plan, parameter_ids, points),
            metadata={"engine": "cartesian_product", "ordering": list(parameter_ids)},
        )


class EnsembleSampler:
    def generate(
        self,
        plan: SamplePlan,
        parameter_ids: tuple[str, ...],
        model_variants: tuple[ModelVariant, ...] = (),
    ) -> SampleBatch:
        _validate_parameter_ids(parameter_ids)
        if parameter_ids:
            raise ValueError("ensemble sampling does not generate continuous coordinates")
        if len(model_variants) != plan.sampler.samples:
            raise ValueError("model variant count must match requested sample count")
        total_weight = sum(float(variant.weight) for variant in model_variants)
        samples = tuple(
            ParameterRealization(
                sample_id=_sample_id(plan, index),
                sample_index=index,
                model_variants={variant.target: variant.value},
                weight=float(variant.weight) / total_weight,
            )
            for index, variant in enumerate(model_variants)
        )
        return SampleBatch(
            samples=samples,
            metadata={
                "engine": "discrete_model_ensemble",
                "variant_ids": [variant.variant_id for variant in model_variants],
            },
        )


_SAMPLERS: dict[SamplerKind, Sampler] = {
    SamplerKind.PSEUDORANDOM: PseudorandomSampler(),
    SamplerKind.LATIN_HYPERCUBE: LatinHypercubeSampler(),
    SamplerKind.SOBOL: SobolSampler(),
    SamplerKind.SWEEP: SweepSampler(),
    SamplerKind.ENSEMBLE: EnsembleSampler(),
}


def generate_samples(
    plan: SamplePlan,
    parameter_ids: tuple[str, ...] = (),
    model_variants: tuple[ModelVariant, ...] = (),
) -> SampleBatch:
    return _SAMPLERS[plan.sampler.kind].generate(plan, parameter_ids, model_variants)
