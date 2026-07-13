from __future__ import annotations

import json

import numpy as np
import pytest

from astro_uq.models import ModelVariant, SamplePlan, SamplerKind, SamplerSpec
from astro_uq.samplers import generate_samples

DIGEST = "12" * 32


def _plan(kind: SamplerKind, samples: int, **kwargs: object) -> SamplePlan:
    return SamplePlan(
        sampler=SamplerSpec(kind=kind, samples=samples, **kwargs),
        campaign_digest=DIGEST,
    )


def _points(batch: object) -> np.ndarray:
    return np.asarray([list(sample.normalized_values.values()) for sample in batch.samples])


def test_pseudorandom_golden_values_and_byte_stability() -> None:
    plan = _plan(SamplerKind.PSEUDORANDOM, 2, seed=7)

    first = generate_samples(plan, ("mass", "area"))
    second = generate_samples(plan, ("mass", "area"))

    np.testing.assert_allclose(
        _points(first),
        [[0.4304847553800488, 0.4371257866599605], [0.482488062218532, 0.3444571721603036]],
        rtol=0.0,
        atol=0.0,
    )
    first_json = json.dumps([sample.model_dump() for sample in first.samples], sort_keys=True)
    second_json = json.dumps([sample.model_dump() for sample in second.samples], sort_keys=True)
    assert first_json == second_json
    assert first.samples[0].sample_id == (
        "a137e178aeed7e05e7a3d2d8c7b9037525f13c36b5198c709ebb72cdd47f1249"
    )
    assert [sample.weight for sample in first.samples] == [0.5, 0.5]


def test_latin_hypercube_stratifies_each_marginal() -> None:
    batch = generate_samples(
        _plan(SamplerKind.LATIN_HYPERCUBE, 8, seed=11), ("mass", "area", "drag")
    )

    strata = np.floor(_points(batch) * 8).astype(int)
    for column in strata.T:
        assert sorted(column) == list(range(8))
    assert batch.metadata["engine"] == "scipy.stats.qmc.LatinHypercube"


def test_unscrambled_sobol_matches_reference_points_and_records_warning() -> None:
    batch = generate_samples(
        _plan(SamplerKind.SOBOL, 4, seed=99, scramble=False, skip=0), ("x", "y")
    )

    np.testing.assert_array_equal(
        _points(batch),
        [[0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]],
    )
    assert batch.warnings == ()
    assert batch.metadata["scramble"] is False

    unbalanced = generate_samples(
        _plan(SamplerKind.SOBOL, 3, scramble=True, skip=2), ("x", "y")
    )
    assert "power-of-two" in unbalanced.warnings[0]
    assert unbalanced.metadata["skip"] == 2


def test_sweep_uses_stable_cartesian_order() -> None:
    plan = _plan(
        SamplerKind.SWEEP,
        4,
        sweep_values={"mass": (0.0, 1.0), "area": (0.25, 0.75)},
    )

    batch = generate_samples(plan, ("mass", "area"))

    np.testing.assert_array_equal(
        _points(batch),
        [[0.0, 0.25], [0.0, 0.75], [1.0, 0.25], [1.0, 0.75]],
    )


def test_ensemble_preserves_variant_identity_and_normalizes_weights() -> None:
    variants = (
        ModelVariant(variant_id="msis", target="atmosphere", value="nrlmsise00", weight=1.0),
        ModelVariant(variant_id="jb2008", target="atmosphere", value="jb2008", weight=3.0),
    )

    batch = generate_samples(_plan(SamplerKind.ENSEMBLE, 2), model_variants=variants)

    assert [sample.model_variants for sample in batch.samples] == [
        {"atmosphere": "nrlmsise00"},
        {"atmosphere": "jb2008"},
    ]
    assert [sample.weight for sample in batch.samples] == [0.25, 0.75]
    assert batch.metadata["variant_ids"] == ["msis", "jb2008"]


def test_sample_identity_is_independent_of_generation_order() -> None:
    plan = _plan(SamplerKind.PSEUDORANDOM, 16, seed=41)

    serial = generate_samples(plan, ("x", "y")).samples
    parallel_order = tuple(reversed(generate_samples(plan, ("x", "y")).samples))

    assert {sample.sample_id: sample.model_dump() for sample in serial} == {
        sample.sample_id: sample.model_dump() for sample in parallel_order
    }


def test_sweep_rejects_count_mismatch() -> None:
    plan = _plan(SamplerKind.SWEEP, 3, sweep_values={"x": (0.0, 1.0)})

    with pytest.raises(ValueError, match="must match requested"):
        generate_samples(plan, ("x",))
