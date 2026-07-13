"""Typed, integrity-checked surrogate lifecycle artifacts."""

from astro_surrogates.io import SurrogateArtifactError, load_npz_artifact, write_npz_artifact
from astro_surrogates.models import (
    DatasetManifest,
    DatasetSplit,
    SimulationDatasetSpec,
    SimulationEpisodeRef,
    SurrogateDomain,
    SurrogateModelArtifact,
    SurrogatePromotionDecision,
    SurrogateTrainingRun,
    SurrogateValidationReport,
)

__all__ = [
    "DatasetManifest",
    "DatasetSplit",
    "SimulationDatasetSpec",
    "SimulationEpisodeRef",
    "SurrogateArtifactError",
    "SurrogateDomain",
    "SurrogateModelArtifact",
    "SurrogatePromotionDecision",
    "SurrogateTrainingRun",
    "SurrogateValidationReport",
    "load_npz_artifact",
    "write_npz_artifact",
]
