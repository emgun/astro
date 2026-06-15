class AstroError(Exception):
    """Base exception for Astro Suite."""


class InvalidScenarioError(AstroError):
    """Raised when a scenario file or object is invalid."""


class InvalidMeasurementFileError(AstroError):
    """Raised when a measurement file is invalid."""


class UnsupportedBackendError(AstroError):
    """Raised when a requested backend is unavailable or unsupported."""


class NumericalConvergenceError(AstroError):
    """Raised when an estimator or propagator fails to converge."""
