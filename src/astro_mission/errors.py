from astro_core.errors import AstroError


class MissionLifecycleError(AstroError):
    """Raised when a mission lifecycle phase or continuity gate fails."""

    def __init__(self, message: str, *, lifecycle_phase: str | None = None) -> None:
        super().__init__(message)
        self.lifecycle_phase = lifecycle_phase
