from astro_core.errors import AstroError


class MissionLifecycleError(AstroError):
    """Raised when a mission lifecycle phase or continuity gate fails."""
