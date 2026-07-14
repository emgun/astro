class MissionAssuranceError(RuntimeError):
    """Raised when a mission-assurance phase cannot produce valid evidence."""

    def __init__(self, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.phase = phase
