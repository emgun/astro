class OperatorError(RuntimeError):
    """Base error for mission operator failures."""


class OperatorPolicyError(OperatorError):
    """Raised when an action exceeds the active authority or design envelope."""


class OperatorEvaluationError(OperatorError):
    """Raised when a candidate cannot be evaluated safely."""
