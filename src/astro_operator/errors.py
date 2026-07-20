class OperatorError(RuntimeError):
    """Base error for mission operator failures."""


class OperatorPolicyError(OperatorError):
    """Raised when an action exceeds the active authority or design envelope."""


class OperatorEvaluationError(OperatorError):
    """Raised when a candidate cannot be evaluated safely."""


class ReasonerError(OperatorError):
    """Base error for a reasoner invocation that produced no valid action."""


class ReasonerConfigurationError(ReasonerError):
    """Raised when reasoner configuration or credentials are invalid."""


class ReasonerUnavailableError(ReasonerError):
    """Raised for a transient timeout, rate limit, or provider outage."""


class ReasonerInvalidResponseError(ReasonerError):
    """Raised when a reasoner response cannot be validated as one action."""


class ReasonerCancelledError(ReasonerError):
    """Raised when a reasoner invocation is cancelled before a decision."""
