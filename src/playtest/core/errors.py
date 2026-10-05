class PlaytestError(Exception):
    """Base exception for Playtest AI."""


class DefinitionError(PlaytestError):
    pass


class EnvironmentError(PlaytestError):
    pass


class PolicyError(PlaytestError):
    pass


class SolverError(PlaytestError):
    pass


class PlaytestTimeoutError(PlaytestError):
    pass