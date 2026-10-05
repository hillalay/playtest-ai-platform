from typing import Protocol

from playtest.core.actions import Action
from playtest.core.results import StepResult


class Environment(Protocol):

    def reset(self, seed: int | None = None):
        ...

    def valid_actions(self) -> list[Action] | None:
        """None means enumeration is unsupported; an empty list means no legal actions."""
        ...

    def step(self, action: Action) -> StepResult:
        ...

    def close(self) -> None:
        ...
