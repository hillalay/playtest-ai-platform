from dataclasses import dataclass, field
from typing import Any, Literal

from playtest.core.actions import Action
from playtest.core.events import Event


@dataclass
class TraceStep:
    step_index: int
    observation: Any
    action: Action
    next_observation: Any
    events: list[Event] = field(default_factory=list)
    state_signature: str | None = None
    reward_signals: dict[str, float] = field(default_factory=dict)
    game_terminal: bool | None = None
    test_boundary_reached: bool | None = None
    invalid_reason: str | None = None
    game_outcome: Literal["SUCCESS", "FAILURE"] | None = None


@dataclass
class EpisodeTrace:
    seed: int | None
    steps: list[TraceStep] = field(default_factory=list)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)
