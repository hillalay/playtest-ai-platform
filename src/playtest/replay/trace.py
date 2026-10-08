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


def validate_trace(result, max_steps):
    if result.trace is None or len(result.trace.steps) != result.steps:
        raise ValueError("Trace step count does not match EpisodeResult.steps.")
    if not 0 <= result.steps <= max_steps:
        raise ValueError("Trace exceeds the episode action budget.")
    if result.outcome == "STEP_LIMIT_REACHED" and result.steps != max_steps:
        raise ValueError("Step-limit outcome must exhaust the action budget.")
    expected = None
    for index, step in enumerate(result.trace.steps, 1):
        if step.step_index != index:
            raise ValueError("Trace step indexes must be consecutive from 1.")
        if step.game_terminal is None or step.test_boundary_reached is None:
            raise ValueError("Runner trace is missing terminal/boundary flags.")
        if index < result.steps and (
            step.game_terminal or step.test_boundary_reached or step.invalid_reason is not None
        ):
            raise ValueError("Trace contains an action after an episode stop signal.")
    if result.steps:
        last = result.trace.steps[-1]
        if last.invalid_reason is not None:
            expected = "INVALID_ACTION"
        elif last.game_terminal:
            expected = last.game_outcome or "GAME_TERMINAL"
        elif last.test_boundary_reached:
            expected = "TEST_BOUNDARY_REACHED"
        if expected is not None and result.outcome != expected:
            raise ValueError("Episode outcome does not match the final trace step.")
    if result.outcome in {"SUCCESS", "FAILURE", "GAME_TERMINAL", "TEST_BOUNDARY_REACHED"}:
        if result.outcome != expected:
            raise ValueError("Terminal outcome has no matching trace stop signal.")
