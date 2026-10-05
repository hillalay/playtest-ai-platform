from dataclasses import dataclass, field
from typing import Any

from playtest.core.events import Event
from playtest.core.actions import Action


@dataclass
class StepResult:
    next_observation: Any

    reward_signals: dict[str, float] = field(default_factory=dict)

    events: list[Event] = field(default_factory=list)

    game_terminal: bool = False

    test_boundary_reached: bool = False

    invalid_reason: str | None = None

    state_signature: str | None = None


@dataclass
class EpisodeResult:
    outcome: str

    steps: int

    duration_seconds: float

    issues: list[Any] = field(default_factory=list)

    coverage_summary: dict[str, Any] = field(default_factory=dict)

    trace_ref: str | None = None