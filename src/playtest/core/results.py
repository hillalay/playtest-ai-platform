from dataclasses import dataclass, field
from typing import Any, Literal

from playtest.core.events import Event


@dataclass
class StepResult:
    next_observation: Any

    reward_signals: dict[str, float] = field(default_factory=dict)

    events: list[Event] = field(default_factory=list)

    game_terminal: bool = False

    test_boundary_reached: bool = False

    invalid_reason: str | None = None

    state_signature: str | None = None

    # Termination alone does not imply success.
    game_outcome: Literal["SUCCESS", "FAILURE"] | None = None


@dataclass
class EpisodeResult:
    outcome: str

    steps: int

    duration_seconds: float

    issues: list[Any] = field(default_factory=list)

    coverage_summary: dict[str, Any] = field(default_factory=dict)

    trace_ref: str | None = None
    trace: Any | None = None


@dataclass
class BatchEpisodeResult:
    """One attempt, including its policy seed and the original (possibly partial) trace."""

    episode: int
    policy_seed: int
    step_budget: int
    result: EpisodeResult
    trace_validated: bool
    action_sequence_signature: str | None = None
    diversity_error: str | None = None


@dataclass
class BatchResult:
    """In-memory batch metrics; action sequence diversity is not state coverage."""

    requested_episodes: int
    attempted_episodes: int
    completed_episodes: int
    total_steps: int | None
    outcomes: dict[str, int]
    technical_errors: int
    unique_action_sequences: int | None
    episode_results: list[BatchEpisodeResult]
    base_seed: int
    average_episode_seconds: float | None
    duration_seconds: float
    stop_reason: str
    stopped_episode: int | None
    diversity_supported: bool
    diversity_episodes: int
    diversity_excluded_episodes: int
