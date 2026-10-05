from dataclasses import dataclass
from enum import Enum
from typing import Any


class SolverGrade(str, Enum):
    FULL_SEARCH = "FULL_SEARCH"
    BOUNDED_SEARCH = "BOUNDED_SEARCH"
    SIMULATION_ORACLE = "SIMULATION_ORACLE"
    NO_SOLVER = "NO_SOLVER"


class ReplayMode(str, Enum):
    EXACT = "EXACT"
    TOLERANT = "TOLERANT"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class CapabilityReport:
    state_observable: bool
    finite_actions: bool
    transition_model: bool
    deterministic: bool
    goal_semantics: bool
    event_instrumentation: bool
    headless: bool
    parallel_safe: bool

    state_coverage: bool
    transition_coverage: bool
    event_coverage: bool
    mechanic_coverage: bool

    solver_grade: SolverGrade
    replay_mode: ReplayMode


def analyze_capabilities(
    definition: dict[str, Any],
) -> CapabilityReport:

    game = definition.get("game", {})
    environment = definition.get("environment", {})
    observation = definition.get("observation", {})
    actions = definition.get("actions", {})
    goals = definition.get("goals", {})
    events = definition.get("events", {})
    coverage = definition.get("coverage", {})
    solver = definition.get("solver", {})

    state_observable = bool(
        observation.get("canonical_state")
        or observation.get("stable_signature")
    )

    finite_actions = bool(
        actions.get("finite", False)
    )

    transition_model = bool(
        solver.get("transition_model", False)
        or solver.get("clone_restore", False)
    )

    deterministic_value = game.get(
        "deterministic",
        False,
    )

    deterministic = deterministic_value is True

    goal_semantics = (
        "success" in goals
    )

    event_instrumentation = bool(
        events.get("definitions")
    )

    headless = bool(
        environment.get(
            "supports_headless",
            False,
        )
    )

    parallel_safe = bool(
        environment.get(
            "parallel_safe",
            False,
        )
    )

    if (
        state_observable
        and finite_actions
        and transition_model
        and goal_semantics
    ):
        solver_grade = SolverGrade.FULL_SEARCH

    elif (
        transition_model
        and goal_semantics
    ):
        solver_grade = SolverGrade.BOUNDED_SEARCH

    elif goal_semantics:
        solver_grade = SolverGrade.SIMULATION_ORACLE

    else:
        solver_grade = SolverGrade.NO_SOLVER

    if deterministic and state_observable:
        replay_mode = ReplayMode.EXACT

    elif state_observable:
        replay_mode = ReplayMode.TOLERANT

    else:
        replay_mode = ReplayMode.UNAVAILABLE

    return CapabilityReport(
        state_observable=state_observable,
        finite_actions=finite_actions,
        transition_model=transition_model,
        deterministic=deterministic,
        goal_semantics=goal_semantics,
        event_instrumentation=event_instrumentation,
        headless=headless,
        parallel_safe=parallel_safe,

        state_coverage=bool(
            coverage.get("state", False)
        ),
        transition_coverage=bool(
            coverage.get("transition", False)
        ),
        event_coverage=bool(
            coverage.get("events", False)
        ),
        mechanic_coverage=bool(
            coverage.get("mechanics", False)
        ),

        solver_grade=solver_grade,
        replay_mode=replay_mode,
    )