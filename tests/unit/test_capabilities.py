from playtest.definition.capabilities import (
    ReplayMode,
    SolverGrade,
    analyze_capabilities,
)

from playtest.definition.loader import (
    load_definition,
)
import pytest


def test_arrow_capabilities():

    definition = load_definition(
        "examples/arrow_grid/playtest.yaml"
    )

    report = analyze_capabilities(
        definition
    )

    assert report.state_observable is True
    assert report.finite_actions is True
    assert report.transition_model is True
    assert report.deterministic is True
    assert report.goal_semantics is True

    assert (
        report.solver_grade
        == SolverGrade.FULL_SEARCH
    )

    assert (
        report.replay_mode
        == ReplayMode.EXACT
    )


@pytest.mark.parametrize("deterministic, grade, replay", [
    (True, SolverGrade.FULL_SEARCH, ReplayMode.EXACT),
    ("seeded", SolverGrade.FULL_SEARCH, ReplayMode.EXACT),
    (False, SolverGrade.SIMULATION_ORACLE, ReplayMode.TOLERANT),
])
def test_capabilities_respect_determinism(deterministic, grade, replay):
    definition = load_definition("examples/arrow_grid/playtest.yaml")
    definition["game"]["deterministic"] = deterministic
    report = analyze_capabilities(definition)
    assert report.solver_grade == grade
    assert report.replay_mode == replay
    assert report.deterministic == (deterministic is not False)


@pytest.mark.parametrize("success", [None, "", "  ", {}, True])
def test_invalid_goal_has_no_solver(success):
    definition = load_definition("examples/arrow_grid/playtest.yaml")
    definition["goals"]["success"] = success
    report = analyze_capabilities(definition)
    assert report.goal_semantics is False
    assert report.solver_grade == SolverGrade.NO_SOLVER
