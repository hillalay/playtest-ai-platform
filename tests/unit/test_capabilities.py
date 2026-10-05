from playtest.definition.capabilities import (
    ReplayMode,
    SolverGrade,
    analyze_capabilities,
)

from playtest.definition.loader import (
    load_definition,
)


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