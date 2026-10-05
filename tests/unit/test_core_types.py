from playtest.core.actions import Action
from playtest.core.events import Event
from playtest.core.results import StepResult


def test_action_creation():
    action = Action(
        type="select_arrow",
        params={"arrow_id": 1}
    )

    assert action.type == "select_arrow"
    assert action.params["arrow_id"] == 1


def test_event_creation():
    event = Event(
        type="arrow_removed",
        payload={"arrow_id": 1}
    )

    assert event.type == "arrow_removed"


def test_step_result_creation():
    result = StepResult(
        next_observation={"active_arrows": 5},
        events=[
            Event(
                type="arrow_removed",
                payload={"arrow_id": 1}
            )
        ]
    )

    assert result.game_terminal is False
    assert len(result.events) == 1