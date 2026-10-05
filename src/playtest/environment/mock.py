from playtest.core.actions import Action
from playtest.core.events import Event
from playtest.core.results import StepResult


class MockEnvironment:

    def __init__(self):
        self.value = 0
        self.target = 5

    def reset(self, seed: int | None = None):
        self.value = 0
        return {"value": self.value}

    def valid_actions(self) -> list[Action]:
        return [
            Action(type="increment"),
            Action(type="decrement"),
        ]

    def step(self, action: Action) -> StepResult:

        if action.type == "increment":
            self.value += 1

        elif action.type == "decrement":
            self.value -= 1

        else:
            return StepResult(
                next_observation={"value": self.value},
                invalid_reason="Unknown action",
            )

        completed = self.value >= self.target

        events = [
            Event(
                type="value_changed",
                payload={"value": self.value},
            )
        ]

        if completed:
            events.append(Event(type="goal_reached"))

        return StepResult(
            next_observation={"value": self.value},
            events=events,
            game_terminal=completed,
            state_signature=str(self.value),
            game_outcome="SUCCESS" if completed else None,
        )

    def close(self) -> None:
        pass
