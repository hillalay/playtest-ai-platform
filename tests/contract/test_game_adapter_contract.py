from playtest.core.actions import Action
from playtest.core.results import StepResult


class FakeAdapter:

    adapter_id = "fake-adapter"
    adapter_version = "1.0.0"

    def __init__(self):
        self.value = 0

    def capabilities(self):
        return {
            "canonical_state": True,
            "clone_restore": True,
            "valid_actions": True,
        }

    def load_level(self, level_ref, seed=None):
        self.value = 0

    def reset(self, seed=None):
        self.value = 0
        return {"value": self.value}

    def observation_spec(self):
        return {
            "value": "int"
        }

    def action_spec(self):
        return {
            "increment": {}
        }

    def valid_actions(self, state=None):
        return [
            Action(type="increment")
        ]

    def step(self, action):
        if action.type == "increment":
            self.value += 1

        return StepResult(
            next_observation={
                "value": self.value
            },
            game_terminal=self.value >= 5,
            state_signature=str(self.value),
        )

    def canonical_state(self):
        return str(self.value).encode()

    def clone_state(self):
        return self.value

    def restore_state(self, handle):
        self.value = handle

    def goal_test(self, state=None):
        return self.value >= 5

    def events(self):
        return []

    def close(self):
        pass


def test_adapter_reset():

    adapter = FakeAdapter()

    adapter.load_level("level_1")

    observation = adapter.reset()

    assert observation["value"] == 0


def test_adapter_step():

    adapter = FakeAdapter()

    adapter.reset()

    result = adapter.step(
        Action(type="increment")
    )

    assert result.next_observation["value"] == 1


def test_adapter_clone_restore():

    adapter = FakeAdapter()

    adapter.reset()

    adapter.step(
        Action(type="increment")
    )

    state = adapter.clone_state()

    adapter.step(
        Action(type="increment")
    )

    assert adapter.canonical_state() == b"2"

    adapter.restore_state(state)

    assert adapter.canonical_state() == b"1"


def test_adapter_goal():

    adapter = FakeAdapter()

    adapter.reset()

    for _ in range(5):
        adapter.step(
            Action(type="increment")
        )

    assert adapter.goal_test() is True