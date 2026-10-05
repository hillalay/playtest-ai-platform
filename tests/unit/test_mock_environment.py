from playtest.core.actions import Action
from playtest.environment.mock import MockEnvironment


def test_mock_environment_reset():

    env = MockEnvironment()

    observation = env.reset()

    assert observation["value"] == 0


def test_mock_environment_step():

    env = MockEnvironment()
    env.reset()

    result = env.step(Action(type="increment"))

    assert result.next_observation["value"] == 1
    assert result.game_terminal is False


def test_mock_environment_reaches_goal():

    env = MockEnvironment()
    env.reset()

    result = None

    for _ in range(5):
        result = env.step(Action(type="increment"))

    assert result is not None
    assert result.game_terminal is True