import pytest

from playtest.core.actions import Action
from playtest.core.results import StepResult
from playtest.environment.mock import MockEnvironment
from playtest.orchestration.runner import EpisodeRunner


class IncrementPolicy:
    def select_action(self, observation, valid_actions):
        return Action(type="increment")


class ResultEnvironment(MockEnvironment):
    def __init__(self, result, actions=None):
        super().__init__()
        self.result = result
        self.actions = actions
        self.step_calls = 0
        self.reset_calls = 0

    def reset(self, seed=None):
        self.reset_calls += 1
        return super().reset(seed)

    def valid_actions(self):
        return super().valid_actions() if self.actions is None else self.actions

    def step(self, action):
        self.step_calls += 1
        return self.result


def test_runner_reaches_goal_in_exactly_five_steps():
    result = EpisodeRunner().run(MockEnvironment(), IncrementPolicy(), max_steps=5)
    assert result.outcome == "SUCCESS"
    assert result.steps == 5
    assert result.issues == []
    assert result.duration_seconds >= 0


def test_runner_stops_at_step_limit():
    result = EpisodeRunner().run(MockEnvironment(), IncrementPolicy(), max_steps=3)
    assert result.outcome == "STEP_LIMIT_REACHED"
    assert result.steps == 3


def test_runner_stops_when_no_actions_are_available():
    env = ResultEnvironment(StepResult({}), actions=[])
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == "NO_VALID_ACTIONS"
    assert result.steps == 0
    assert env.step_calls == 0


def test_runner_reports_unsupported_action_enumeration():
    class NonEnumeratingEnvironment(ResultEnvironment):
        def valid_actions(self, state=None):
            return None

    env = NonEnumeratingEnvironment(StepResult({}))
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == "UNSUPPORTED_ACTION_ENUMERATION"
    assert result.issues
    assert result.steps == env.step_calls == 0
    assert result.trace.steps == []


@pytest.mark.parametrize("game_outcome, expected", [
    ("SUCCESS", "SUCCESS"),
    ("FAILURE", "FAILURE"),
    (None, "GAME_TERMINAL"),
])
def test_runner_preserves_terminal_outcome(game_outcome, expected):
    env = ResultEnvironment(StepResult({}, game_terminal=True, game_outcome=game_outcome))
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == expected
    assert result.steps == env.step_calls == 1


def test_runner_stops_at_test_boundary():
    env = ResultEnvironment(StepResult({}, test_boundary_reached=True))
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == "TEST_BOUNDARY_REACHED"
    assert result.steps == env.step_calls == 1


def test_terminal_outcome_takes_precedence_over_test_boundary():
    env = ResultEnvironment(StepResult(
        {}, game_terminal=True, game_outcome="FAILURE", test_boundary_reached=True,
    ))
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == "FAILURE"
    assert env.step_calls == 1


def test_runner_records_rejected_action_and_stops():
    env = ResultEnvironment(StepResult(
        {}, invalid_reason="Rejected action", game_terminal=True, game_outcome="SUCCESS",
    ))
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == "INVALID_ACTION"
    assert result.issues == ["Rejected action"]
    assert result.steps == env.step_calls == 1


def test_runner_rejects_policy_action_outside_valid_actions():
    env = ResultEnvironment(StepResult({}), actions=[Action(type="decrement")])
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == "INVALID_ACTION"
    assert result.issues
    assert result.steps == env.step_calls == 0


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True])
def test_runner_rejects_invalid_step_limit_before_reset(max_steps):
    env = ResultEnvironment(StepResult({}))
    with pytest.raises(ValueError, match="positive integer"):
        EpisodeRunner().run(env, IncrementPolicy(), max_steps=max_steps)
    assert env.reset_calls == env.step_calls == 0
