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


@pytest.mark.parametrize("operation", ["reset", "valid_actions", "step"])
def test_runner_reports_technical_errors_without_closing_caller_environment(monkeypatch, operation):
    env = MockEnvironment()
    closes = []
    monkeypatch.setattr(env, "close", lambda: closes.append(True))

    def fail(*args, **kwargs):
        raise OSError(f"{operation} disconnected")

    monkeypatch.setattr(env, operation, fail)
    result = EpisodeRunner().run(env, IncrementPolicy())
    assert result.outcome == "TECHNICAL_ERROR"
    assert result.steps == 0 and result.trace.steps == []
    assert result.issues == [f"OSError: {operation} disconnected"]
    assert not closes


@pytest.mark.parametrize("error,outcome", [
    (TimeoutError("timeout"), "TECHNICAL_ERROR"),
    (KeyboardInterrupt(), "INTERRUPTED"),
])
def test_runner_keeps_completed_transitions_when_next_step_fails(monkeypatch, error, outcome):
    env = MockEnvironment()
    step = env.step

    def fail_after_one(action):
        if env.value == 1:
            raise error
        return step(action)

    monkeypatch.setattr(env, "step", fail_after_one)
    result = EpisodeRunner().run(env, IncrementPolicy(), max_steps=3, seed=42)
    assert result.outcome == outcome
    assert result.steps == len(result.trace.steps) == 1
    assert result.trace.seed == 42
    assert result.trace.steps[0].next_observation == {"value": 1}
    assert result.issues


def test_runner_reports_policy_exception_without_sending_an_action():
    class BrokenPolicy:
        def select_action(self, observation, valid_actions):
            raise RuntimeError("policy failed")

    env = ResultEnvironment(StepResult({}))
    result = EpisodeRunner().run(env, BrokenPolicy())
    assert result.outcome == "TECHNICAL_ERROR"
    assert result.steps == env.step_calls == 0
    assert result.trace.steps == []
    assert "policy failed" in result.issues[0]


def test_policy_cannot_mutate_environment_or_legal_action_validation():
    class SharedEnvironment(ResultEnvironment):
        def reset(self, seed=None):
            self.observation = {"nested": {"value": 0}}
            self.actions = [Action("increment", {"value": 0})]
            return self.observation

    class MutatingPolicy:
        def select_action(self, observation, valid_actions):
            observation["nested"]["value"] = 999
            valid_actions[0].params["value"] = 999
            return valid_actions[0]

    env = SharedEnvironment(StepResult({}))
    result = EpisodeRunner().run(env, MutatingPolicy())
    assert result.outcome == "INVALID_ACTION"
    assert result.steps == env.step_calls == 0
    assert env.observation == {"nested": {"value": 0}}
    assert env.actions == [Action("increment", {"value": 0})]
