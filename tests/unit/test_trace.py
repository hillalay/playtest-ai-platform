from playtest.environment.mock import MockEnvironment
from playtest.orchestration.runner import EpisodeRunner
from playtest.policies.random import RandomPolicy


def test_runner_records_trace():

    environment = MockEnvironment()
    policy = RandomPolicy(seed=42)
    runner = EpisodeRunner()

    result = runner.run(
        environment=environment,
        policy=policy,
        max_steps=10,
        seed=42,
    )

    assert result.trace is not None
    assert len(result.trace.steps) == result.steps

    first_step = result.trace.steps[0]

    assert first_step.action is not None
    assert first_step.next_observation is not None


def test_trace_snapshots_mutable_observations_actions_and_events():
    from playtest.core.actions import Action
    from playtest.core.events import Event
    from playtest.core.results import StepResult

    class MutableEnvironment:
        def reset(self, seed=None):
            self.observation = {"nested": {"value": 0}}
            self.action = Action("increment", {"value": 0})
            self.event = Event("changed", {"value": 0})
            self.events = [self.event]
            return self.observation

        def valid_actions(self):
            return [self.action]

        def step(self, action):
            self.observation["nested"]["value"] += 1
            self.action.params["value"] += 1
            self.event.payload["value"] += 1
            return StepResult(self.observation, events=self.events)

    class FirstActionPolicy:
        def select_action(self, observation, valid_actions):
            return valid_actions[0]

    env = MutableEnvironment()
    result = EpisodeRunner().run(env, FirstActionPolicy(), max_steps=2)
    env.observation["nested"]["value"] = 99
    env.action.params["value"] = 99
    env.event.payload["value"] = 99
    env.events.clear()
    assert [s.observation["nested"]["value"] for s in result.trace.steps] == [0, 1]
    assert [s.next_observation["nested"]["value"] for s in result.trace.steps] == [1, 2]
    assert [s.action.params["value"] for s in result.trace.steps] == [0, 1]
    assert [s.events[0].payload["value"] for s in result.trace.steps] == [1, 2]


def test_trace_preserves_reward_flags_invalid_reason_and_unknown_signature():
    from playtest.core.results import StepResult
    from test_runner import IncrementPolicy, ResultEnvironment

    step_result = StepResult(
        {"value": 1}, reward_signals={"progress": 7.5},
        game_terminal=True, test_boundary_reached=True,
        invalid_reason="rejected", game_outcome="FAILURE",
    )
    result = EpisodeRunner().run(ResultEnvironment(step_result), IncrementPolicy())
    step_result.reward_signals["progress"] = -100
    step_result.next_observation["value"] = 999
    step_result.game_terminal = False
    step = result.trace.steps[0]
    assert step.reward_signals == {"progress": 7.5}
    assert step.game_terminal is True and step.test_boundary_reached is True
    assert step.invalid_reason == "rejected" and step.game_outcome == "FAILURE"
    assert step.state_signature is None
    assert step.next_observation == {"value": 1}
    assert result.outcome == "INVALID_ACTION"


def test_trace_leaves_absent_reward_signals_empty():
    from playtest.core.results import StepResult
    from test_runner import IncrementPolicy, ResultEnvironment

    result = EpisodeRunner().run(ResultEnvironment(StepResult({})), IncrementPolicy(), max_steps=1)
    step = result.trace.steps[0]
    assert step.reward_signals == {}
    assert step.game_terminal is False and step.test_boundary_reached is False
    assert step.game_outcome is None and step.state_signature is None
