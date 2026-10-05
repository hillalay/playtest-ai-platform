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