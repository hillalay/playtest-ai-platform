from playtest.environment.mock import MockEnvironment
from playtest.orchestration.runner import EpisodeRunner
from playtest.policies.random import RandomPolicy


def test_runner_executes_episode():

    environment = MockEnvironment()
    policy = RandomPolicy(seed=42)
    runner = EpisodeRunner()

    result = runner.run(
        environment=environment,
        policy=policy,
        max_steps=100,
        seed=42,
    )

    assert result.steps <= 100
    assert result.outcome in {
        "SUCCESS",
        "STEP_LIMIT_REACHED",
        "NO_VALID_ACTIONS",
    }