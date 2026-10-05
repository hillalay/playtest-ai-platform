import time

from playtest.core.results import EpisodeResult
from playtest.environment.interface import Environment
from playtest.policies.base import Policy


class EpisodeRunner:

    def run(
        self,
        environment: Environment,
        policy: Policy,
        max_steps: int = 100,
        seed: int | None = None,
    ) -> EpisodeResult:

        start_time = time.perf_counter()

        observation = environment.reset(seed=seed)

        for step_index in range(1, max_steps + 1):

            valid_actions = environment.valid_actions()

            if not valid_actions:
                return EpisodeResult(
                    outcome="NO_VALID_ACTIONS",
                    steps=step_index - 1,
                    duration_seconds=time.perf_counter() - start_time,
                )

            action = policy.select_action(
                observation=observation,
                valid_actions=valid_actions,
            )

            result = environment.step(action)

            observation = result.next_observation

            if result.game_terminal:
                return EpisodeResult(
                    outcome="SUCCESS",
                    steps=step_index,
                    duration_seconds=time.perf_counter() - start_time,
                )

        return EpisodeResult(
            outcome="STEP_LIMIT_REACHED",
            steps=max_steps,
            duration_seconds=time.perf_counter() - start_time,
        )