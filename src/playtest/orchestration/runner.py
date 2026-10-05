import time

from playtest.core.results import EpisodeResult
from playtest.environment.interface import Environment
from playtest.policies.base import Policy
from playtest.replay.trace import EpisodeTrace, TraceStep


class EpisodeRunner:

    def run(
        self,
        environment: Environment,
        policy: Policy,
        max_steps: int = 100,
        seed: int | None = None,
    ) -> EpisodeResult:

        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        start_time = time.perf_counter()

        observation = environment.reset(seed=seed)
        trace = EpisodeTrace(seed=seed)

        for step_index in range(1, max_steps + 1):

            valid_actions = environment.valid_actions()

            if not valid_actions:
                return EpisodeResult(
                    outcome="NO_VALID_ACTIONS",
                    steps=step_index - 1,
                    duration_seconds=time.perf_counter() - start_time,
                    trace=trace,
                )

            action = policy.select_action(
                observation=observation,
                valid_actions=valid_actions,
            )

            if action not in valid_actions:
                return EpisodeResult(
                    outcome="INVALID_ACTION",
                    steps=step_index - 1,
                    duration_seconds=time.perf_counter() - start_time,
                    issues=["Policy selected an action outside valid_actions"],
                    trace=trace,
                )

            previous_observation = observation
            result = environment.step(action)

            trace.add_step(
                TraceStep(
                    step_index=step_index,
                    observation=previous_observation,
                    action=action,
                    next_observation=result.next_observation,
                    events=result.events,
                    state_signature=result.state_signature,
                )
            )

            observation = result.next_observation

            if result.invalid_reason is not None:
                return EpisodeResult(
                    outcome="INVALID_ACTION",
                    steps=step_index,
                    duration_seconds=time.perf_counter() - start_time,
                    issues=[result.invalid_reason],
                    trace=trace,
                )

            if result.game_terminal:
                return EpisodeResult(
                    outcome=result.game_outcome or "GAME_TERMINAL",
                    steps=step_index,
                    duration_seconds=time.perf_counter() - start_time,
                    trace=trace,
                )

            if result.test_boundary_reached:
                return EpisodeResult(
                    outcome="TEST_BOUNDARY_REACHED",
                    steps=step_index,
                    duration_seconds=time.perf_counter() - start_time,
                    trace=trace,
                )

        return EpisodeResult(
            outcome="STEP_LIMIT_REACHED",
            steps=max_steps,
            duration_seconds=time.perf_counter() - start_time,
            trace=trace,
        )
