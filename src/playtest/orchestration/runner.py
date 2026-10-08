import time
from copy import deepcopy

from playtest.core.results import EpisodeResult
from playtest.environment.interface import Environment
from playtest.policies.base import Policy
from playtest.replay.trace import EpisodeTrace, TraceStep


class EpisodeRunner:
    """Run one episode; the caller owns closing the environment.

    seed belongs to environment.reset(), independently of the policy's RNG.
    Technical errors and keyboard interruption retain completed trace entries.
    """

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

        trace = EpisodeTrace(seed=seed)
        try:
            observation = environment.reset(seed=seed)

            for step_index in range(1, max_steps + 1):

                valid_actions = environment.valid_actions()

                if valid_actions is None:
                    return EpisodeResult(
                        outcome="UNSUPPORTED_ACTION_ENUMERATION",
                        steps=step_index - 1,
                        duration_seconds=time.perf_counter() - start_time,
                        issues=["EpisodeRunner requires an enumerated list of valid actions"],
                        trace=trace,
                    )

                if not valid_actions:
                    return EpisodeResult(
                        outcome="NO_VALID_ACTIONS",
                        steps=step_index - 1,
                        duration_seconds=time.perf_counter() - start_time,
                        trace=trace,
                    )

                action = policy.select_action(
                    observation=deepcopy(observation),
                    valid_actions=deepcopy(valid_actions),
                )

                if action not in valid_actions:
                    return EpisodeResult(
                        outcome="INVALID_ACTION",
                        steps=step_index - 1,
                        duration_seconds=time.perf_counter() - start_time,
                        issues=["Policy selected an action outside valid_actions"],
                        trace=trace,
                    )

                previous_observation = deepcopy(observation)
                recorded_action = deepcopy(action)
                result = environment.step(action)

                trace.add_step(
                    TraceStep(
                        step_index=step_index,
                        observation=previous_observation,
                        action=recorded_action,
                        next_observation=deepcopy(result.next_observation),
                        events=deepcopy(result.events),
                        state_signature=result.state_signature,
                        reward_signals=deepcopy(result.reward_signals),
                        game_terminal=result.game_terminal,
                        test_boundary_reached=result.test_boundary_reached,
                        invalid_reason=result.invalid_reason,
                        game_outcome=result.game_outcome,
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
        except (Exception, KeyboardInterrupt) as exc:
            return EpisodeResult(
                outcome="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "TECHNICAL_ERROR",
                steps=len(trace.steps),
                duration_seconds=time.perf_counter() - start_time,
                issues=[f"{type(exc).__name__}: {exc}"],
                trace=trace,
            )
