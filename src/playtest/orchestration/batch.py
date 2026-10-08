"""Sequential, engine-independent batches; the caller owns the adapter lifecycle."""

import json
import math
import secrets
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import replace
from hashlib import sha256
from statistics import mean
from time import perf_counter

from playtest.adapters.base import GameAdapter
from playtest.core.actions import Action
from playtest.core.results import BatchEpisodeResult, BatchResult, EpisodeResult
from playtest.orchestration.runner import EpisodeRunner
from playtest.policies.base import Policy
from playtest.replay.trace import validate_trace


NORMAL_OUTCOMES = {
    "SUCCESS", "FAILURE", "GAME_TERMINAL", "TEST_BOUNDARY_REACHED",
    "NO_VALID_ACTIONS", "STEP_LIMIT_REACHED",
}
ACTION_SEQUENCE_VERSION = "action-sequence-v1"
PolicyFactory = Callable[[int, int], Policy]


def _check_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_json(item)
        return
    raise ValueError("Action parameters require finite JSON values and string dictionary keys.")


def action_sequence_signature(actions: Iterable[Action]) -> str:
    """Fingerprint action type/params only, not state or gameplay equivalence.

    v1 accepts JSON scalars, lists and string-keyed dictionaries, with finite
    floats. Tuples, custom objects, non-string keys and cycles are unsupported.
    """
    sequence = []
    for action in actions:
        if not isinstance(action, Action) or type(action.type) is not str or type(action.params) is not dict:
            raise ValueError("Action sequence requires Actions with string types and dictionary params.")
        sequence.append({"type": action.type, "params": action.params})
    try:
        _check_json(sequence)
        payload = json.dumps(
            {"version": ACTION_SEQUENCE_VERSION, "actions": sequence},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Unsupported action sequence: use acyclic, finite JSON parameters.") from exc
    return f"{ACTION_SEQUENCE_VERSION}:sha256:{sha256(payload).hexdigest()}"


class BatchRunner:
    """Reuse one caller-owned adapter and EpisodeRunner, with a factory per episode.

    The factory receives a zero-based index and policy seed (base_seed + index).
    None generates a recorded 64-bit base seed. Runtime reset seeds remain None.
    Completed episodes have a validated trace and a NORMAL_OUTCOMES stop reason;
    this includes budget/no-action stops, not just game wins. Diversity excludes
    partial/failed episodes and measures action sequences, never state coverage.
    """

    def run(
        self,
        adapter: GameAdapter,
        policy_factory: PolicyFactory,
        *,
        episodes: int = 100,
        max_steps: int = 100,
        base_seed: int | None = None,
        max_total_steps: int | None = None,
    ) -> BatchResult:
        for name, value in (("episodes", episodes), ("max_steps", max_steps)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if base_seed is not None and type(base_seed) is not int:
            raise ValueError("base_seed must be an integer or None.")
        if max_total_steps is not None and (type(max_total_steps) is not int or max_total_steps < 0):
            raise ValueError("max_total_steps must be a non-negative integer or None.")
        if not callable(policy_factory):
            raise ValueError("policy_factory must be callable.")

        start = perf_counter()
        actual_seed = secrets.randbits(64) if base_seed is None else base_seed
        runner = EpisodeRunner()
        records = []
        signatures = set()
        total_steps = 0
        diversity_supported = True
        stop_reason = "EPISODES_COMPLETED"
        stopped_episode = None
        for index in range(episodes):
            if max_total_steps is not None and total_steps >= max_total_steps:
                stop_reason = "TOTAL_STEP_LIMIT_REACHED"
                stopped_episode = records[-1].episode if records else None
                break
            step_budget = max_steps
            if max_total_steps is not None:
                step_budget = min(max_steps, max_total_steps - total_steps)
            policy_seed = actual_seed + index
            episode_start = perf_counter()
            result = None
            trace_validated = False
            try:
                policy = policy_factory(index, policy_seed)
                result = runner.run(adapter, policy, max_steps=step_budget, seed=None)
                validate_trace(result, step_budget)
                trace_validated = True
            except (Exception, KeyboardInterrupt) as exc:
                outcome = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "TECHNICAL_ERROR"
                issue = f"{type(exc).__name__}: {exc}"
                if result is None:
                    result = EpisodeResult(outcome, 0, perf_counter() - episode_start, issues=[issue])
                else:
                    result = replace(result, outcome=outcome, issues=[*result.issues, issue])
            record = BatchEpisodeResult(index + 1, policy_seed, step_budget, result, trace_validated)
            records.append(record)
            if trace_validated:
                total_steps += result.steps
            if result.outcome not in NORMAL_OUTCOMES:
                record.diversity_error = "Incomplete or failed episode; excluded from action sequence diversity."
                stop_reason, stopped_episode = result.outcome, index + 1
                break
            try:
                record.action_sequence_signature = action_sequence_signature(
                    step.action for step in result.trace.steps
                )
                signatures.add(record.action_sequence_signature)
            except ValueError as exc:
                diversity_supported = False
                record.diversity_error = str(exc)
            except KeyboardInterrupt:
                diversity_supported = False
                record.diversity_error = "KeyboardInterrupt during action sequence fingerprinting."
                stop_reason, stopped_episode = "INTERRUPTED", index + 1
                break

        outcomes = Counter(record.result.outcome for record in records)
        diversity_episodes = sum(record.action_sequence_signature is not None for record in records)
        return BatchResult(
            requested_episodes=episodes, attempted_episodes=len(records),
            completed_episodes=sum(
                record.trace_validated and record.result.outcome in NORMAL_OUTCOMES for record in records
            ),
            total_steps=total_steps if all(record.trace_validated for record in records) else None,
            outcomes=dict(outcomes), technical_errors=outcomes["TECHNICAL_ERROR"],
            unique_action_sequences=len(signatures) if diversity_supported else None,
            episode_results=records, base_seed=actual_seed,
            average_episode_seconds=mean(record.result.duration_seconds for record in records) if records else None,
            duration_seconds=perf_counter() - start, stop_reason=stop_reason,
            stopped_episode=stopped_episode, diversity_supported=diversity_supported,
            diversity_episodes=diversity_episodes, diversity_excluded_episodes=len(records) - diversity_episodes,
        )
