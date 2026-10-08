"""Manual Unity smoke check using EpisodeRunner and a fresh seeded policy per episode."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from statistics import mean

from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter
from playtest.orchestration.runner import EpisodeRunner
from playtest.policies.random import RandomPolicy


NORMAL_OUTCOMES = {
    "SUCCESS", "FAILURE", "GAME_TERMINAL", "TEST_BOUNDARY_REACHED",
    "NO_VALID_ACTIONS", "STEP_LIMIT_REACHED",
}


def report(kind, **data):
    print(json.dumps({"kind": kind, **data}, allow_nan=False), flush=True)


def validate_trace(result, max_steps):
    if result.trace is None or len(result.trace.steps) != result.steps:
        raise ValueError("Trace step count does not match EpisodeResult.steps.")
    if not 0 <= result.steps <= max_steps:
        raise ValueError("Trace exceeds the episode action budget.")
    if result.outcome == "STEP_LIMIT_REACHED" and result.steps != max_steps:
        raise ValueError("Step-limit outcome must exhaust the action budget.")
    expected = None
    for index, step in enumerate(result.trace.steps, 1):
        if step.step_index != index:
            raise ValueError("Trace step indexes must be consecutive from 1.")
        if step.game_terminal is None or step.test_boundary_reached is None:
            raise ValueError("Runner trace is missing terminal/boundary flags.")
        if index < result.steps and (
            step.game_terminal or step.test_boundary_reached or step.invalid_reason is not None
        ):
            raise ValueError("Trace contains an action after an episode stop signal.")
    if result.steps:
        last = result.trace.steps[-1]
        if last.invalid_reason is not None:
            expected = "INVALID_ACTION"
        elif last.game_terminal:
            expected = last.game_outcome or "GAME_TERMINAL"
        elif last.test_boundary_reached:
            expected = "TEST_BOUNDARY_REACHED"
        if expected is not None and result.outcome != expected:
            raise ValueError("Episode outcome does not match the final trace step.")
    if result.outcome in {"SUCCESS", "FAILURE", "GAME_TERMINAL", "TEST_BOUNDARY_REACHED"}:
        if result.outcome != expected:
            raise ValueError("Terminal outcome has no matching trace stop signal.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="RandomPolicy seed only; never a Unity runtime seed")
    parser.add_argument("--behavior", default=None, help="Exact ML-Agents behavior name")
    parser.add_argument("--file-name", default=None, help="Unity executable; omit for Unity Editor")
    parser.add_argument("--timeout-wait", type=int, default=120)
    parser.add_argument("--max-wait-steps", type=int, default=100)
    args = parser.parse_args(argv)
    for name in ("episodes", "max_steps", "timeout_wait"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_wait_steps < 0:
        parser.error("--max-wait-steps must be non-negative")

    adapter = UnityMLAgentsAdapter(
        file_name=args.file_name, timeout_wait=args.timeout_wait,
        behavior_name=args.behavior, max_wait_steps=args.max_wait_steps,
    )
    runner = EpisodeRunner()
    records = []
    batch_error = close_error = None
    interrupted = False
    episode = None
    try:
        report("start", episodes=args.episodes, max_steps=args.max_steps, policy_seed=args.seed,
               message="Waiting for Unity; press Play in Editor mode.")
        adapter.load_level("static")
        for episode in range(1, args.episodes + 1):
            # Each run owns its explicit reset; seed=None leaves Unity's runtime unseeded.
            result = runner.run(adapter, RandomPolicy(seed=args.seed), max_steps=args.max_steps)
            trace_validated = True
            try:
                validate_trace(result, args.max_steps)
            except ValueError as exc:
                trace_validated = False
                result = replace(result, outcome="TECHNICAL_ERROR", issues=[*result.issues, str(exc)])
            record = {
                "episode": episode, "policy_seed": args.seed,
                "trace_validated": trace_validated, **asdict(result),
            }
            records.append(record)
            report("episode", **record)
            if result.outcome not in NORMAL_OUTCOMES:
                break
    except KeyboardInterrupt:
        interrupted = True
        batch_error = "KeyboardInterrupt"
    except Exception as exc:
        batch_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            adapter.close()
        except KeyboardInterrupt:
            interrupted = True
            close_error = "KeyboardInterrupt during close"
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"

    outcomes = Counter(r["outcome"] for r in records)
    interrupted = interrupted or bool(outcomes["INTERRUPTED"])
    completed = sum(r["outcome"] in NORMAL_OUTCOMES for r in records)
    technical_errors = outcomes["TECHNICAL_ERROR"] + int(
        batch_error is not None and batch_error != "KeyboardInterrupt"
    ) + int(close_error is not None)
    exit_code = 130 if interrupted else int(
        completed != args.episodes or batch_error is not None or close_error is not None
    )
    report(
        "summary", requested_episodes=args.episodes, attempted_episodes=len(records),
        completed_episodes=completed, unattempted_episodes=args.episodes - len(records),
        outcomes=dict(outcomes),
        total_steps=sum(r["steps"] for r in records) if all(r["trace_validated"] for r in records) else None,
        technical_errors=technical_errors,
        average_episode_seconds=mean(r["duration_seconds"] for r in records) if records else None,
        batch_error=batch_error, error_episode=episode if batch_error else None,
        close_error=close_error, interrupted=interrupted, exit_code=exit_code,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
