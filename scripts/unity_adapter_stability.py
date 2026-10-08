"""Manual repeated-episode check for a static Unity scene; no game outcome oracle."""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from statistics import mean
from time import perf_counter

from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter


def report(kind, **data):
    print(json.dumps({"kind": kind, **data}, allow_nan=False), flush=True)


def run_episodes(adapter, episodes, max_steps):
    records = []
    baseline = None
    for episode in range(1, episodes + 1):
        start = perf_counter()
        record = {
            "episode": episode,
            "steps": 0,
            "attempted_step": None,
            "outcome": None,
            "game_outcome": None,
            "reset_seconds": 0.0,
            "last_observation": None,
            "error": None,
        }
        try:
            reset_start = perf_counter()
            try:
                record["last_observation"] = adapter.reset()
            finally:
                record["reset_seconds"] = perf_counter() - reset_start
            actions = adapter.valid_actions()
            if actions is None:
                raise RuntimeError("Stability check requires enumerated valid actions.")
            # reset() returns only an observation; inspect the backend's cached
            # terminal batch so a terminal on reset is not called NO_VALID_ACTIONS.
            terminal = bool(len(adapter.terminal_steps))
            interrupted = terminal and bool(adapter.terminal_steps.interrupted[0])
            record["terminal_on_reset"] = terminal
            snapshot = deepcopy((record["last_observation"], actions))
            if baseline is None:
                baseline = snapshot
            matches = snapshot == baseline
            report(
                "reset", episode=episode, observation=record["last_observation"],
                valid_actions=[asdict(action) for action in actions],
                baseline_match=matches, terminal=terminal, interrupted=interrupted,
            )
            if not matches:
                record["outcome"] = "RESET_MISMATCH"
            elif terminal:
                record["outcome"] = "TEST_BOUNDARY_REACHED" if interrupted else "GAME_TERMINAL"
            else:
                for step in range(1, max_steps + 1):
                    if not actions:
                        record["outcome"] = "NO_VALID_ACTIONS"
                        break
                    action = actions[0]
                    record["attempted_step"] = step
                    result = adapter.step(action)
                    record["steps"] = step
                    record["last_observation"] = deepcopy(result.next_observation)
                    record["game_outcome"] = result.game_outcome
                    report(
                        "step", episode=episode, step=step, action=asdict(action),
                        observation=result.next_observation,
                        game_terminal=result.game_terminal,
                        test_boundary_reached=result.test_boundary_reached,
                        game_outcome=result.game_outcome, invalid_reason=result.invalid_reason,
                        reward_signals=result.reward_signals,
                    )
                    if result.invalid_reason is not None:
                        raise RuntimeError(f"Enumerated action was rejected: {result.invalid_reason}")
                    if result.game_terminal:
                        record["outcome"] = "GAME_TERMINAL"
                        break
                    if result.test_boundary_reached:
                        record["outcome"] = "TEST_BOUNDARY_REACHED"
                        break
                    # Do not request another action after the final budgeted step.
                    if step < max_steps:
                        actions = adapter.valid_actions()
                        if actions is None:
                            raise RuntimeError("Stability check requires enumerated valid actions.")
                else:
                    record["outcome"] = "STEP_LIMIT_REACHED"
        except KeyboardInterrupt:
            record["outcome"] = "INTERRUPTED"
            record["error"] = "KeyboardInterrupt"
        except Exception as exc:
            record["outcome"] = "TECHNICAL_ERROR"
            record["error"] = f"{type(exc).__name__}: {exc}"
        record.update(
            game_terminal=record["outcome"] == "GAME_TERMINAL",
            test_boundary_reached=record["outcome"] == "TEST_BOUNDARY_REACHED",
            no_valid_actions=record["outcome"] == "NO_VALID_ACTIONS",
            max_steps_reached=record["outcome"] == "STEP_LIMIT_REACHED",
            technical_error=record["outcome"] == "TECHNICAL_ERROR",
            reset_mismatch=record["outcome"] == "RESET_MISMATCH",
            episode_seconds=perf_counter() - start,
        )
        records.append(record)
        report("episode", **record)
        if record["outcome"] in ("TECHNICAL_ERROR", "RESET_MISMATCH", "INTERRUPTED"):
            break
    return records


def summarize(records, requested, close_error=None):
    counts = {
        key: sum(record[key] for record in records)
        for key in ("game_terminal", "test_boundary_reached", "no_valid_actions",
                    "max_steps_reached", "technical_error", "reset_mismatch")
    }
    completed = sum(record["outcome"] in (
        "GAME_TERMINAL", "TEST_BOUNDARY_REACHED", "NO_VALID_ACTIONS", "STEP_LIMIT_REACHED"
    ) for record in records)
    interrupted = any(record["outcome"] == "INTERRUPTED" for record in records)
    exit_code = 130 if interrupted else int(completed != requested or close_error is not None)
    return {
        "requested_episodes": requested,
        "attempted_episodes": len(records),
        "completed_episodes": completed,
        "unattempted_episodes": requested - len(records),
        **counts,
        "technical_error": counts["technical_error"] + int(close_error is not None),
        "interrupted": interrupted,
        "close_error": close_error,
        "average_episode_seconds": mean(r["episode_seconds"] for r in records) if records else None,
        "average_reset_seconds": mean(r["reset_seconds"] for r in records) if records else None,
        "exit_code": exit_code,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
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
        file_name=args.file_name, behavior_name=args.behavior,
        timeout_wait=args.timeout_wait, max_wait_steps=args.max_wait_steps,
    )
    records = []
    close_error = None
    try:
        report("start", **vars(args), message="Waiting for Unity; press Play in Editor mode.")
        records = run_episodes(adapter, args.episodes, args.max_steps)
    finally:
        try:
            adapter.close()
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
    summary = summarize(records, args.episodes, close_error)
    report("summary", **summary)
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
