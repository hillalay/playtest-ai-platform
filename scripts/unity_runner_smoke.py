"""Manual Unity batch smoke check; policy seeds advance independently of runtime seeds."""

import argparse
import json
from dataclasses import asdict, fields

from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter
from playtest.orchestration.batch import BatchRunner
from playtest.policies.random import RandomPolicy
from playtest.replay.trace import validate_trace


def report(kind, **data):
    print(json.dumps({"kind": kind, **data}, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Batch base policy seed; never a Unity runtime seed")
    parser.add_argument("--max-total-steps", type=int, default=None, help="Optional batch action budget (may be zero)")
    parser.add_argument("--include-traces", action="store_true", help="Include full EpisodeResult traces in JSONL")
    parser.add_argument("--behavior", default=None, help="Exact ML-Agents behavior name")
    parser.add_argument("--file-name", default=None, help="Unity executable; omit for Unity Editor")
    parser.add_argument("--timeout-wait", type=int, default=120)
    parser.add_argument("--max-wait-steps", type=int, default=100)
    args = parser.parse_args(argv)
    for name in ("episodes", "max_steps", "timeout_wait"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_wait_steps", "max_total_steps"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    adapter = UnityMLAgentsAdapter(
        file_name=args.file_name, timeout_wait=args.timeout_wait,
        behavior_name=args.behavior, max_wait_steps=args.max_wait_steps,
    )
    summary = {
        "requested_episodes": args.episodes, "attempted_episodes": 0,
        "completed_episodes": 0, "total_steps": None, "outcomes": {},
        "technical_errors": 0, "average_episode_seconds": None,
        "base_seed": args.seed, "unique_action_sequences": None,
        "diversity_supported": False, "diversity_episodes": 0,
        "diversity_excluded_episodes": 0, "stop_reason": None, "stopped_episode": None,
        "duration_seconds": None,
    }
    batch_error = close_error = None
    interrupted = False
    episode = None
    try:
        report("start", episodes=args.episodes, max_steps=args.max_steps, base_seed=args.seed,
               max_total_steps=args.max_total_steps,
               message="Waiting for Unity; press Play in Editor mode.")
        adapter.load_level("static")
        batch = BatchRunner().run(
            adapter, lambda index, policy_seed: RandomPolicy(seed=policy_seed),
            episodes=args.episodes, max_steps=args.max_steps,
            base_seed=args.seed, max_total_steps=args.max_total_steps,
        )
        summary = {field.name: getattr(batch, field.name) for field in fields(batch) if field.name != "episode_results"}
        for item in batch.episode_results:
            episode = item.episode
            result = item.result
            record = {
                "episode": item.episode, "policy_seed": item.policy_seed, "step_budget": item.step_budget,
                "outcome": result.outcome, "steps": result.steps, "duration_seconds": result.duration_seconds,
                "issues": result.issues, "trace_validated": item.trace_validated,
                "action_sequence_signature": item.action_sequence_signature,
                "diversity_error": item.diversity_error,
            }
            if args.include_traces:
                record.update(asdict(result))
            report("episode", **record)
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

    interrupted = interrupted or summary["stop_reason"] == "INTERRUPTED"
    summary["technical_errors"] += int(
        batch_error is not None and batch_error != "KeyboardInterrupt"
    ) + int(close_error is not None)
    exit_code = 130 if interrupted else int(
        summary["technical_errors"] > 0 or batch_error is not None or close_error is not None
        or summary["stop_reason"] not in {"EPISODES_COMPLETED", "TOTAL_STEP_LIMIT_REACHED"}
        or not summary["diversity_supported"]
    )
    report(
        "summary", **summary, unattempted_episodes=args.episodes - summary["attempted_episodes"],
        action_sequence_basis="unity_discrete engine-index actions; not state coverage",
        batch_error=batch_error, error_episode=episode if batch_error else summary["stopped_episode"],
        close_error=close_error, interrupted=interrupted, exit_code=exit_code,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
