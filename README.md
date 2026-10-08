# Playtest AI

Playtest AI is a game-agnostic automated playtesting platform. This repository is being organized around the contracts and milestones in [the technical specification](dosyalar/Playtest_AI_Document.pdf).

## Repository layout

- `src/playtest/`: platform packages. These are package boundaries; implementation will be added as contracts are defined.
- `src/playtest/definition/playtest_definition.schema.json`: packaged game definition schema, shared by the CLI and tests.
- `schemas/`: placeholders for the remaining contracts; these reject all input until implemented.
- `cli/`, `api/`, `sdk/unity/`: planned integration surfaces.
- `examples/arrow_grid/`: planned first game example.
- `tests/`: unit, contract, integration, property, and benchmark test locations.
- `core/`, `agents/`, `solvers/`, `games/`: existing Python prototype, retained in place for now. It has not yet been migrated to `src/playtest/`.

## Setup

```bash
python -m pip install -e .
python test_arrow_adapter.py
python test_arrow_runner.py
```

The prototype runner uses multiprocessing, so the runner example needs an environment that allows child processes.

## Next milestone

Define the Universal Test Model contracts, replace the schema placeholders, and create a fake environment that completes one run end to end before moving the prototype modules.

## Unity ML-Agents adapter

`playtest.adapters.unity_mlagents.UnityMLAgentsAdapter` version **0.2.0** implements
the existing `GameAdapter` protocol without changes to core, Runner or policies.
The backend owns Unity lifecycle, behavior/agent selection, mask validation,
bounded decision waiting, observation caching and `StepResult` construction.
`ObservationCodec` and `ActionMapper` in `playtest.adapters.unity_codecs` own
game semantics. ML-Agents imports are lazy and confined to the backend.

Use Python **3.10** and the validated spike's **mlagents-envs 1.1.0**. The optional
`unity` extra pins that version; it is unnecessary for offline unit tests.

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[dev,unity]'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

For a manual integration check, open the already configured Unity scene. Its
agent must use Behavior Type `Default`. Run the following from the repository
root, then press Play in Unity when the connection waits. Omit `--behavior` when
the environment advertises exactly one behavior; otherwise use its exact name
(including an ML-Agents team suffix, if present).

```powershell
.\.venv\Scripts\python.exe scripts\unity_adapter_smoke.py --steps 3
.\.venv\Scripts\python.exe scripts\unity_adapter_smoke.py --behavior 'YourBehavior?team=0' --steps 3
```

These are alternative invocations. The smoke check connects with `file_name=None`,
resets, enumerates and sends legal actions, stops at termination or interruption,
then explicitly resets again and closes. It reports observations without claiming
reset determinism or game success. No legal actions is reported without inventing
a command. It is never run automatically by pytest. `--file-name` may name an
existing compatible Unity executable; the adapter still cannot inject levels.
`scripts/unity_spike.py` remains unchanged and can still be run separately:

```powershell
.\.venv\Scripts\python.exe scripts\unity_spike.py
```

The raw codec preserves every sensor tensor, in sensor order, as nested Python
lists under `{"observations": [...]}`. It accepts sensor shapes from the actual
behavior spec, including multiple tensors, without an Arrow record format.
It cannot establish complete semantic state or an outcome, so `canonical_state()`,
`state_signature`, `goal_test()` and `game_outcome` are `None` by default.

The default mapper uses `Action("unity_discrete", {"index": N})`. This is explicitly
an engine command for local smoke testing, **not a cross-product semantic action**.
Configure semantic actions through an explicit table that covers every index of
the actual branch with unique actions. For example, for a verified five-action
game integration (the entity IDs below are illustrative):

```python
from playtest.adapters.unity_codecs import DiscreteActionMapper
from playtest.adapters.unity_mlagents import UnityMLAgentsAdapter
from playtest.core.actions import Action

mapper = DiscreteActionMapper({
    index: Action("select_arrow", {"arrow_id": entity_id})
    for index, entity_id in enumerate([90, 12, 721, 48, 333])
})
adapter = UnityMLAgentsAdapter(action_mapper=mapper)
```

The game integration must verify the table against its Unity protocol. The adapter
never derives it from arrow IDs or observation order. For changing mappings,
implement `ActionMapper.validate(action_count)`, `action_spec()`,
`from_index(index, observation)` and `to_index(action, observation)`. `to_index`
must raise `ValueError` for an unknown semantic action. The backend checks its
integer result against bounds and the cached availability mask. ML-Agents mask
`True` disables an action; a missing mask enables the entire branch; an all-True
mask produces `[]`. Rejected actions return `invalid_reason` without sending a
command, advancing Unity, or crediting a reward.

For semantic observations, implement `ObservationCodec` or subclass
`RawObservationCodec`: `decode(observations)` receives copied, unbatched NumPy
sensor arrays; `observation_spec()` describes its output. Set
`canonicalization_version` to a non-empty game/protocol-specific version only
when sufficient state is observable and `canonical_state(observation)` produces
deterministic bytes for semantically equivalent states. The backend caches those
bytes and creates `<canonicalization_version>:sha256:<digest>` signatures; it
never uses Python `hash()` or object identity. A raw observation hash alone is not
proof of complete state. Incompatible canonicalization versions cannot be compared.

Set `goal_semantics=True` only when `game_outcome(observation)` decodes an explicit
game protocol signal into `"SUCCESS"`, `"FAILURE"` or `None` (unknown).
`goal_test()` then returns `True`, `False` or `None` respectively. Rewards,
`EndEpisode()` and the absence of legal actions never establish success or failure.
Callbacks must be deterministic and free of transport/game side effects.

| Capability | Current support |
| --- | --- |
| Behavior selection | Explicit exact name, or automatic when exactly one exists |
| Agent/action space | One agent, one discrete branch, no continuous actions |
| Current level | `load_level("static")` is an explicit no-op; `reset()` resets Unity |
| Sensors/actions | Cached observations, legal action enumeration and mask validation |
| Delayed decisions | Up to `max_wait_steps` additional simulation advances (default 100) |
| Canonical state/outcomes | Only when the configured codec explicitly supports them |
| Clone/restore | `clone_state()` returns `None`; restoration raises `EnvironmentError` |
| Arbitrary levels, runtime seed, historical state queries | Unsupported; raise `EnvironmentError` |
| Deterministic replay, transition model, headless/parallel safety | No capability claim |
| Game event instrumentation | Unsupported; `events()` and step events are empty lists |

A terminal observation is cached and ends action availability until an explicit
reset. A terminal takes precedence if the same agent also appears in decisions.
Different agent IDs in the same batch are conservatively rejected. A changed
agent ID without termination is a technical error, not a fresh episode.
Non-interrupted termination sets `game_terminal`; ML-Agents interruption sets
`test_boundary_reached`. Neither is a confirmed game outcome on its own.
Multiple registered behaviors are allowed when selected explicitly, but other
active agents/behaviors are rejected before stepping: ML-Agents would otherwise
apply implicit zero actions to them. Undetected hidden agents remain a game
integration responsibility; simultaneous multi-agent control is unsupported.

Connection/reset/step failures, malformed protocol data and exhausted waiting
limits raise the existing `playtest.core.errors.EnvironmentError` and close the
acquired environment. Cleanup failures preserve the original failure context.
`close()` is idempotent and final, including after failed initialization.
The ML-Agents constructor owns resources it creates before returning; its own
failure cleanup applies when construction raises before the adapter acquires it.

For Sprint S4, keep environment `seed=None`; `RandomPolicy(seed=...)` can still
seed action selection independently. The Runner returns `TECHNICAL_ERROR` with
issues and a partial trace for technical exceptions, or `INTERRUPTED` for a
keyboard interruption. The caller owns connection cleanup and batch stopping.
It reports unknown termination as `GAME_TERMINAL`, interruption as
`TEST_BOUNDARY_REACHED`, and no actions as `NO_VALID_ACTIONS`. A terminal received
during reset also appears as no actions to the current Runner because reset
returns only an observation. Reconcile PGD claims with runtime capabilities:
`examples/arrow_grid/playtest.yaml` describes a richer abstract example and is
not proof that this Unity integration supports cloning, replay or full state.
Each game's decision hook must guarantee logical gameplay completion. Waiting is
bounded by advances, with `timeout_wait` per network call, not a strict total
wall-time or cancellation budget. Real Editor and repeated-episode reliability
must be checked manually before S4 batch use.

### Manual 100-episode stability check (Sprint S3)

Run this manually in the configured static scene using Python 3.10. Start the
script, then press Play in Unity Editor when it waits for the connection:

```powershell
.\.venv\Scripts\python.exe scripts\unity_adapter_stability.py --episodes 100 --max-steps 50
```

Defaults are `--episodes 100`, `--max-steps 50`, `--timeout-wait 120` and
`--max-wait-steps 100`. Use `--behavior 'YourBehavior?team=0'` for an exact
behavior name, or `--file-name 'C:\path\Game.exe'` to launch a compatible build.
Episodes, action budget and connection timeout must be positive integers;
`--max-wait-steps` may be zero. Network calls use `--timeout-wait`; there is no
separate overall wall-time deadline.

The script keeps one adapter/environment connection for the whole run. Every
episode explicitly resets, then selects the first action in the current valid
action list. It stops that episode on `GAME_TERMINAL`, `TEST_BOUNDARY_REACHED`,
`NO_VALID_ACTIONS` or `STEP_LIMIT_REACHED`. It never sends an action after a
terminal, and never substitutes an action for an empty mask. A terminal received
during reset is reported with `terminal_on_reset=true`, rather than as no actions.
Simultaneous terminal/decision batches use the adapter's existing rules: terminal
takes priority for the same ID; distinct IDs are conservatively rejected.

The first reset's raw observation and valid action list form an exact comparison
baseline. Every subsequent reset must match both. `RESET_MISMATCH` stops the run
before another action. This is a regression check for this static deterministic
test scene, **not canonical state, full-state proof, or a determinism guarantee
for other Unity games**. Sensor order and valid action order participate in the
comparison. The raw codec leaves game success/failure unknown, regardless of
rewards, termination or lack of legal actions.

Output is flushed JSON Lines: `start`, `reset`, `step`, `episode`, then `summary`.
Reset entries include observations, valid actions and baseline comparison; step
entries include the action, returned observation, reward and terminal/boundary
flags. Each episode includes its stop reason, flags, returned step count, last
attempted step, last known observation, error and timings. Episode duration
includes reset and step reporting; reset duration measures `adapter.reset()`
only, including the initial connection. Averages include all attempted episodes,
including failed attempts, and exclude final connection cleanup.

The summary counts requested, attempted, completed and unattempted episodes,
each stop reason, technical errors and reset mismatches. Completed means an
episode reached one of the four normal stop reasons; it does not mean a game
win. Technical failures and mismatches stop the entire run, so the remaining
requested episodes are reported as unattempted. Cleanup errors are also reported
and counted as technical errors. `close()` runs in `finally`, including after a
keyboard interruption. Exit codes: `0` for all requested episodes completed with
no cleanup error, `1` for a technical failure or reset mismatch, `2` for invalid
CLI arguments, and `130` for keyboard interruption.

The offline pytest suite uses fake Unity transport; it does not open Unity.
The existing spike and smoke scripts remain separate manual checks. Unity C#
sources are outside this repository and have not been changed. The project owner
reported the completed real Unity S3 validation: **100/100 episodes on one
connection, 100 GAME_TERMINAL, 0 technical errors and 0 reset mismatches**.
S3 is complete on that reported evidence; this is not a confirmed game-success
count or a claim that the S4 Runner smoke has been run in Unity.

### Manual Runner + RandomPolicy smoke check (Sprint S4.1-S4.3)

Use the same configured scene and Python 3.10 environment. Run this manually,
then press Play in Unity Editor while the connection waits:

```powershell
.\.venv\Scripts\python.exe scripts\unity_runner_smoke.py --episodes 3 --max-steps 50 --seed 42
```

Defaults are three episodes and 50 actions per episode. `--seed` seeds only
`RandomPolicy`; omitting it leaves action selection unseeded. `--behavior`,
`--file-name`, `--timeout-wait` (default 120), and `--max-wait-steps` (default 100)
have the same connection semantics as the S3 scripts. Episodes, action budget
and timeout must be positive integers; waiting advances may be zero.

The script selects `load_level("static")` once and creates one adapter and one
Runner for the batch. Each episode creates a fresh `RandomPolicy` with the same
provided seed and calls `EpisodeRunner.run(adapter, policy, max_steps=...)`.
The Runner explicitly calls `adapter.reset(seed=None)` once per episode, selects
and checks legal actions, advances the adapter and records the transitions.
The script has no action-selection/gameplay loop and never resets a second time
between Runner calls. The next Runner call owns the next explicit reset.

Identical valid-action sequences produce identical choices when using the same
policy seed. This does not seed Unity, guarantee deterministic gameplay, or
guarantee a win. `policy_seed` is reported per episode; `trace.seed` retains its
existing meaning as the environment reset seed and is `None` here. RandomPolicy
does not modify its inputs. The Runner gives policies copied observations and
action lists so a policy cannot mutate the adapter's state or legal-action check.

The existing Runner outcomes remain `GAME_TERMINAL`, `TEST_BOUNDARY_REACHED`,
`NO_VALID_ACTIONS` and `STEP_LIMIT_REACHED` (the **MAX_STEPS** stop condition).
An explicit game-protocol outcome can still yield `SUCCESS` or `FAILURE`; the
raw Unity codec yields neither. Rejected actions are `INVALID_ACTION`, and
unsupported enumeration is `UNSUPPORTED_ACTION_ENUMERATION`. These failures,
technical errors, trace-validation failures and keyboard interruptions stop the
batch before another episode. The adapter closes in `finally` after the batch;
the backend may also close itself immediately on a transport failure.

TraceStep now has additive, defaulted fields for `reward_signals`,
`game_terminal`, `test_boundary_reached`, `invalid_reason` and `game_outcome`.
Each Runner step copies these along with action, observation, next observation,
events and optional state signature. Missing reward signals remain `{}`; unknown
state signatures/outcomes remain `None`. Legacy TraceStep construction still
works; omitted terminal/boundary fields default to `None` (unknown). This adds
recording, not a replay engine or a full-state/canonical-state guarantee.

Runner technical exceptions previously propagated. They now return an
`EpisodeResult(outcome="TECHNICAL_ERROR")` containing issues and all completed
trace entries. Keyboard interruption similarly returns `INTERRUPTED`. Invalid
`max_steps` still raises `ValueError` before reset. A failed `step()` without a
returned StepResult has no fabricated transition: step totals count recorded,
returned transitions, and cannot prove whether Unity applied the last attempted
command before a disconnect. Reset still returns only an observation; a terminal
received during reset is exposed to this generic Runner as `NO_VALID_ACTIONS`.
No Unity-specific state inspection has been added to the Runner.

Flushed JSON Lines contain a `start` record, an `episode` record for every
attempt, and a final `summary`. Each episode includes the complete EpisodeResult
and trace, its policy seed and trace-validation status. Validation checks step
counts, action budget, consecutive indexes, stop flags and final outcome
consistency. It does not prove gameplay correctness. The summary includes
requested/completed/unattempted episode counts, outcome counts, total recorded
steps, technical errors and average episode duration across all recorded
attempts, including failed ones. With no recorded attempts the average is
`None`; if trace validation fails the total step count is `None`. Episode timing
includes reset (and initial connection) but excludes script reporting and final
cleanup. Cleanup failures preserve episode results and fail the batch.

Exit codes are `0` for all requested episodes ending normally with no cleanup
error, `1` for a failed batch, `2` for invalid CLI arguments and `130` for keyboard
interruption. A normal outcome includes a game terminal, boundary, no actions or
step limit, and does not mean a game win. Pytest uses fake Unity transport and
does not start this manual integration check against a real Editor or build.

Later S4 work includes persistent run storage, broader batch orchestration,
RuleBasedPolicy, structured cancellation and overall wall-time budgets. The
current smoke output can be redirected to a file; no replay, solver, coverage
engine, RL, PlayHive exchange or dashboard has been added. The abstract Arrow
PGD's richer capabilities are not runtime guarantees. Ponytail 5.1.0's existing
configuration, including the disabled session-start hook, remains unchanged.
