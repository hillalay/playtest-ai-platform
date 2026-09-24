"""
core/runner.py

Game-Agnostic Simulation Runner.

Bir GameAdapter ve Agent alarak bir level'ı
birden fazla kez simüle eder ve temel
playtest metriklerini üretir.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional, Type
import time

import numpy as np

from core.base_adapter import BaseGameAdapter
from core.base_agent import BaseAgent


# =========================================================
# Result constants
# =========================================================

RESULT_WIN = "win"
RESULT_LOSS = "loss"
RESULT_DEADLOCK = "deadlock"
RESULT_TIMEOUT = "timeout"

RESULT_INVALID_ACTION_TYPE = "invalid_action_type"
RESULT_INVALID_ACTION_RANGE = "invalid_action_range"
RESULT_INVALID_ACTION_MASK = "invalid_action_mask"

RESULT_INVALID_MASK_TYPE = "invalid_action_mask_type"
RESULT_INVALID_MASK_DIMENSIONS = "invalid_action_mask_dimensions"
RESULT_INVALID_MASK_LENGTH = "invalid_action_mask_length"
RESULT_INVALID_MASK_VALUES = "invalid_action_mask_values"


# =========================================================
# Action mask validation
# =========================================================

def validate_action_mask(
    action_mask: np.ndarray,
    max_actions: int,
) -> Optional[str]:
    """
    Action mask'in BaseGameAdapter contract'ına
    uygun olup olmadığını kontrol eder.

    Returns:
        None:
            Mask geçerli.

        str:
            Mask geçersizse hata sebebi.
    """

    if not isinstance(action_mask, np.ndarray):
        return RESULT_INVALID_MASK_TYPE

    if action_mask.ndim != 1:
        return RESULT_INVALID_MASK_DIMENSIONS

    if len(action_mask) != max_actions:
        return RESULT_INVALID_MASK_LENGTH

    if action_mask.dtype.kind not in ("b", "i", "u"):
        return RESULT_INVALID_MASK_TYPE

    if not np.all(np.isin(action_mask, [0, 1])):
        return RESULT_INVALID_MASK_VALUES

    return None


def action_mask_is_invalid(
    action_mask: np.ndarray,
    action: int,
) -> bool:
    """
    Verilen action'ın mask tarafından
    geçerli kabul edilip edilmediğini kontrol eder.
    """

    return bool(action_mask[action] != 1)


# =========================================================
# Single episode
# =========================================================

def _run_single_episode(
    adapter_cls: Type[BaseGameAdapter],
    level_data: Dict[str, Any],
    agent: BaseAgent,
    max_steps: int = 150,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tek bir episode çalıştırır.

    Bu fonksiyon multiprocessing tarafından
    ayrı process'lerde çalıştırılabilir.
    """

    # -----------------------------------------------------
    # Seed
    # -----------------------------------------------------

    if seed is not None:
        np.random.seed(seed)

        try:
            agent.set_seed(seed)
        except Exception:
            pass

    # -----------------------------------------------------
    # Game initialization
    # -----------------------------------------------------

    game = adapter_cls()

    game.load_level(level_data)

    observation = game.reset()

    agent.reset()

    steps = 0

    branching_history = []
    action_history = []

    # -----------------------------------------------------
    # Episode loop
    # -----------------------------------------------------

    while steps < max_steps:

        # -------------------------------------------------
        # Get action mask
        # -------------------------------------------------

        action_mask = game.get_action_mask()
        max_actions = game.get_max_actions()

        mask_error = validate_action_mask(
            action_mask,
            max_actions,
        )

        if mask_error is not None:
            return {
                "won": False,
                "reason": mask_error,
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history,
            }

        valid_action_count = int(
            np.sum(action_mask)
        )

        # -------------------------------------------------
        # Deadlock
        # -------------------------------------------------

        if valid_action_count == 0:
            return {
                "won": False,
                "reason": RESULT_DEADLOCK,
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history,
            }

        branching_history.append(
            valid_action_count
        )

        # -------------------------------------------------
        # Agent chooses action
        # -------------------------------------------------

        action = agent.act(
            observation,
            action_mask,
        )

        # -------------------------------------------------
        # Validate action type
        # -------------------------------------------------

        if not isinstance(
            action,
            (int, np.integer),
        ):
            return {
                "won": False,
                "reason": RESULT_INVALID_ACTION_TYPE,
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history,
            }

        action = int(action)

        # -------------------------------------------------
        # Validate action range
        # -------------------------------------------------

        if action < 0 or action >= max_actions:
            return {
                "won": False,
                "reason": RESULT_INVALID_ACTION_RANGE,
                "steps": steps,
                "action": action,
                "branching_history": branching_history,
                "action_history": action_history,
            }

        # -------------------------------------------------
        # Validate action mask
        # -------------------------------------------------

        if action_mask_is_invalid(
            action_mask,
            action,
        ):
            return {
                "won": False,
                "reason": RESULT_INVALID_ACTION_MASK,
                "steps": steps,
                "action": action,
                "branching_history": branching_history,
                "action_history": action_history,
            }

        action_history.append(action)

        # -------------------------------------------------
        # Execute action
        # -------------------------------------------------

        observation, reward, done, info = game.step(
            action
        )

        steps += 1

        # -------------------------------------------------
        # Terminal state
        # -------------------------------------------------

        if done:

            status = info.get("status")
            reason = info.get("reason")

            is_win = (
                status == BaseGameAdapter.STATUS_WIN
                or reason in (
                    "cleared",
                    "figure_rescued",
                )
            )

            return {
                "won": is_win,
                "reason": (
                    reason
                    if reason is not None
                    else (
                        RESULT_WIN
                        if is_win
                        else RESULT_LOSS
                    )
                ),
                "status": status,
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history,
            }

    # -----------------------------------------------------
    # Timeout
    # -----------------------------------------------------

    return {
        "won": False,
        "reason": RESULT_TIMEOUT,
        "steps": steps,
        "branching_history": branching_history,
        "action_history": action_history,
    }


# =========================================================
# Simulation Runner
# =========================================================

class SimulationRunner:
    """
    Merkezi simulation runner.

    Bir level'ı bir Agent ile birden fazla kez
    çalıştırır ve sonuçları toplar.
    """

    def __init__(
        self,
        max_workers: int = 4,
    ):
        if max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than 0"
            )

        self.max_workers = max_workers

    def run_batch(
        self,
        adapter_cls: Type[BaseGameAdapter],
        level_data: Dict[str, Any],
        agent: BaseAgent,
        iterations: int = 500,
        max_steps: int = 150,
        seed: int = 42,
    ) -> Dict[str, Any]:

        if iterations <= 0:
            raise ValueError(
                "iterations must be greater than 0"
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than 0"
            )

        if seed is None:
            seed = 42

        start_time = time.time()

        # -------------------------------------------------
        # Parallel execution
        # -------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            futures = []

            for episode_index in range(iterations):

                future = executor.submit(
                    _run_single_episode,
                    adapter_cls,
                    level_data,
                    agent,
                    max_steps,
                    seed + episode_index,
                )

                futures.append(future)

            results = [
                future.result()
                for future in futures
            ]

        elapsed_time = time.time() - start_time

        # -------------------------------------------------
        # Basic metrics
        # -------------------------------------------------

        total_runs = len(results)

        wins = sum(
            1
            for result in results
            if result["won"]
        )

        deadlocks = sum(
            1
            for result in results
            if result["reason"] == RESULT_DEADLOCK
        )

        timeouts = sum(
            1
            for result in results
            if result["reason"] == RESULT_TIMEOUT
        )

        # -------------------------------------------------
        # Invalid action metrics
        # -------------------------------------------------

        invalid_action_type = sum(
            1
            for result in results
            if result["reason"]
            == RESULT_INVALID_ACTION_TYPE
        )

        invalid_action_range = sum(
            1
            for result in results
            if result["reason"]
            == RESULT_INVALID_ACTION_RANGE
        )

        invalid_action_mask = sum(
            1
            for result in results
            if result["reason"]
            == RESULT_INVALID_ACTION_MASK
        )

        invalid_actions = (
            invalid_action_type
            + invalid_action_range
            + invalid_action_mask
        )

        # -------------------------------------------------
        # Invalid mask metrics
        # -------------------------------------------------

        invalid_mask_type = sum(
            1
            for result in results
            if result["reason"]
            == RESULT_INVALID_MASK_TYPE
        )

        invalid_mask_dimensions = sum(
            1
            for result in results
            if result["reason"]
            == RESULT_INVALID_MASK_DIMENSIONS
        )

        invalid_mask_length = sum(
            1
            for result in results
            if result["reason"]
            == RESULT_INVALID_MASK_LENGTH
        )

        invalid_mask_values = sum(
            1
            for result in results
            if result["reason"]
            == RESULT_INVALID_MASK_VALUES
        )

        invalid_masks = (
            invalid_mask_type
            + invalid_mask_dimensions
            + invalid_mask_length
            + invalid_mask_values
        )

        # -------------------------------------------------
        # Step metrics
        # -------------------------------------------------

        winning_steps = [
            result["steps"]
            for result in results
            if result["won"]
        ]

        # -------------------------------------------------
        # Branching factor
        # -------------------------------------------------

        all_branching = [
            branching
            for result in results
            for branching in result[
                "branching_history"
            ]
        ]

        # -------------------------------------------------
        # Final report
        # -------------------------------------------------

        return {
            "agent_name": agent.name,

            "total_simulations": total_runs,

            "win_rate": round(
                wins / total_runs,
                4,
            ),

            "deadlock_rate": round(
                deadlocks / total_runs,
                4,
            ),

            "timeout_rate": round(
                timeouts / total_runs,
                4,
            ),

            "invalid_action_rate": round(
                invalid_actions / total_runs,
                4,
            ),

            "invalid_action_type_rate": round(
                invalid_action_type / total_runs,
                4,
            ),

            "invalid_action_range_rate": round(
                invalid_action_range / total_runs,
                4,
            ),

            "invalid_action_mask_rate": round(
                invalid_action_mask / total_runs,
                4,
            ),

            "invalid_mask_rate": round(
                invalid_masks / total_runs,
                4,
            ),

            "invalid_mask_type_rate": round(
                invalid_mask_type / total_runs,
                4,
            ),

            "invalid_mask_dimensions_rate": round(
                invalid_mask_dimensions / total_runs,
                4,
            ),

            "invalid_mask_length_rate": round(
                invalid_mask_length / total_runs,
                4,
            ),

            "invalid_mask_values_rate": round(
                invalid_mask_values / total_runs,
                4,
            ),

            "avg_steps_to_win": (
                round(
                    float(
                        np.mean(winning_steps)
                    ),
                    2,
                )
                if winning_steps
                else None
            ),

            "min_steps_to_win": (
                int(np.min(winning_steps))
                if winning_steps
                else None
            ),

            "avg_branching_factor": (
                round(
                    float(
                        np.mean(all_branching)
                    ),
                    2,
                )
                if all_branching
                else 0.0
            ),

            "simulation_time_sec": round(
                elapsed_time,
                3,
            ),
            "results": results,
        }