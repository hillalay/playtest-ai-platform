"""
core/runner.py

Çok Çekirdekli Paralel Simülasyon Motoru
(Game-Agnostic Simulation Runner).

Herhangi bir Game Adapter ve Agent alarak bir seviyeyi
yüzlerce/binlerce kez paralel olarak simüle eder ve
ham test metriklerini toplar.
"""

from concurrent.futures import ProcessPoolExecutor
import time
from typing import Any, Dict, Type

import numpy as np

from core.base_adapter import BaseGameAdapter
from core.base_agent import BaseAgent


def _run_single_episode(
    adapter_cls: Type[BaseGameAdapter],
    level_data: Dict[str, Any],
    agent: BaseAgent,
    max_steps: int = 150
) -> Dict[str, Any]:
    """
    Tek bir simulation episode'unu baştan sona çalıştırır.

    Multiprocessing uyumluluğu için modül seviyesinde
    tanımlanmıştır.
    """

    # ---------------------------------------------------------
    # 1. Oyunu başlat
    # ---------------------------------------------------------

    game = adapter_cls()

    game.load_level(level_data)

    obs = game.reset()

    # Her episode başında agent state'ini sıfırla.
    agent.reset()

    steps = 0

    branching_history = []
    action_history = []

    # ---------------------------------------------------------
    # 2. Oyun döngüsü
    # ---------------------------------------------------------

    while steps < max_steps:

        # Mevcut durumdaki geçerli action'ları al.
        mask = game.get_action_mask()

        valid_action_count = int(np.sum(mask))
        
        mask_error = validate_action_mask(
            mask,
            valid_action_count
        )
        
        if mask_error is not None:
            return {
                "won": False,
                "reason": mask_error,
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history
            }
        
        valid_action_count = int(np.sum(mask))

        # -----------------------------------------------------
        # Deadlock kontrolü
        # -----------------------------------------------------

        if valid_action_count == 0:
            return {
                "won": False,
                "reason": "deadlock",
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history
            }

        branching_history.append(valid_action_count)

        # -----------------------------------------------------
        # 3. Agent action seçer
        # -----------------------------------------------------

        action = agent.act(obs, mask)

        # -----------------------------------------------------
        # 4. Action validation
        # -----------------------------------------------------

        # Action gerçekten integer mı?
        if not isinstance(action, (int, np.integer)):
            return {
                "won": False,
                "reason": "invalid_action_type",
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history
            }

        action = int(action)

        # Action ID action space sınırları içinde mi?
        max_actions = game.get_max_actions()

        if action < 0 or action >= max_actions:
            return {
                "won": False,
                "reason": "invalid_action_range",
                "steps": steps,
                "action": action,
                "branching_history": branching_history,
                "action_history": action_history
            }

        # Action mask'e göre bu action gerçekten geçerli mi?
        if action_mask_is_invalid(mask, action):
            return {
                "won": False,
                "reason": "invalid_action_mask",
                "steps": steps,
                "action": action,
                "branching_history": branching_history,
                "action_history": action_history
            }

        action_history.append(action)

        # -----------------------------------------------------
        # 5. Geçerli action'ı oyuna uygula
        # -----------------------------------------------------

        obs, reward, done, info = game.step(action)

        steps += 1

        # -----------------------------------------------------
        # 6. Oyun bitti mi?
        # -----------------------------------------------------

        if done:

            is_win = (
                info.get("status") == "win"
                or info.get("reason") in [
                    "cleared",
                    "figure_rescued"
                ]
            )

            return {
                "won": is_win,
                "reason": info.get(
                    "reason",
                    "cleared" if is_win else "loss"
                ),
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history
            }

    # ---------------------------------------------------------
    # 7. Maximum step sınırına ulaşıldı
    # ---------------------------------------------------------

    return {
        "won": False,
        "reason": "timeout",
        "steps": steps,
        "branching_history": branching_history,
        "action_history": action_history
    }
def validate_action_mask(
    action_mask: np.ndarray,
    action: int
) -> str | None:
    """
    Action mask'in BaseGameAdapter contract'ına uygun olup olmadığını kontrol eder. 
    Geçerliyse: None 
    Geçersizse: Hata sebebini string olarak döndürür.
    """
    # Mask gerçekten NumPy array mi?
    if not isinstance(action_mask, np.ndarray):
        return "invalid_action_mask_type"
    
    # Mask 1 boyutlu olmalı.
    if action_mask.ndim != 1:
        return "invalid_action_mask_dimension"
    
    # Mask uzunluğu action space ile aynı olmalı.
    if len(action_mask) != max_actions:
        return "invalid_action_mask_length"
    
    # Mask yalnızca 0 ve 1 içermeli.
    if not np.all(np.isin(action_mask, [0, 1])):
        return "invalid_action_mask_values"
    
    
    return None


def action_mask_is_invalid(
    action_mask: np.ndarray,
    action: int
) -> bool:
    """
    Action'ın action mask içerisinde geçerli olup olmadığını
    kontrol eder.

    True  -> action geçersiz
    False -> action geçerli
    """

    return action_mask[action] != 1


class SimulationRunner:
    """
    Seviyeleri çoklu çekirdekte paralel simüle eden
    merkezi simulation motoru.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def run_batch(
        self,
        adapter_cls: Type[BaseGameAdapter],
        level_data: Dict[str, Any],
        agent: BaseAgent,
        iterations: int = 500,
        max_steps: int = 150
    ) -> Dict[str, Any]:
        """
        Verilen seviyeyi belirtilen agent ile
        'iterations' kez simüle eder.

        Örneğin:

            iterations = 500

        ise level 500 kez oynatılır.

        Sonuç olarak:

            win_rate
            deadlock_rate
            timeout_rate
            invalid_action_rate
            avg_steps_to_win
            min_steps_to_win
            avg_branching_factor

        gibi metrikler döndürülür.
        """

        start_time = time.time()

        # -----------------------------------------------------
        # Paralel simulation
        # -----------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            futures = [
                executor.submit(
                    _run_single_episode,
                    adapter_cls,
                    level_data,
                    agent,
                    max_steps
                )
                for _ in range(iterations)
            ]

            results = [
                future.result()
                for future in futures
            ]

        elapsed_time = time.time() - start_time

        # -----------------------------------------------------
        # Temel metrikler
        # -----------------------------------------------------

        total_runs = len(results)

        wins = sum(
            1
            for result in results
            if result["won"]
        )

        deadlocks = sum(
            1
            for result in results
            if result["reason"] == "deadlock"
        )

        timeouts = sum(
            1
            for result in results
            if result["reason"] == "timeout"
        )

        # -----------------------------------------------------
        # Invalid action metrikleri
        # -----------------------------------------------------

        invalid_action_type = sum(
            1
            for result in results
            if result["reason"] == "invalid_action_type"
        )

        invalid_action_range = sum(
            1
            for result in results
            if result["reason"] == "invalid_action_range"
        )

        invalid_action_mask = sum(
            1
            for result in results
            if result["reason"] == "invalid_action_mask"
        )

        invalid_actions = (
            invalid_action_type
            + invalid_action_range
            + invalid_action_mask
        )

        # -----------------------------------------------------
        # Step metrikleri
        # -----------------------------------------------------

        winning_steps = [
            result["steps"]
            for result in results
            if result["won"]
        ]

        # -----------------------------------------------------
        # Branching metrikleri
        # -----------------------------------------------------

        all_branching = [
            branching
            for result in results
            for branching in result["branching_history"]
        ]

        # -----------------------------------------------------
        # Final report
        # -----------------------------------------------------

        return {
            "agent_name": agent.name,

            "total_simulations": total_runs,

            "win_rate": round(
                wins / total_runs,
                4
            ),

            "deadlock_rate": round(
                deadlocks / total_runs,
                4
            ),

            "timeout_rate": round(
                timeouts / total_runs,
                4
            ),

            "invalid_action_rate": round(
                invalid_actions / total_runs,
                4
            ),

            "invalid_action_type_rate": round(
                invalid_action_type / total_runs,
                4
            ),

            "invalid_action_range_rate": round(
                invalid_action_range / total_runs,
                4
            ),

            "invalid_action_mask_rate": round(
                invalid_action_mask / total_runs,
                4
            ),

            "avg_steps_to_win": (
                round(
                    float(np.mean(winning_steps)),
                    2
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
                    float(np.mean(all_branching)),
                    2
                )
                if all_branching
                else 0.0
            ),

            "simulation_time_sec": round(
                elapsed_time,
                3
            )
        }

