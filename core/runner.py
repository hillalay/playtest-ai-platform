"""
core/runner.py
Çok Çekirdekli Paralel Simülasyon Motoru (Game-Agnostic Simulation Runner).
Herhangi bir adaptörü ve ajanı alarak bir seviyeyi yüzlerce/binlerce kez 
paralel çekirdeklerde oynatır ve ham metrikleri toplar.
"""

from concurrent.futures import ProcessPoolExecutor
import time
import copy
from typing import Any, Dict, List, Type
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
    Tek bir simülasyon oyununu (episode) baştan sona oynatan bağımsız worker fonksiyonu.
    Multiprocessing uyumluluğu için modül seviyesinde tanımlanmıştır.
    """
    # 1. Oyunu başlat ve seviyeyi yükle
    game = adapter_cls()
    game.load_level(level_data)
    obs = game.reset()
    agent.reset()

    steps = 0
    branching_history = []
    action_history = []
    
    # 2. Oyun döngüsü
    while steps < max_steps:
        mask = game.get_action_mask()
        valid_action_count = int(np.sum(mask))
        
        # Eğer hiç geçerli hamle kalmadıysa (Kilitlenme / Deadlock)
        if valid_action_count == 0:
            return {
                "won": False,
                "reason": "deadlock",
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history
            }

        branching_history.append(valid_action_count)

        # Ajan hamle seçer
        action = agent.act(obs, mask)
        
        if not isinstance(action, (int, np.integer)):
            return { "won": False, 
                     "reason": "invalid_action_type", 
                     "steps": steps, 
                     "branching_history": branching_history, "action_history": action_history 
            }
        action = int(action)
        if action < 0 or action >= game.get_max_actions(): 
            return { "won": False, 
                    "reason": "invalid_action_range", 
                    "steps": steps, "action": action, "branching_history": branching_history, "action_history": action_history 
            }
            
        if action_mask[action] != 1: 
            return { "won": False, 
                    "reason": "invalid_action_mask", 
                    "steps": steps, "action": action, "branching_history": branching_history, "action_history": action_history 
            }
            
        action_history.append(action)
        # --------------------------------------------------------- # Action geçerli → oyuna gönder # ---------------------------------------------------------
        obs, reward, done, info = game.step(action) 
        steps += 1
        
        if done:
            is_win = (info.get("status") == "win" or info.get("reason") in ["cleared", "figure_rescued"])
            return {
                "won": is_win,
                "reason": info.get("reason", "cleared" if is_win else "loss"),
                "steps": steps,
                "branching_history": branching_history,
                "action_history": action_history
            }

    # Adım sınırı aşıldı (Zaman aşımı / Timeout)
    return {
        "won": False,
        "reason": "timeout",
        "steps": steps,
        "branching_history": branching_history,
        "action_history": action_history
    }


class SimulationRunner:
    """
    Seviyeleri çoklu çekirdekte paralel simüle eden merkezi motor.
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
        Verilen seviyeyi belirtilen ajan ile 'iterations' (örn: 500) kez oynatır
        ve istatistiksel sonuçları özetler.
        """
        start_time = time.time()

        # Çoklu işlemci havuzunda paralel simülasyon
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(_run_single_episode, adapter_cls, level_data, agent, max_steps)
                for _ in range(iterations)
            ]
            results = [f.result() for f in futures]

        elapsed_time = time.time() - start_time

        # Metriklerin Çıkarılması
        total_runs = len(results)
        wins = sum(1 for r in results if r["won"])
        deadlocks = sum(1 for r in results if r["reason"] == "deadlock")
        timeouts = sum(1 for r in results if r["reason"] == "timeout")
        
        winning_steps = [r["steps"] for r in results if r["won"]]
        all_branching = [
            b for r in results for b in r["branching_history"]
        ]

        return {
            "agent_name": agent.name,
            "total_simulations": total_runs,
            "win_rate": round(wins / total_runs, 4),
            "deadlock_rate": round(deadlocks / total_runs, 4),
            "timeout_rate": round(timeouts / total_runs, 4),
            "avg_steps_to_win": round(float(np.mean(winning_steps)), 2) if winning_steps else None,
            "min_steps_to_win": int(np.min(winning_steps)) if winning_steps else None,
            "avg_branching_factor": round(float(np.mean(all_branching)), 2) if all_branching else 0.0,
            "simulation_time_sec": round(elapsed_time, 3)
        }