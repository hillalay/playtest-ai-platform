"""
agents/standard_agents.py

Game-agnostic standart test agent'ları.

Bu dosyada üç temel agent bulunur:

1. RandomMaskedAgent
   Geçerli action'lar arasından tamamen rastgele seçim yapar.

2. HumanProxyAgent
   Basit epsilon/error-rate tabanlı insan davranışı baseline'ıdır.

3. SolverAgent
   Daha önce bir solver tarafından üretilmiş solution dizisini oynar.

Önemli:

    Solver != Agent

Solver çözüm üretir.

Agent çözümü / action'ları uygular.

Akış:

    Game Adapter
         ↓
      Solver
         ↓
    solution[]
         ↓
    SolverAgent
         ↓
      actions
"""

import random
from typing import Iterable, Optional

import numpy as np

from core.base_agent import BaseAgent


class RandomMaskedAgent(BaseAgent):
    """
    Geçerli action'lar arasından tamamen rastgele seçim yapan agent.

    Agent oyunun iç yapısını bilmez.

    Sadece action_mask kullanır.
    """

    def __init__(self) -> None:
        super().__init__(name="RandomMaskedAgent")

        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """
        Agent'ın random generator'ını seed eder.
        """
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """
        Geçerli action'lar arasından rastgele bir action seçer.

        Args:
            observation:
                Oyunun mevcut observation'ı.

                Random agent bunu kullanmaz.

            action_mask:
                Geçerli action'ları gösteren binary array.

        Returns:
            Seçilen action ID'si.
        """

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            raise ValueError(
                "RandomMaskedAgent: No valid actions available."
            )

        return int(self._rng.choice(valid_actions))


class HumanProxyAgent(BaseAgent):
    """
    Basit insan davranışı baseline agent'ı.

    error_rate / epsilon oranında rastgele geçerli bir
    action seçer.

    Geri kalan durumda deterministic olarak ilk geçerli
    action'ı seçer.

    Önemli:

        Bu gerçek bir insan davranış modeli değildir.

    Şimdilik playtest pipeline'ında basit bir human-like
    baseline olarak kullanılır.

    Örnek:

        error_rate = 0.20

        %20 -> random valid action
        %80 -> deterministic first valid action
    """

    def __init__(self, error_rate: float = 0.20) -> None:
        super().__init__(name="HumanProxyAgent")

        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(
                "error_rate must be between 0.0 and 1.0"
            )

        self.error_rate = float(error_rate)

        self._rng = random.Random()

    def set_seed(self, seed: int) -> None:
        """
        Agent'ın random generator'ını seed eder.
        """
        self._rng.seed(seed)

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """
        Mevcut action mask üzerinden action seçer.
        """

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            raise ValueError(
                "HumanProxyAgent: No valid actions available."
            )

        # İnsan hatası / dikkatsizlik anı
        if self._rng.random() < self.error_rate:
            return int(self._rng.choice(valid_actions))

        # Basit deterministic tercih
        return int(valid_actions[0])


class SolverAgent(BaseAgent):
    """
    Bir solver tarafından üretilmiş çözüm dizisini oynayan agent.

    SolverAgent çözüm üretmez.

    Kendisine verilen solution'ı sırayla uygular.

    Örneğin:

        solution = [2, 5, 8, 3]

    Agent:

        2
        5
        8
        3

    action'larını sırasıyla döndürür.

    Solver ile Agent arasındaki ayrım:

        Solver:
            Game State → Solution

        SolverAgent:
            Solution → Actions
    """

    def __init__(
        self,
        solution: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__(name="SolverAgent")

        self.solution = (
            [int(action) for action in solution]
            if solution is not None
            else []
        )

        self.solution_index = 0

    def set_solution(self, solution: Iterable[int]) -> None:
        """
        Agent'a yeni bir çözüm dizisi verir.

        Örneğin:

            [2, 5, 8, 3]
        """

        self.solution = [int(action) for action in solution]
        self.solution_index = 0

    def reset(self) -> None:
        """
        Yeni episode başladığında çözümün başından başlar.
        """

        self.solution_index = 0

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """
        Çözüm dizisindeki sıradaki action'ı döndürür.

        Agent'ın seçtiği action'ın gerçekten geçerli olup
        olmadığını Runner kontrol eder.
        """

        if self.solution_index >= len(self.solution):
            raise ValueError(
                "SolverAgent has no remaining actions in the solution."
            )

        action = self.solution[self.solution_index]

        self.solution_index += 1

        return int(action)


# ---------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------

# Eski isimlendirmeyi kullanan kodların kırılmasını önlemek
# için alias bırakıyoruz.

EpsilonRandomAgent = HumanProxyAgent

# Bu alias özellikle dikkat amaçlıdır:
#
# OptimalSolverAgent çözüm üretmez.
# Bir solver tarafından üretilen solution'ı oynar.
#
# Yeni kodda SolverAgent kullanılması tercih edilir.
OptimalSolverAgent = SolverAgent