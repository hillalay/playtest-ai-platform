"""
agents/standard_agents.py
Herhangi bir oyunu test edebilen 3 temel standart ajan:
1. RandomMaskedAgent: Taban rastgele gürültü testi (Şansla geçilme oranı).
2. HumanProxyAgent: İnsan benzeri hata payı (epsilon) içeren ortalama oyuncu.
3. OptimalSolverAgent: Geriye dönük arama ile teorik minimum hamleyi bulan çözücü.
"""

import random
import copy
from typing import List
import numpy as np
from core.base_agent import BaseAgent



class RandomMaskedAgent(BaseAgent):
    """
    Geçerli action'lar arasından tamamen rastgele seçim yapan agent.

    Agent oyunun iç yapısını bilmez.
    Sadece action_mask kullanır.
    """

    def __init__(self):
        super().__init__(name="RandomMaskedAgent")

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray
    ) -> int:
        """
        Geçerli action'lar arasından rastgele bir action seçer.

        Args:
            observation:
                Oyunun mevcut observation'ı.
                Random agent bunu kullanmaz ancak BaseAgent
                contract'ının bir parçası olduğu için alınır.

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

        return int(np.random.choice(valid_actions))



class EpsilonRandomAgent(BaseAgent):
    """ 
   Epsilon-greedy tarzı basit bir baseline agent. error_rate / epsilon oranında rastgele geçerli bir action seçer. Geri kalan durumda deterministic olarak ilk geçerli action'ı seçer. NOT: Bu gerçek bir insan davranış modeli değildir. Şimdilik insan benzeri davranış için baseline olarak kullanılır. 
   """
    
    def __init__(self, error_rate: float = 0.20):
        super().__init__(name="EpsilonRandomAgent")
        
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0.0 and 1.0")
        self.error_rate = error_rate

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """ Mevcut action mask üzerinden action seçer. error_rate: Rastgele action seçme olasılığı. Örneğin: error_rate = 0.20 %20 → random valid action %80 → ilk valid action """
        
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            raise ValueError("EpsilonRandomAgent: No valid actions available.")
            return 0

        # İnsan hatası / dikkatsizlik anı
        if random.random() < self.error_rate:
            return int(np.random.choice(valid_actions))

        # Mantıklı seçim (varsayılan ilk geçerli hamle)
        return int(valid_actions[0])


class SolverAgent(BaseAgent):
    """
    Bir solver tarafından üretilmiş çözüm dizisini oynayan agent.

    SolverAgent çözüm üretmez.
    Kendisine verilen solution'ı sırayla uygular.

    Örneğin:

        solution = [2, 5, 8, 3]

    Agent sırasıyla:

        2
        5
        8
        3

    action'larını döndürür.
    """

    def __init__(self, solution=None):
        super().__init__(name="SolverAgent")

        self.solution = list(solution) if solution is not None else []
        self.solution_index = 0

    def set_solution(self, solution):
        """
        Agent'a yeni bir çözüm dizisi verir.

        Örneğin:

            [2, 5, 8, 3]
        """

        self.solution = list(solution)
        self.solution_index = 0

    def reset(self) -> None:
        """
        Yeni episode başladığında çözümün başından başlar.
        """

        self.solution_index = 0

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray
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

        action = int(self.solution[self.solution_index])

        self.solution_index += 1

        return action

