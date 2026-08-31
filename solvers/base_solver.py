"""
solvers/base_solver.py

Oyun seviyelerini çözmek için kullanılan solver'ların
ortak arayüzünü tanımlar.

Solver'ın görevi:

    "Bu level nasıl çözülebilir?"

Solver, Agent değildir.

Solver bir solution üretir.

Örneğin:

    [2, 5, 8, 3]

Bu action dizisi daha sonra SolverAgent tarafından
oynanabilir.

Architecture:

    Game Adapter
          ↓
       Solver
          ↓
      solution
          ↓
     SolverAgent
          ↓
       Runner
"""

from abc import ABC, abstractmethod
from typing import List

from core.base_adapter import BaseGameAdapter


class BaseSolver(ABC):
    """
    Tüm solver'ların ortak arayüzü.

    Solver oyun kurallarını doğrudan bilmez.
    Game-specific state ve kurallar GameAdapter tarafından
    sağlanır.
    """

    @abstractmethod
    def solve(
        self,
        game_adapter: BaseGameAdapter,
    ) -> List[int]:
        """
        Verilen level için bir çözüm action dizisi üretir.

        Args:
            game_adapter:
                Çözülecek level'ın GameAdapter instance'ı.

        Returns:
            List[int]:
                Çözüm bulunduysa action ID listesi.

                Örneğin:

                    [2, 5, 8, 3]

                Çözüm bulunamazsa:

                    []
        """
        pass