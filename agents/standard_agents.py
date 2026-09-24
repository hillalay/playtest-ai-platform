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
from core.personas import PersonaProfile


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
    Parametrik insan davranışı baseline agent'ı.

    HumanProxyAgent doğrudan bir PersonaProfile kullanır.

    Örneğin:

        HumanProxyAgent(DEFAULT_PERSONAS["rusher"])

    veya:

        HumanProxyAgent(DEFAULT_PERSONAS["cautious"])
    """

    def __init__(
        self,
        persona: PersonaProfile,
    ) -> None:
        super().__init__(
            name=f"HumanProxyAgent-{persona.name}"
        )

        self.persona = persona

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
        Persona profiline göre action seçer.

        Şimdilik yalnızca error_rate davranışını kullanır.
        Diğer persona parametreleri ilerleyen aşamalarda
        davranışa bağlanacaktır.
        """

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            raise ValueError(
                f"{self.name}: No valid actions available."
            )

        # Persona'nın hata davranışı
        if self._rng.random() < self.persona.error_rate:
            return int(
                self._rng.choice(valid_actions)
            )

        # Şimdilik mevcut baseline davranışını koruyoruz.
        return int(valid_actions[0])


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