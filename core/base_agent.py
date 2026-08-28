"""
core/base_agent.py

Tüm test ajanlarının ortak arayüzü.

Agent'lar oyunun iç yapısını bilmez.
Sadece observation ve action mask üzerinden karar verir.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """
    Oyundan bağımsız standart Agent arayüzü.

    Agent'ın görevi:
        observation + action_mask
            ↓
        action_id
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray
    ) -> int:
        """
        Mevcut oyun durumuna göre bir action seçer.

        Args:
            observation:
                Oyunun mevcut durumunu temsil eden numpy array.

            action_mask:
                Hangi action'ların geçerli olduğunu gösteren
                binary numpy array.

                Örneğin:

                    [1, 0, 1, 0, 1]

                1 = geçerli
                0 = geçersiz

        Returns:
            int:
                Seçilen action'ın ID'si.
        """
        pass

    def reset(self) -> None:
        """
        Episode başlamadan önce agent'ın geçici state'ini sıfırlar.

        Stateful agent'lar bunu override edebilir.

        Örneğin:
            - Solver'ın mevcut solution index'i
            - RL agent'ın episode memory'si
            - Human proxy'nin geçmiş davranışları
        """
        pass

