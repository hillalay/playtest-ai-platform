"""
core/base_agent.py

Tüm test ajanlarının ortak arayüzü.

Agent'lar oyunun iç yapısını bilmez.
Sadece observation ve action mask üzerinden karar verir.

Contract:

    observation + action_mask
            ↓
        action_id
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

    Agent oyun kurallarını bilmez.
    Game-specific logic adapter katmanında bulunur.
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """
        Mevcut oyun durumuna göre bir action seçer.

        Args:
            observation:
                Oyunun mevcut observation'ını temsil eden
                numpy array.

            action_mask:
                Mevcut durumda hangi action'ların geçerli olduğunu
                gösteren 1 boyutlu numpy array.

                Maskenin uzunluğu action space'in toplam
                boyutuna eşit olmalıdır.

                True / 1  -> action geçerli
                False / 0 -> action geçersiz

                Örneğin:

                    np.array(
                        [True, False, True, False, True]
                    )

        Returns:
            int:
                Seçilen action'ın ID'si.

        Notes:
            Agent'ın döndürdüğü action'ın gerçekten geçerli olup
            olmadığını doğrulamak Runner'ın sorumluluğundadır.
        """
        pass

    def reset(self) -> None:
        """
        Episode başlamadan önce agent'ın geçici state'ini sıfırlar.

        Stateless agent'lar bu metodu değiştirmek zorunda değildir.

        Stateful agent'lar bunu override edebilir.

        Örneğin:

            - Solver'ın solution index'i
            - RL agent'ın episode memory'si
            - Human proxy'nin geçmiş davranışları
        """
        pass

    def set_seed(self, seed: int) -> None:
        """
        Agent'ın random davranışları için seed ayarlar.

        Varsayılan implementation stateless agent'lar için
        herhangi bir işlem yapmaz.

        Stateful / stochastic agent'lar bunu override edebilir.

        Runner multiprocessing kullandığında her episode'a
        farklı bir seed verilmesini sağlar.
        """
        pass