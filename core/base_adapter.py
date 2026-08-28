"""
core/base_adapter.py

Tüm oyunların Playtest AI Platform'a bağlanması için
uygulaması gereken temel arayüz (Game-Agnostic Contract).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseGameAdapter(ABC):
    """
    Oyun ile Playtest AI Platform arasındaki standart arayüz.

    Platform oyunun iç yapısını bilmez.
    Oyuna ait tüm özel bilgiler adapter tarafından platformun
    anlayacağı standart formata dönüştürülür.
    """

    # Oyunun standart durumları.
    STATUS_RUNNING = "running"
    STATUS_WIN = "win"
    STATUS_LOSS = "loss"
    STATUS_DEADLOCK = "deadlock"
    STATUS_TIMEOUT = "timeout"
    STATUS_INVALID_ACTION = "invalid_action"

    @abstractmethod
    def load_level(self, level_data: Dict[str, Any]) -> None:
        """
        Seviye verisini oyuna yükler.

        Örneğin:
            {
                "width": 5,
                "height": 5,
                "blocks": [...]
            }

        Adapter bu veriyi kendi oyun state'ine dönüştürür.
        """
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Oyunu başlangıç durumuna getirir.

        Returns:
            np.ndarray:
                Oyunun başlangıç observation'ı.
        """
        pass

    @abstractmethod
    def step(
        self,
        action_id: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Bir aksiyonu uygular.

        Returns:
            observation:
                Hamle sonrasındaki oyun durumu.

            reward:
                Hamlenin ödül/cezası.

            done:
                Episode tamamlandı mı?

            info:
                Standart oyun sonucu bilgileri.

        Örnek:

            {
                "status": "running"
            }

        Kazanma:

            {
                "status": "win",
                "reason": "figure_rescued"
            }

        Kaybetme:

            {
                "status": "loss",
                "reason": "player_failed"
            }

        Deadlock:

            {
                "status": "deadlock",
                "reason": "no_valid_actions"
            }
        """
        pass

    @abstractmethod
    def get_action_mask(self) -> np.ndarray:
        """
        Mevcut durumda geçerli aksiyonları döndürür.

        Örnek:

            [1, 0, 1, 0, 1]

        Burada:
            1 = aksiyon geçerli
            0 = aksiyon geçersiz
        """
        pass

    @abstractmethod
    def get_max_actions(self) -> int:
        """
        Oyunun toplam discrete action space boyutunu döndürür.

        Örneğin 5x5 grid:

            25

        dönebilir.
        """
        pass

    @abstractmethod
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Observation'ın boyutunu döndürür.

        Örneğin:

            (5, 5)

        veya:

            (5, 5, 3)
        """
        pass

    @abstractmethod
    def get_state_signature(self) -> str:
        """
        Mevcut oyun state'inin benzersiz imzasını döndürür.

        Solver ve loop detection tarafından kullanılır.

        Aynı oyun state'i her zaman aynı signature'ı üretmelidir.
        """
        pass

    def validate_action(self, action_id: int) -> bool:
        """
        Bir action'ın mevcut durumda geçerli olup olmadığını kontrol eder.

        Bu kontrol Runner tarafından kullanılabilir.

        Böylece agent yanlışlıkla:
            - negatif action
            - action space dışındaki action
            - maskelenmiş action

        seçerse bunu tespit edebiliriz.
        """

        action_mask = self.get_action_mask()

        if action_id < 0 or action_id >= len(action_mask):
            return False

        return bool(action_mask[action_id] == 1)

    def is_solvable_topologically(self) -> bool:
        """
        Opsiyonel yapısal solvability kontrolü.

        Her oyun için anlamlı olmak zorunda değildir.
        Bu nedenle varsayılan olarak True döndürür.

        Topolojik analiz destekleyen puzzle adapterları
        bu metodu override edebilir.
        """
        return True

