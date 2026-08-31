"""
core/base_adapter.py

Tüm oyunların Playtest AI Platform'a bağlanması için
uygulaması gereken temel arayüz (Game-Agnostic Contract).

Platform oyunun iç yapısını bilmez.

Game-specific logic:

    Game Adapter

tarafından platformun standart contract'ına dönüştürülür.
"""

from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, Tuple

import numpy as np


class BaseGameAdapter(ABC):
    """
    Oyun ile Playtest AI Platform arasındaki standart arayüz.

    Platform oyunun iç yapısını bilmez.

    Oyuna ait tüm özel bilgiler adapter tarafından platformun
    anlayacağı standart formata dönüştürülür.
    """

    # ---------------------------------------------------------
    # Standard episode statuses
    # ---------------------------------------------------------

    STATUS_RUNNING = "running"
    STATUS_WIN = "win"
    STATUS_LOSS = "loss"
    STATUS_DEADLOCK = "deadlock"
    STATUS_TIMEOUT = "timeout"
    STATUS_INVALID_ACTION = "invalid_action"

    # ---------------------------------------------------------
    # Level lifecycle
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Environment interaction
    # ---------------------------------------------------------

    @abstractmethod
    def step(
        self,
        action_id: int,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Bir action uygular.

        Returns:

            observation:
                Hamle sonrasındaki oyun durumu.

            reward:
                Hamlenin ödül/cezası.

            done:
                Episode tamamlandı mı?

            info:
                Standart oyun sonucu bilgileri.

        Running:

            {
                "status": "running"
            }

        Win:

            {
                "status": "win",
                "reason": "figure_rescued"
            }

        Loss:

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

    # ---------------------------------------------------------
    # Action space
    # ---------------------------------------------------------

    @abstractmethod
    def get_action_mask(self) -> np.ndarray:
        """
        Mevcut durumda geçerli action'ları döndürür.

        Returns:
            np.ndarray:
                1 boyutlu binary action mask.

        Contract:

            len(mask) == get_max_actions()

            True / 1  -> action geçerli
            False / 0 -> action geçersiz

        Example:

            max_actions = 5

            mask = np.array(
                [True, False, True, False, True]
            )
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

    # ---------------------------------------------------------
    # Observation
    # ---------------------------------------------------------

    @abstractmethod
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Observation'ın beklenen boyutunu döndürür.

        Örneğin:

            (5, 5)

        veya:

            (5, 5, 3)
        """
        pass

    # ---------------------------------------------------------
    # State identity
    # ---------------------------------------------------------

    @abstractmethod
    def get_state_signature(self) -> str:
        """
        Mevcut oyun state'inin benzersiz imzasını döndürür.

        Solver ve loop detection tarafından kullanılır.

        Aynı oyun state'i her zaman aynı signature'ı
        üretmelidir.
        """
        pass

    # ---------------------------------------------------------
    # Cloning
    # ---------------------------------------------------------

    def clone(self) -> "BaseGameAdapter":
        """
        Mevcut adapter state'inin bağımsız bir kopyasını oluşturur.

        Solver gibi state-space search yapan sistemler tarafından
        kullanılır.

        Adapter'ın deepcopy ile uyumlu olması durumunda varsayılan
        implementation yeterlidir.

        Deepcopy uygun değilse game-specific adapter bu metodu
        override edebilir.
        """
        return copy.deepcopy(self)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def validate_action(self, action_id: int) -> bool:
        """
        Bir action'ın mevcut durumda geçerli olup olmadığını
        kontrol eder.

        Kontroller:

            - action integer mı?
            - negatif mi?
            - action space dışında mı?
            - action mask tarafından izin veriliyor mu?
        """

        if not isinstance(action_id, (int, np.integer)):
            return False

        action_id = int(action_id)

        try:
            action_mask = self.get_action_mask()
            max_actions = self.get_max_actions()
        except Exception:
            return False

        if not isinstance(action_mask, np.ndarray):
            return False

        if action_mask.ndim != 1:
            return False

        if len(action_mask) != max_actions:
            return False

        if action_mask.dtype.kind not in ("b", "i", "u"):
            return False

        if not np.all(np.isin(action_mask, [0, 1])):
            return False

        if action_id < 0 or action_id >= max_actions:
            return False

        return bool(action_mask[action_id] == 1)

    # ---------------------------------------------------------
    # Optional structural analysis
    # ---------------------------------------------------------

    def is_solvable_topologically(self) -> bool:
        """
        Opsiyonel yapısal solvability kontrolü.

        Her oyun için anlamlı olmak zorunda değildir.

        Bu nedenle varsayılan olarak True döndürür.

        Topolojik analiz destekleyen puzzle adapter'ları
        bu metodu override edebilir.
        """
        return True