"""
core/base_adapter.py
Tüm oyunların 'playtest-ai-platform' motoruna bağlanması için 
uygulaması gereken temel soyut sınıf (Game-Agnostic Contract).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseGameAdapter(ABC):
    """
    Herhangi bir oyunun (Arrow Puzzle, Match-3, Grid vb.) simülatör,
    doğrulayıcı ve AI ajanlarıyla konuşabilmesini sağlayan adaptör arayüzü.
    """

    @abstractmethod
    def load_level(self, level_data: Dict[str, Any]) -> None:
        """
        Seviye verisini (JSON/Dict) oyuna yükler ve dahili tahta durumunu hazırlar.
        """
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Tahtayı başlangıç durumuna getirir ve ilk durum tensörünü (observation) döner.
        """
        pass

    @abstractmethod
    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Seçilen aksiyonu uygular.
        
        Dönüş Değeri:
            observation (np.ndarray): Yeni tahta durumu
            reward (float): Hamleden kazanılan ödül/ceza
            done (bool): Oyun bitti mi? (Kazanma veya Kilitlenme)
            info (dict): Ek hata/durum detayları (Örn: {"reason": "deadlock"})
        """
        pass

    @abstractmethod
    def get_action_mask(self) -> np.ndarray:
        """
        Aksiyon Maskeleme (Action Masking):
        Mevcut durumda hangi hamlelerin geçerli (1), hangilerinin kilitli/yasak (0) 
        olduğunu gösteren 1D ikili dizi (binary array) döner.
        """
        pass

    @abstractmethod
    def get_max_actions(self) -> int:
        """
        Oyundaki toplam olası aksiyon uzayı boyutu (Discrete Action Space Size).
        Örn: 5x5 tahta için 25.
        """
        pass

    @abstractmethod
    def get_observation_shape(self) -> Tuple[int, ...]:
        """
        Gözlem matrisinin boyutları. Örn: (5, 5) veya (Height, Width, Channels).
        """
        pass

    @abstractmethod
    def get_state_signature(self) -> str:
        """
        Tahtanın anlık durumunun benzersiz hash/string imzası.
        Döngü (loop) ve tekrar eden durumları anında tespit etmek için kullanılır.
        """
        pass

    def is_solvable_topologically(self) -> bool:
        """
        Varsayılan topolojik kontrol. Alt sınıflar (Örn: Kahn Algoritması)
        bunu ezerek matematiksel döngü tespiti yapabilir.
        """
        return True