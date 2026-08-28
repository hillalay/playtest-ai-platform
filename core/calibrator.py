"""
core/calibrator.py
Zorluk Kalibrasyonu, Tier 1-1000 Atayıcı ve Akış (Flow) Değerlendirici.
Kahn topolojik metrikleri ile simülasyon sonuçlarını birleştirerek
seviyeye 0.0 - 1.0 arası normalize zorluk skoru ve Tier derecesi atar.
"""

from typing import Any, Dict, Optional
import numpy as np


class DifficultyReport:
    """Hesaplanan zorluk ve analiz raporunu tutan veri yapısı."""
    def __init__(
        self,
        difficulty_index: float,
        tier: int,
        pacing_category: str,
        deceptive_risk_detected: bool,
        summary_metrics: Dict[str, Any]
    ):
        self.difficulty_index = difficulty_index
        self.tier = tier
        self.pacing_category = pacing_category
        self.deceptive_risk_detected = deceptive_risk_detected
        self.summary_metrics = summary_metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "difficulty_index": round(self.difficulty_index, 4),
            "tier": self.tier,
            "pacing_category": self.pacing_category,
            "deceptive_risk_detected": self.deceptive_risk_detected,
            "summary_metrics": self.summary_metrics
        }


class LevelCalibrator:
    """
    Simülasyon verilerini matematiksel zorluk katsayılarına ve
    oyun tasarımcılarının anlayacağı Tier derecelerine dönüştüren motor.
    """

    # Ağırlık Katsayıları (King & EA SEED Formülasyonu)
    WEIGHT_WIN_RATE = 0.35      # Başarısızlık oranı (1 - win_rate)
    WEIGHT_DEADLOCK = 0.25      # Tahtanın kilitlenme riski
    WEIGHT_BRANCHING = 0.20     # Karar anındaki seçenek çokluğu (bilişsel yük)
    WEIGHT_CHAIN_DEPTH = 0.10   # Bağımlılık zinciri uzunluğu
    WEIGHT_FREE_RATIO = 0.10    # Başlangıçta kilitli olma oranı (1 - free_ratio)

    @classmethod
    def calibrate(
        cls,
        sim_metrics: Dict[str, Any],
        kahn_metrics: Optional[Dict[str, Any]] = None,
        max_tier: int = 1000
    ) -> DifficultyReport:
        """
        Ham metrikleri alır, normalize eder ve Tier 1-1000 aralığına yerleştirir.
        """
        kahn = kahn_metrics or {}
        
        # 1. Ham Verilerin Alınması
        win_rate = sim_metrics.get("win_rate", 0.5)
        deadlock_rate = sim_metrics.get("deadlock_rate", 0.0)
        avg_branching = sim_metrics.get("avg_branching_factor", kahn.get("c3_branching_factor", 2.0))
        min_steps = sim_metrics.get("min_steps_to_win", kahn.get("c4_max_chain_depth", 10)) or 10
        free_ratio = kahn.get("c5_free_ratio", 0.3)

        # 2. Alt Skorların 0.0 - 1.0 Arasında Normalize Edilmesi
        score_win_loss = 1.0 - win_rate
        score_deadlock = deadlock_rate
        score_branching = min(avg_branching / 8.0, 1.0)        # 8 ve üzeri seçenek max bilişsel yük
        score_chain = min(min_steps / 25.0, 1.0)               # 25+ adım en uzun zincir
        score_free_blocked = 1.0 - min(free_ratio, 1.0)        # Başlangıçta az serbest ok = yüksek zorluk

        # 3. Nihai Ağırlıklı Zorluk İndeksi (DI ∈ [0.0, 1.0])
        difficulty_index = (
            score_win_loss * cls.WEIGHT_WIN_RATE +
            score_deadlock * cls.WEIGHT_DEADLOCK +
            score_branching * cls.WEIGHT_BRANCHING +
            score_chain * cls.WEIGHT_CHAIN_DEPTH +
            score_free_blocked * cls.WEIGHT_FREE_RATIO
        )
        difficulty_index = float(np.clip(difficulty_index, 0.0, 1.0))

        # 4. Tier Derecesi (1 ile max_tier / 1000 arası)
        tier = max(1, min(max_tier, int(round(difficulty_index * max_tier))))

        # 5. Oyun Akışı (Pacing) Kategorisi
        if tier <= 150:
            pacing_category = "tutorial_easy"
        elif tier <= 450:
            pacing_category = "casual_flow"
        elif tier <= 750:
            pacing_category = "tactical_challenge"
        else:
            pacing_category = "boss_difficulty"

        # 6. Sinsi Tuzak / Deceptive Risk Analizi (JAIST SCN Bulgusu)
        # Eğer optimal bot çok rahat geçiyor ama insan botunun kilitlenme oranı yüksekse
        deceptive_risk = (deadlock_rate > 0.30 and win_rate < 0.60)

        summary = {
            "win_rate": win_rate,
            "deadlock_rate": deadlock_rate,
            "min_steps_to_win": min_steps,
            "avg_branching_factor": avg_branching,
            "free_ratio": free_ratio
        }

        return DifficultyReport(
            difficulty_index=difficulty_index,
            tier=tier,
            pacing_category=pacing_category,
            deceptive_risk_detected=deceptive_risk,
            summary_metrics=summary
        )