"""
core/calibrator.py

Zorluk Kalibrasyonu, Tier 1-1000 Atayıcı ve Flow Değerlendirici.

Simulation sonuçlarını ve opsiyonel topolojik metrikleri
birleştirerek seviyeye:

    difficulty_index ∈ [0.0, 1.0]

ve:

    tier ∈ [1, 1000]

değeri atar.

Amaç:

    Raw Simulation Data
            +
    Structural Metrics
            ↓
      Difficulty Index
            ↓
       Tier 1-1000
            ↓
       Pacing Category
"""

from typing import Any, Dict, Optional

import numpy as np


class DifficultyReport:
    """
    Hesaplanan zorluk ve analiz raporunu tutan veri yapısı.
    """

    def __init__(
        self,
        difficulty_index: float,
        tier: int,
        pacing_category: str,
        deceptive_risk_detected: bool,
        summary_metrics: Dict[str, Any],
    ):
        self.difficulty_index = float(
            difficulty_index
        )

        self.tier = int(tier)

        self.pacing_category = pacing_category

        self.deceptive_risk_detected = bool(
            deceptive_risk_detected
        )

        self.summary_metrics = summary_metrics

    def to_dict(self) -> Dict[str, Any]:
        """
        Raporu JSON-compatible dictionary haline getirir.
        """

        return {
            "difficulty_index": round(
                self.difficulty_index,
                4,
            ),
            "tier": self.tier,
            "pacing_category": self.pacing_category,
            "deceptive_risk_detected": (
                self.deceptive_risk_detected
            ),
            "summary_metrics": self.summary_metrics,
        }


class LevelCalibrator:
    """
    Simulation verilerini matematiksel zorluk katsayılarına
    ve oyun tasarımcılarının anlayacağı Tier derecelerine
    dönüştüren motor.
    """

    # ---------------------------------------------------------
    # Difficulty weights
    # ---------------------------------------------------------

    WEIGHT_WIN_RATE = 0.35
    WEIGHT_DEADLOCK = 0.25
    WEIGHT_BRANCHING = 0.20
    WEIGHT_CHAIN_DEPTH = 0.10
    WEIGHT_FREE_RATIO = 0.10

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """
        Bir değeri belirtilen aralıkta tutar.
        """

        return float(
            np.clip(
                value,
                minimum,
                maximum,
            )
        )

    @classmethod
    def calibrate(
        cls,
        sim_metrics: Dict[str, Any],
        kahn_metrics: Optional[
            Dict[str, Any]
        ] = None,
        max_tier: int = 1000,
    ) -> DifficultyReport:
        """
        Ham metrikleri alır, normalize eder ve
        Tier 1-max_tier aralığına yerleştirir.
        """

        if max_tier <= 0:
            raise ValueError(
                "max_tier must be greater than 0"
            )

        if not isinstance(sim_metrics, dict):
            raise TypeError(
                "sim_metrics must be a dictionary"
            )

        kahn = (
            kahn_metrics
            if kahn_metrics is not None
            else {}
        )

        # -----------------------------------------------------
        # 1. Raw metrics
        # -----------------------------------------------------

        win_rate = cls._clamp(
            float(
                sim_metrics.get(
                    "win_rate",
                    0.5,
                )
            )
        )

        deadlock_rate = cls._clamp(
            float(
                sim_metrics.get(
                    "deadlock_rate",
                    0.0,
                )
            )
        )

        avg_branching = max(
            0.0,
            float(
                sim_metrics.get(
                    "avg_branching_factor",
                    kahn.get(
                        "c3_branching_factor",
                        2.0,
                    ),
                )
                or 0.0
            ),
        )

        min_steps_raw = sim_metrics.get(
            "min_steps_to_win"
        )

        if min_steps_raw is None:
            min_steps_raw = kahn.get(
                "c4_max_chain_depth"
            )

        min_steps = (
            float(min_steps_raw)
            if min_steps_raw is not None
            else None
        )

        if min_steps is not None:
            min_steps = max(
                0.0,
                min_steps,
            )

        free_ratio = cls._clamp(
            float(
                kahn.get(
                    "c5_free_ratio",
                    0.3,
                )
            )
        )

        # -----------------------------------------------------
        # 2. Normalized scores
        # -----------------------------------------------------

        # Düşük win rate = yüksek difficulty
        score_win_loss = (
            1.0 - win_rate
        )

        # Yüksek deadlock = yüksek difficulty
        score_deadlock = deadlock_rate

        # Çok fazla branch = daha yüksek cognitive load
        score_branching = cls._clamp(
            avg_branching / 8.0
        )

        # Minimum solution depth
        #
        # Eğer çözüm uzunluğu bilinmiyorsa 25'i
        # üst sınır olarak kullanıyoruz.
        chain_depth_for_score = (
            min_steps
            if min_steps is not None
            else 25.0
        )

        score_chain = cls._clamp(
            chain_depth_for_score / 25.0
        )

        # Başlangıçta az free action = daha yüksek difficulty
        score_free_blocked = (
            1.0 - free_ratio
        )

        # -----------------------------------------------------
        # 3. Weighted Difficulty Index
        # -----------------------------------------------------

        difficulty_index = (
            score_win_loss
            * cls.WEIGHT_WIN_RATE
            +
            score_deadlock
            * cls.WEIGHT_DEADLOCK
            +
            score_branching
            * cls.WEIGHT_BRANCHING
            +
            score_chain
            * cls.WEIGHT_CHAIN_DEPTH
            +
            score_free_blocked
            * cls.WEIGHT_FREE_RATIO
        )

        difficulty_index = cls._clamp(
            difficulty_index
        )

        # -----------------------------------------------------
        # 4. Tier
        # -----------------------------------------------------

        tier = max(
            1,
            min(
                max_tier,
                int(
                    round(
                        difficulty_index
                        * max_tier
                    )
                ),
            ),
        )

        # -----------------------------------------------------
        # 5. Pacing category
        # -----------------------------------------------------

        if tier <= 150:
            pacing_category = "tutorial_easy"

        elif tier <= 450:
            pacing_category = "casual_flow"

        elif tier <= 750:
            pacing_category = "tactical_challenge"

        else:
            pacing_category = "boss_difficulty"

        # -----------------------------------------------------
        # 6. Deceptive puzzle detection
        # -----------------------------------------------------

        optimal_win_rate = sim_metrics.get(
            "optimal_win_rate"
        )

        if optimal_win_rate is None:
            optimal_win_rate = kahn.get(
                "optimal_win_rate"
            )

        if optimal_win_rate is not None:

            optimal_win_rate = cls._clamp(
                float(optimal_win_rate)
            )

            deceptive_risk = (
                optimal_win_rate >= 0.95
                and win_rate < 0.60
                and deadlock_rate > 0.30
            )

        else:
            # Backward-compatible fallback.
            deceptive_risk = (
                deadlock_rate > 0.30
                and win_rate < 0.60
            )

        # -----------------------------------------------------
        # 7. Summary
        # -----------------------------------------------------

        summary = {
            "win_rate": win_rate,
            "deadlock_rate": deadlock_rate,
            "min_steps_to_win": (
                int(min_steps)
                if min_steps is not None
                else None
            ),
            "avg_branching_factor": (
                avg_branching
            ),
            "free_ratio": free_ratio,
        }

        if optimal_win_rate is not None:
            summary[
                "optimal_win_rate"
            ] = optimal_win_rate

        return DifficultyReport(
            difficulty_index=difficulty_index,
            tier=tier,
            pacing_category=pacing_category,
            deceptive_risk_detected=deceptive_risk,
            summary_metrics=summary,
        )