"""
core/exporter.py

Unity JSON Freeze & Seviye Dışa Aktarım Boru Hattı.

Doğrulanmış ve simülasyonla Tier 1-1000 zorluk puanı
atanmış seviyeleri Unity C# istemcisinin tüketebileceği
standart JSON formatında kaydeder.

Pipeline:

    Generated Level
          ↓
    Simulation
          ↓
    Calibration
          ↓
    DifficultyReport
          ↓
       Exporter
          ↓
    Frozen JSON
          ↓
        Unity
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from core.calibrator import DifficultyReport


class LevelExporter:
    """
    Seviye verilerini standartlaştırılmış JSON dosyalarına
    basan modül.
    """

    # ---------------------------------------------------------
    # JSON serialization
    # ---------------------------------------------------------

    @staticmethod
    def _json_default(value: Any) -> Any:
        """
        NumPy gibi JSON tarafından doğrudan serialize
        edilemeyen tipleri standart Python tiplerine dönüştürür.
        """

        if isinstance(
            value,
            np.integer,
        ):
            return int(value)

        if isinstance(
            value,
            np.floating,
        ):
            return float(value)

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            np.bool_,
        ):
            return bool(value)

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable"
        )

    # ---------------------------------------------------------
    # Payload
    # ---------------------------------------------------------

    @staticmethod
    def format_level_payload(
        level_id: int,
        raw_level_data: Dict[str, Any],
        difficulty_report: DifficultyReport,
        topological_order: Optional[
            List[int]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Unity C# tarafından tüketilecek nihai veri şemasını
        oluşturur.
        """

        if level_id <= 0:
            raise ValueError(
                "level_id must be greater than 0"
            )

        warnings = []

        if difficulty_report.deceptive_risk_detected:
            warnings.append(
                "HIGH_DEADLOCK_RISK_DECEPTIVE_PUZZLE"
            )

        payload = {
            "schema_version": "1.1.0",

            "level_id": level_id,

            "tier": difficulty_report.tier,

            "difficulty_index": (
                difficulty_report.difficulty_index
            ),

            "pacing_category": (
                difficulty_report.pacing_category
            ),

            "warnings": warnings,

            "dimensions": {
                "width": raw_level_data.get(
                    "width",
                    10,
                ),
                "height": raw_level_data.get(
                    "height",
                    10,
                ),
            },

            "mask": raw_level_data.get(
                "mask",
                [],
            ),

            "blocks": raw_level_data.get(
                "blocks",
                [],
            ),

            "solution_metadata": {
                "topological_solution": (
                    topological_order
                    if topological_order is not None
                    else []
                ),

                "min_steps": (
                    difficulty_report.summary_metrics.get(
                        "min_steps_to_win",
                        0,
                    )
                    or 0
                ),

                "bot_win_rate": (
                    difficulty_report.summary_metrics.get(
                        "win_rate",
                        0.0,
                    )
                ),

                "deadlock_rate": (
                    difficulty_report.summary_metrics.get(
                        "deadlock_rate",
                        0.0,
                    )
                ),

                "avg_branching": (
                    difficulty_report.summary_metrics.get(
                        "avg_branching_factor",
                        0.0,
                    )
                ),
            },
        }

        return payload

    # ---------------------------------------------------------
    # Single export
    # ---------------------------------------------------------

    @classmethod
    def export_to_file(
        cls,
        level_id: int,
        raw_level_data: Dict[str, Any],
        difficulty_report: DifficultyReport,
        output_dir: str = "output_levels",
        topological_order: Optional[
            List[int]
        ] = None,
    ) -> str:
        """
        Tek bir seviyeyi:

            level_0001.json

        formatında diske kaydeder.

        Returns:
            Kaydedilen dosyanın tam yolu.
        """

        if level_id <= 0:
            raise ValueError(
                "level_id must be greater than 0"
            )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        payload = cls.format_level_payload(
            level_id=level_id,
            raw_level_data=raw_level_data,
            difficulty_report=difficulty_report,
            topological_order=topological_order,
        )

        filename = (
            f"level_{level_id:04d}.json"
        )

        file_path = os.path.join(
            output_dir,
            filename,
        )

        with open(
            file_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
                default=cls._json_default,
            )

        return file_path

    # ---------------------------------------------------------
    # Batch export
    # ---------------------------------------------------------

    @classmethod
    def export_batch_sorted_by_tier(
        cls,
        evaluated_levels: List[
            Dict[str, Any]
        ],
        output_dir: str = "output_levels",
    ) -> List[str]:
        """
        Üretilen seviyeleri difficulty/tier derecesine göre
        küçükten büyüğe sıralar.

        Daha sonra:

            Level 1
            Level 2
            ...
            Level N

        şeklinde export eder.

        Expected input:

            [
                {
                    "raw_level": {...},
                    "difficulty_report": DifficultyReport,
                    "topological_order": [...]
                }
            ]
        """

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Difficulty sıralaması
        # -----------------------------------------------------

        sorted_levels = sorted(
            evaluated_levels,
            key=lambda item: (
                item[
                    "difficulty_report"
                ].tier,

                item[
                    "difficulty_report"
                ].difficulty_index,

                item.get(
                    "level_id",
                    0,
                ),
            ),
        )

        # -----------------------------------------------------
        # Export
        # -----------------------------------------------------

        saved_files = []

        for new_id, item in enumerate(
            sorted_levels,
            start=1,
        ):

            raw_data = item[
                "raw_level"
            ]

            report = item[
                "difficulty_report"
            ]

            solution = item.get(
                "topological_order",
                [],
            )

            path = cls.export_to_file(
                level_id=new_id,
                raw_level_data=raw_data,
                difficulty_report=report,
                output_dir=output_dir,
                topological_order=solution,
            )

            saved_files.append(path)

        return saved_files