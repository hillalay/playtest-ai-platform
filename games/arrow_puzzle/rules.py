"""
games/arrow_puzzle/rules.py
Arrow Puzzle GDD v1.1 ve TDD v1.1 standartlarına tam uyumlu Kural Motoru.
- Çok hücreli organik ok blokları (Multi-cell Snake Blocks)
- Katı Cisim (Rigid Body) Doğrusal Çıkış Kontrolü
- Rezerve Merkez Figür Engeli
- Blok Seviyesinde Aksiyon Maskeleme
"""

from typing import Any, Dict, List, Tuple, Set
import numpy as np
from core.base_adapter import BaseGameAdapter


class Direction:
    EMPTY = 0
    UP = 1     # Yukarı: dy = -1, dx = 0
    RIGHT = 2  # Sağ:    dy = 0,  dx = 1
    DOWN = 3   # Aşağı:  dy = 1,  dx = 0
    LEFT = 4   # Sol:    dy = 0,  dx = -1

    DELTA = {
        UP: (0, -1),
        RIGHT: (1, 0),
        DOWN: (0, 1),
        LEFT: (-1, 0),
    }


class CellType:
    PASSIVE = 0   # Silüet dışı boş alan (oklar geçebilir)
    ACTIVE = 1    # Okların yerleştiği alan
    RESERVED = 2  # Merkezdeki kurtarılacak figür (geçilemez engel)


class ArrowBlock:
    """Çok hücreli organik bir ok bloğunu temsil eder."""

    def __init__(self, block_id: int, direction: int, cells: List[Tuple[int, int]]):
        self.id = block_id
        self.direction = direction
        self.cells = cells  # [(x1, y1), (x2, y2), ...]

    @property
    def head(self) -> Tuple[int, int]:
        return self.cells[0]  # İlk hücre okun başıdır


class ArrowPuzzleAdapter(BaseGameAdapter):
    """
    Arrow Puzzle Resmi GDD v1.1 Kural Motoru Adaptörü.
    """

    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        # Passive / Active / Reserved
        self.mask_grid = np.zeros((height, width), dtype=np.int32)
        # Hangi hücrede hangi block_id var
        self.occupancy_grid = np.zeros((height, width), dtype=np.int32)
        self.blocks: Dict[int, ArrowBlock] = {}
        self.initial_level_data: Dict[str, Any] = {}
        self.lives = 3  # GDD 3.5 Can Sistemi

    def load_level(self, level_data: Dict[str, Any]) -> None:
        """
        JSON seviye verisini yükler.
        """
        self.initial_level_data = level_data
        self.width = level_data["width"]
        self.height = level_data["height"]
        self.reset()

    def reset(self) -> np.ndarray:
        """Tahtayı başlangıç durumuna getirir."""
        self.lives = 3
        self.blocks.clear()
        self.mask_grid = np.array(self.initial_level_data.get(
            "mask", np.zeros((self.height, self.width))
        ), dtype=np.int32)

        self.occupancy_grid = np.zeros(
            (self.height, self.width), dtype=np.int32)

        # Blokları yükle
        for b_data in self.initial_level_data.get("blocks", []):
            b_id = b_data["id"]
            direction = b_data["direction"]
            cells = [tuple(c) for c in b_data["cells"]]

            block = ArrowBlock(b_id, direction, cells)
            self.blocks[b_id] = block

            for cx, cy in cells:
                self.occupancy_grid[cy, cx] = b_id

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """AI için tahta durumunu döner (Doluluk + Maske)."""
        return np.copy(self.occupancy_grid)

    def can_block_exit(self, block_id: int) -> bool:
        """
        TDD Bölüm 10.1: Katı Cisim (Rigid Body) Doğrusal Tarama.
        Bloğun TÜM hücreleri, okun yönünde tahta dışına çıkana kadar 
        başka bir blokla veya Rezerve Merkez Figürle karşılaşmıyor mu?
        """
        if block_id not in self.blocks:
            return False

        block = self.blocks[block_id]
        dx, dy = Direction.DELTA[block.direction]
        block_cells_set = set(block.cells)

        # Bloğu oluşturan her bir hücreyi fırlama yönünde adım adım tara
        for cx, cy in block.cells:
            curr_x = cx + dx
            curr_y = cy + dy

            while 0 <= curr_x < self.width and 0 <= curr_y < self.height:
                # 1. Rezerve Merkez Figüre çarptı mı?
                if self.mask_grid[curr_y, curr_x] == CellType.RESERVED:
                    return False

                # 2. Başka bir ok bloğuna çarptı mı?
                occ_id = self.occupancy_grid[curr_y, curr_x]
                if occ_id != 0 and occ_id != block_id:
                    return False  # Yol tıkalı

                curr_x += dx
                curr_y += dy

        return True  # Tüm hücrelerin çıkış yolu tamamen açık

    def get_action_mask(self) -> np.ndarray:
        """
        Her bloğun ID'sine karşılık gelen hamle geçerlilik maskesi döner.
        """
        mask = np.zeros(self.get_max_actions(), dtype=np.int32)
        for b_id in self.blocks.keys():
            if self.can_block_exit(b_id):
                mask[b_id] = 1
        return mask

    def step(self, block_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Seçilen ID'li bloğu fırlatmayı dener.
        """
        if block_id not in self.blocks:
            return self._get_observation(), -0.5, False, {"status": "invalid_id"}

        # 1. Yol Tıkalıysa (GDD 3.5: Can -1)
        if not self.can_block_exit(block_id):
            self.lives -= 1
            is_dead = (self.lives <= 0)
            return (
                self._get_observation(),
                -1.0 if is_dead else -0.2,
                is_dead,
                {"status": "blocked", "lives_left": self.lives,
                    "reason": "out_of_lives" if is_dead else "hit_obstacle"}
            )

        # 2. Blok Başarıyla Fırlatıldı (Hücreleri Temizle)
        block = self.blocks.pop(block_id)
        for cx, cy in block.cells:
            self.occupancy_grid[cy, cx] = 0

        # 3. Kazanma Durumu (Tüm aktif oklar bitti, figür kurtarıldı)
        if len(self.blocks) == 0:
            return (
                self._get_observation(),
                1.0,
                True,
                {"status": "win", "reason": "figure_rescued"}
            )

        # 4. Kilitlenme (Deadlock) Durumu (Ok kaldı ama hiç çıkabilecek hamle yok)
        action_mask = self.get_action_mask()
        if np.sum(action_mask) == 0:
            return (
                self._get_observation(),
                -1.0,
                True,
                {"status": "loss", "reason": "deadlock"}
            )

        # 5. Başarılı Hamle Devam Ediyor
        return (
            self._get_observation(),
            0.1,
            False,
            {"status": "in_progress", "remaining_blocks": len(self.blocks)}
        )

    def get_max_actions(self) -> int:
        # Maksimum olası blok ID sayısı (örneğin 100)
        return max(100, len(self.initial_level_data.get("blocks", [])) + 1)

    def get_observation_shape(self) -> Tuple[int, ...]:
        return (self.height, self.width)

    def get_state_signature(self) -> str:
        return f"rem_{sorted(self.blocks.keys())}"
