"""
solvers/standard_solvers.py

Playtest AI Platform için standart solver implementasyonları.

Solver'ın görevi:

    Game Adapter
          ↓
       Solver
          ↓
      solution
          ↓
     SolverAgent
          ↓
       Runner

Solver oyunu doğrudan test etmez.

Bir GameAdapter üzerinde state-space search yaparak
çözüm action dizisi üretir.

Bu dosyadaki solver'lar game-agnostic'tir.

Oyunun:
    - kurallarını
    - level tasarımını
    - özel reason değerlerini
    - iç state yapısını

bilmezler.

Sadece BaseGameAdapter contract'ını kullanırlar.
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from core.base_adapter import BaseGameAdapter
from solvers.base_solver import BaseSolver


# =========================================================
# Helper Functions
# =========================================================

def _get_valid_actions(
    game_adapter: BaseGameAdapter,
) -> Optional[np.ndarray]:
    """
    Adapter'dan geçerli action'ları alır ve contract'a göre
    doğrular.

    Returns:
        np.ndarray:
            Geçerli action ID'leri.

        None:
            Action mask contract'a uygun değilse.

    Solver'ın oyunun iç yapısını bilmemesi için bütün
    validation yalnızca BaseGameAdapter contract'ına göre yapılır.
    """

    action_mask = game_adapter.get_action_mask()

    max_actions = game_adapter.get_max_actions()

    # -----------------------------------------------------
    # Mask type
    # -----------------------------------------------------

    if not isinstance(action_mask, np.ndarray):
        return None

    # -----------------------------------------------------
    # Mask dimensions
    # -----------------------------------------------------

    if action_mask.ndim != 1:
        return None

    # -----------------------------------------------------
    # Mask length
    # -----------------------------------------------------

    if len(action_mask) != max_actions:
        return None

    # -----------------------------------------------------
    # Mask dtype
    # -----------------------------------------------------

    if action_mask.dtype.kind not in (
        "b",
        "i",
        "u",
    ):
        return None

    # -----------------------------------------------------
    # Mask values
    # -----------------------------------------------------

    if not np.all(
        np.isin(
            action_mask,
            [0, 1],
        )
    ):
        return None

    # -----------------------------------------------------
    # Valid actions
    # -----------------------------------------------------

    return np.flatnonzero(action_mask)


def _is_win(
    info,
) -> bool:
    """
    Solver'ın terminal state'in kazanma durumu olup olmadığını
    kontrol eder.

    ÖNEMLİ:

    Bu fonksiyon oyun-specific reason değerlerini bilmez.

    Örneğin:

        "figure_rescued"
        "board_cleared"
        "target_destroyed"

    gibi reason'lara bakmaz.

    Yalnızca BaseGameAdapter contract'ındaki standart
    STATUS_WIN değerini kullanır.
    """

    if not isinstance(info, dict):
        return False

    return (
        info.get("status")
        == BaseGameAdapter.STATUS_WIN
    )


# =========================================================
# BFS Solver
# =========================================================

class BFSSolver(BaseSolver):
    """
    Breadth-First Search (BFS) kullanan standart solver.

    Her action'ın maliyetinin eşit olduğu varsayılır.

    Bu nedenle BFS, çözüm bulunduğunda teorik olarak
    minimum hamleli solution'ı döndürür.

    Örnek:

        solution = [
            2,
            5,
            8,
            3,
        ]

    Bu solution daha sonra SolverAgent tarafından
    oynanabilir.

    BFS game-agnostic'tir.

    Oyunun kurallarını bilmez.

    Sadece:

        get_action_mask()
        get_max_actions()
        step()
        get_state_signature()
        clone()

    gibi BaseGameAdapter contract'ını kullanır.
    """

    def __init__(
        self,
        max_depth: int = 50,
    ):
        """
        Args:
            max_depth:
                BFS'in arayacağı maksimum hamle derinliği.

        Raises:
            ValueError:
                max_depth 0 veya negatifse.
        """

        if max_depth <= 0:
            raise ValueError(
                "max_depth must be greater than 0"
            )

        self.max_depth = max_depth

    def solve(
        self,
        game_adapter: BaseGameAdapter,
    ) -> List[int]:
        """
        Verilen level için minimum hamleli çözüm arar.

        Args:
            game_adapter:
                Level yüklenmiş GameAdapter.

        Returns:
            List[int]:

                Çözüm bulunduysa action ID listesi.

                Örneğin:

                    [3, 7, 2, 5]

                Çözüm bulunamazsa:

                    []
        """

        # -----------------------------------------------------
        # Initial state
        # -----------------------------------------------------

        initial_state = game_adapter.clone()

        initial_state.reset()

        initial_signature = (
            initial_state.get_state_signature()
        )

        # -----------------------------------------------------
        # BFS queue
        # -----------------------------------------------------

        queue = deque(
            [
                (
                    initial_state,
                    [],
                )
            ]
        )

        # Aynı state'i tekrar ziyaret etmemek için
        # state signature tutuyoruz.
        visited_states = {
            initial_signature
        }

        # -----------------------------------------------------
        # BFS search
        # -----------------------------------------------------

        while queue:

            current_game, path = queue.popleft()

            # -------------------------------------------------
            # Maximum depth
            # -------------------------------------------------

            if len(path) >= self.max_depth:
                continue

            # -------------------------------------------------
            # Valid actions
            # -------------------------------------------------

            valid_actions = _get_valid_actions(
                current_game
            )

            if valid_actions is None:
                continue

            # Deadlock state
            if len(valid_actions) == 0:
                continue

            # -------------------------------------------------
            # Expand state
            # -------------------------------------------------

            for action in valid_actions:

                action = int(action)

                # Her branch'in bağımsız bir oyun state'i
                # üzerinde çalışması gerekir.
                cloned_game = current_game.clone()

                _, _, done, info = (
                    cloned_game.step(action)
                )

                new_path = path + [action]

                # -------------------------------------------------
                # Terminal state
                # -------------------------------------------------

                if done:

                    if _is_win(info):
                        return new_path

                    # Win olmayan terminal state'leri
                    # search'e eklemiyoruz.
                    continue

                # -------------------------------------------------
                # State signature
                # -------------------------------------------------

                state_signature = (
                    cloned_game.get_state_signature()
                )

                # Aynı state daha önce ziyaret edildiyse
                # tekrar queue'ya ekleme.
                if state_signature in visited_states:
                    continue

                visited_states.add(
                    state_signature
                )

                queue.append(
                    (
                        cloned_game,
                        new_path,
                    )
                )

        # -----------------------------------------------------
        # Solution not found
        # -----------------------------------------------------

        return []


# =========================================================
# DFS Solver
# =========================================================

class DFSSolver(BaseSolver):
    """
    Depth-First Search (DFS) kullanan standart solver.

    BFS gibi minimum çözümü garanti etmez.

    Avantajları:

        - Büyük state space'lerde BFS'e göre daha az
          memory kullanabilir.
        - Hızlı solvability kontrolü için kullanılabilir.
        - Baseline comparison yapılabilir.

    Dezavantajları:

        - Bulduğu ilk çözüm minimum hamleli olmak zorunda değildir.

    DFS de tamamen game-agnostic'tir.
    """

    def __init__(
        self,
        max_depth: int = 50,
    ):
        """
        Args:
            max_depth:
                Aranacak maksimum çözüm derinliği.

        Raises:
            ValueError:
                max_depth 0 veya negatifse.
        """

        if max_depth <= 0:
            raise ValueError(
                "max_depth must be greater than 0"
            )

        self.max_depth = max_depth

    def solve(
        self,
        game_adapter: BaseGameAdapter,
    ) -> List[int]:
        """
        DFS ile bir çözüm arar.

        Çözüm bulunamazsa:

            []

        döndürür.
        """

        # -----------------------------------------------------
        # Initial state
        # -----------------------------------------------------

        initial_state = game_adapter.clone()

        initial_state.reset()

        initial_signature = (
            initial_state.get_state_signature()
        )

        # -----------------------------------------------------
        # DFS stack
        # -----------------------------------------------------

        stack = [
            (
                initial_state,
                [],
            )
        ]

        # DFS için state'in hangi depth'te ziyaret edildiğini
        # takip ediyoruz.
        #
        # Bu, aynı state'e daha sığ bir depth'ten ulaşılması
        # durumunda search'ün yanlışlıkla engellenmesini önler.
        visited_depth = {
            initial_signature: 0
        }

        # -----------------------------------------------------
        # DFS search
        # -----------------------------------------------------

        while stack:

            current_game, path = stack.pop()

            current_depth = len(path)

            # -------------------------------------------------
            # Maximum depth
            # -------------------------------------------------

            if current_depth >= self.max_depth:
                continue

            # -------------------------------------------------
            # Valid actions
            # -------------------------------------------------

            valid_actions = _get_valid_actions(
                current_game
            )

            if valid_actions is None:
                continue

            # Deadlock state
            if len(valid_actions) == 0:
                continue

            # -------------------------------------------------
            # Expand state
            # -------------------------------------------------

            # Ters sırada push ederek action ID sırasını
            # deterministic hale getiriyoruz.
            for action in reversed(valid_actions):

                action = int(action)

                # Her branch'in bağımsız bir state üzerinde
                # çalışması gerekir.
                cloned_game = current_game.clone()

                _, _, done, info = (
                    cloned_game.step(action)
                )

                new_path = path + [action]

                new_depth = len(new_path)

                # -------------------------------------------------
                # Terminal state
                # -------------------------------------------------

                if done:

                    if _is_win(info):
                        return new_path

                    # Win olmayan terminal state'leri
                    # search'e eklemiyoruz.
                    continue

                # -------------------------------------------------
                # State signature
                # -------------------------------------------------

                state_signature = (
                    cloned_game.get_state_signature()
                )

                # -------------------------------------------------
                # Depth-aware visited check
                # -------------------------------------------------

                previous_depth = visited_depth.get(
                    state_signature
                )

                # Eğer bu state'e daha önce aynı veya daha
                # sığ bir depth'te ulaştıysak tekrar aramaya
                # gerek yok.
                if (
                    previous_depth is not None
                    and previous_depth <= new_depth
                ):
                    continue

                visited_depth[state_signature] = (
                    new_depth
                )

                stack.append(
                    (
                        cloned_game,
                        new_path,
                    )
                )

        # -----------------------------------------------------
        # Solution not found
        # -----------------------------------------------------

        return []