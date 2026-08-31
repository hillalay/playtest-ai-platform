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
Bir GameAdapter üzerinde search yaparak çözüm action dizisi üretir.

Bu dosyadaki solver'lar game-agnostic'tir.
Oyunun kurallarını bilmezler.

Sadece BaseGameAdapter contract'ını kullanırlar.
"""

from collections import deque
from typing import List

import numpy as np

from core.base_adapter import BaseGameAdapter
from solvers.base_solver import BaseSolver


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
    """

    def __init__(self, max_depth: int = 50):
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
        # state signature'larını tutuyoruz.
        visited_states = {
            initial_signature
        }

        # -----------------------------------------------------
        # BFS search
        # -----------------------------------------------------

        while queue:

            current_game, path = queue.popleft()

            # Maksimum derinliğe ulaşıldıysa
            # bu state'i genişletme.
            if len(path) >= self.max_depth:
                continue

            # -------------------------------------------------
            # Action mask
            # -------------------------------------------------

            action_mask = current_game.get_action_mask()

            max_actions = current_game.get_max_actions()

            # Adapter contract validation.
            if not isinstance(
                action_mask,
                np.ndarray,
            ):
                continue

            if action_mask.ndim != 1:
                continue

            if len(action_mask) != max_actions:
                continue

            if not np.all(
                np.isin(
                    action_mask,
                    [0, 1],
                )
            ):
                continue

            # -------------------------------------------------
            # Valid actions
            # -------------------------------------------------

            valid_actions = np.flatnonzero(
                action_mask
            )

            # Hiçbir action yoksa deadlock state.
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

                    status = info.get("status")
                    reason = info.get("reason")

                    is_win = (
                        status
                        == BaseGameAdapter.STATUS_WIN
                        or reason in (
                            "cleared",
                            "figure_rescued",
                        )
                    )

                    # Çözüm bulundu.
                    if is_win:
                        return new_path

                    # Loss / deadlock / timeout gibi
                    # terminal state'leri search'e ekleme.
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


class DFSSolver(BaseSolver):
    """
    Depth-First Search (DFS) kullanan standart solver.

    BFS gibi minimum çözümü garanti etmez.

    Avantajı:
        Büyük state space'lerde belleği BFS'e göre
        daha kontrollü kullanabilir.

    Dezavantajı:
        Bulduğu ilk çözüm minimum hamleli olmak zorunda değildir.

    Bu solver özellikle:
        - hızlı solvability kontrolü
        - geniş state space
        - baseline comparison

    için kullanılabilir.
    """

    def __init__(self, max_depth: int = 50):
        """
        Args:
            max_depth:
                Aranacak maksimum çözüm derinliği.
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

        Çözüm bulunamazsa [] döndürür.
        """

        initial_state = game_adapter.clone()

        initial_state.reset()

        initial_signature = (
            initial_state.get_state_signature()
        )

        # Stack elemanı:
        #
        # (
        #     game_state,
        #     path
        # )
        #
        stack = [
            (
                initial_state,
                [],
            )
        ]

        visited_states = {
            initial_signature
        }

        while stack:

            current_game, path = stack.pop()

            if len(path) >= self.max_depth:
                continue

            action_mask = current_game.get_action_mask()

            max_actions = current_game.get_max_actions()

            if not isinstance(
                action_mask,
                np.ndarray,
            ):
                continue

            if action_mask.ndim != 1:
                continue

            if len(action_mask) != max_actions:
                continue

            if not np.all(
                np.isin(
                    action_mask,
                    [0, 1],
                )
            ):
                continue

            valid_actions = np.flatnonzero(
                action_mask
            )

            if len(valid_actions) == 0:
                continue

            # Ters sırada push ederek action ID sırasını
            # deterministic hale getiriyoruz.
            for action in reversed(valid_actions):

                action = int(action)

                cloned_game = current_game.clone()

                _, _, done, info = (
                    cloned_game.step(action)
                )

                new_path = path + [action]

                if done:

                    status = info.get("status")
                    reason = info.get("reason")

                    is_win = (
                        status
                        == BaseGameAdapter.STATUS_WIN
                        or reason in (
                            "cleared",
                            "figure_rescued",
                        )
                    )

                    if is_win:
                        return new_path

                    continue

                state_signature = (
                    cloned_game.get_state_signature()
                )

                if state_signature in visited_states:
                    continue

                visited_states.add(
                    state_signature
                )

                stack.append(
                    (
                        cloned_game,
                        new_path,
                    )
                )

        return []