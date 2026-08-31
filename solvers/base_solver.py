"""
solvers/base_solver.py

Oyun seviyelerini çözmek için kullanılan solver'lar.

Solver'ın görevi:

    "Bu level nasıl çözülebilir?"

Solver, Agent değildir.

Solver bir solution üretir.

Örneğin:

    [2, 5, 8, 3]

Bu action dizisi daha sonra SolverAgent tarafından
oynanabilir.

Architecture:

    Game Adapter
          ↓
       Solver
          ↓
      solution
          ↓
     SolverAgent
          ↓
       Runner
"""

import copy
from abc import ABC, abstractmethod
from collections import deque
from typing import List

import numpy as np

from core.base_adapter import BaseGameAdapter


class BaseSolver(ABC):
    """
    Tüm solver'ların ortak arayüzü.
    """

    @abstractmethod
    def solve(
        self,
        game_adapter: BaseGameAdapter,
    ) -> List[int]:
        """
        Verilen level için bir çözüm action dizisi üretir.

        Örneğin:

            [2, 5, 8, 3]

        Çözüm bulunamazsa:

            []

        döndürülür.
        """
        pass


class BFSSolver(BaseSolver):
    """
    Breadth-First Search (BFS) kullanan solver.

    Her action'ın maliyeti eşit kabul edildiğinde
    BFS minimum hamleli çözümü bulur.

    Bu nedenle:

        BFS solution

    teorik olarak shortest-path solution'dır.
    """

    def __init__(self, max_depth: int = 50):
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
        Level için minimum hamleli çözümü arar.

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

        # Queue elemanı:
        #
        # (
        #     oyun_state,
        #     o ana kadar yapılan action'lar
        # )
        #
        queue = deque(
            [
                (
                    initial_state,
                    [],
                )
            ]
        )

        visited_states = {
            initial_signature
        }

        # -----------------------------------------------------
        # BFS
        # -----------------------------------------------------

        while queue:

            current_game, path = queue.popleft()

            # Maksimum çözüm derinliğine ulaştıysak
            # bu state'i daha fazla genişletme.
            if len(path) >= self.max_depth:
                continue

            # -------------------------------------------------
            # Action mask
            # -------------------------------------------------

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

            if action_mask.dtype.kind not in (
                "b",
                "i",
                "u",
            ):
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

            # Deadlock state
            if len(valid_actions) == 0:
                continue

            # -------------------------------------------------
            # Expand actions
            # -------------------------------------------------

            for action in valid_actions:

                action = int(action)

                # Adapter'ın bağımsız clone'unu oluştur.
                cloned_game = current_game.clone()

                _, _, done, info = (
                    cloned_game.step(action)
                )

                new_path = path + [action]

                # -------------------------------------------------
                # Win
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

                    if is_win:
                        return new_path

                    # Loss/deadlock/timeout gibi terminal
                    # state'leri search'e eklemiyoruz.
                    continue

                # -------------------------------------------------
                # State signature
                # -------------------------------------------------

                state_signature = (
                    cloned_game.get_state_signature()
                )

                # Aynı state daha önce görüldüyse tekrar
                # incelemeye gerek yok.
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