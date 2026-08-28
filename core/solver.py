"""
Oyun seviyelerini çözmek için kullanılan solver'lar.

Solver'ın görevi:
    "Bu level nasıl çözülebilir?"

Solver, Agent değildir.

Solver bir solution üretir.
Örneğin:

    [2, 5, 8, 3]

Bu action dizisi daha sonra SolverAgent tarafından oynanabilir.
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
    def solve(self, game_adapter: BaseGameAdapter) -> List[int]:
        """
        Verilen level için bir çözüm action dizisi üretir.

        Örneğin:

            [2, 5, 8, 3]

        """

        pass


class BFSSolver(BaseSolver):
    """
    Breadth-First Search (BFS) kullanan solver.

    BFS, her hamlenin maliyeti eşit kabul edildiğinde
    en az hamleli çözümü bulur.
    """

    def __init__(self, max_depth: int = 50):
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")

        self.max_depth = max_depth

    def solve(self, game_adapter: BaseGameAdapter) -> List[int]:
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

        initial_state = copy.deepcopy(game_adapter)
        initial_state.reset()

        initial_signature = initial_state.get_state_signature()

        # Queue elemanı:
        #
        # (
        #     oyun_state,
        #     o ana kadar yapılan action'lar
        # )
        #
        queue = deque([
            (initial_state, [])
        ])

        visited_states = {
            initial_signature
        }

        while queue:
            current_game, path = queue.popleft()

            # Maksimum çözüm derinliğine ulaştıysak
            # bu state'i daha fazla genişletme.
            if len(path) >= self.max_depth:
                continue

            action_mask = current_game.get_action_mask()

            valid_actions = np.flatnonzero(action_mask)

            for action in valid_actions:
                action = int(action)

                cloned_game = copy.deepcopy(current_game)

                _, _, done, info = cloned_game.step(action)

                new_path = path + [action]

                # Standart platform contract'ımızı kullanıyoruz.
                if done and info.get("status") == BaseGameAdapter.STATUS_WIN:
                    return new_path

                # Oyun bitmediyse yeni state'i incele.
                if done:
                    continue

                state_signature = cloned_game.get_state_signature()

                # Aynı state'i daha önce gördüysek
                # tekrar incelememize gerek yok.
                if state_signature in visited_states:
                    continue

                visited_states.add(state_signature)

                queue.append(
                    (cloned_game, new_path)
                )

        # Çözüm bulunamadı.
        return []

