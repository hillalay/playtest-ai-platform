
from importlib.metadata import version

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


def get_valid_actions(decision_steps):
    """ML-Agents: True = masked, False = enabled."""
    if decision_steps.action_mask is None:
        return None

    mask = decision_steps.action_mask[0][0]

    return [
        index
        for index, is_masked in enumerate(mask)
        if not is_masked
    ]


def get_observation(decision_steps):
    """Tek ajanlı test ortamının observation kopyası."""
    assert len(decision_steps) == 1, (
        "Bu spike tam olarak bir DecisionStep bekliyor."
    )

    return [
        np.array(obs, copy=True)
        for obs in decision_steps.obs
    ]


def assert_same_observations(first, second):
    assert len(first) == len(second), (
        "Observation tensor sayısı değişti."
    )

    for index, (a, b) in enumerate(zip(first, second)):
        np.testing.assert_array_equal(
            a,
            b,
            err_msg=f"Reset sonrası observation {index} farklı."
        )


env = UnityEnvironment(
    file_name=None,
    timeout_wait=120,
)

try:
    print("Unity bağlantısı bekleniyor...")
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    print("Unity bağlantısı başarılı.")
    print("Behavior:", behavior_name)
    print("ML-Agents:", version("mlagents-envs"))

    # 1. Başlangıç state'ini kaydet.
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    initial_observation = get_observation(decision_steps)
    initial_valid_actions = get_valid_actions(decision_steps)

    assert initial_valid_actions, (
        "Başlangıçta geçerli action bulunamadı."
    )

    print("\nBaşlangıç observation:")
    for obs in initial_observation:
        print(obs)

    print("Başlangıç valid actions:", initial_valid_actions)

    input("\nAction göndermek için ENTER'a bas...")

    # 2. Geçerli bir action uygula.
    selected_action = initial_valid_actions[0]

    actions = ActionTuple(
        discrete=np.array(
            [[selected_action]],
            dtype=np.int32
        )
    )

    env.set_actions(behavior_name, actions)
    env.step()

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    print("\nGönderilen action:", selected_action)

    if len(terminal_steps) > 0:
        print("Action sonrasında episode terminal oldu.")
    else:
        changed_observation = get_observation(decision_steps)
        new_valid_actions = get_valid_actions(decision_steps)

        print("Yeni valid actions:", new_valid_actions)

        assert any(
            not np.array_equal(before, after)
            for before, after in zip(
                initial_observation,
                changed_observation
            )
        ), "Action sonrası observation değişmedi."

        print("PASS: Action sonrası observation değişti.")

    # 3. Python üzerinden reset.
    input("\nLevel'i resetlemek için ENTER'a bas...")

    env.reset()

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    reset_observation = get_observation(decision_steps)
    reset_valid_actions = get_valid_actions(decision_steps)

    print("\nReset sonrası observation:")
    for obs in reset_observation:
        print(obs)

    print("Reset sonrası valid actions:", reset_valid_actions)

    # 4. Deterministic reset doğrulaması.
    assert_same_observations(
        initial_observation,
        reset_observation
    )

    assert initial_valid_actions == reset_valid_actions, (
        "Reset sonrası action mask başlangıçla aynı değil."
    )

    print("\nPASS: Reset sonrası observation aynı.")
    print("PASS: Reset sonrası valid actions aynı.")
    print("PASS: Reset doğrulaması başarılı!")

finally:
    env.close()
