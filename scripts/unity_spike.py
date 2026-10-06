from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from importlib.metadata import version


def get_valid_actions(decision_steps):
    """
    ML-Agents Python action_mask semantiği:
    True  = masked / disabled
    False = enabled

    Bu fonksiyon mask'i bizim için gerçek valid action ID listesine çevirir.
    """

    if decision_steps.action_mask is None:
        return None

    mask = decision_steps.action_mask[0][0]

    return [
        action_id
        for action_id, is_masked in enumerate(mask)
        if not is_masked
    ]


env = UnityEnvironment(
    file_name=None,
    timeout_wait=120
)

try:
    print("Unity bağlantısı bekleniyor...")

    env.reset()

    print("Unity bağlantısı başarılı.")

    behavior_name = list(env.behavior_specs.keys())[0]
    print("Behavior:", behavior_name)

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    print("\nİlk observation:")
    for obs in decision_steps.obs:
        print(obs)

    print("\nML-Agents Python (mlagents-envs):", version("mlagents-envs"))
    print("Not: Python action_mask içinde True = disabled, False = enabled.")

    print("\nInitial raw action mask:")
    print(decision_steps.action_mask)

    initial_valid_actions = get_valid_actions(decision_steps)

    print("Initial valid actions:")
    print(initial_valid_actions)

    input(
        "\nUnity ekranına bak. Arrow'lar duruyor olmalı. "
        "Action göndermek için ENTER'a bas..."
    )

    selected_action = 2

    actions = ActionTuple(
        discrete=np.array([[selected_action]], dtype=np.int32)
    )

    env.set_actions(behavior_name, actions)
    env.step()

    print(f"\nArrow {selected_action} seçildi.")

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    print("\nYeni observation:")
    for obs in decision_steps.obs:
        print(obs)

    print("\nNew raw action mask:")
    print(decision_steps.action_mask)

    new_valid_actions = get_valid_actions(decision_steps)

    print("New valid actions:")
    print(new_valid_actions)

    input(
        f"\nUnity ekranına bak. Arrow_{selected_action} kaybolmuş olmalı. "
        "Kapatmak için ENTER'a bas..."
    )

finally:
    env.close()