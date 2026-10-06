from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

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
    print(" New Action mask")
    print(decision_steps.action_mask)

    print("İlk observation:")
    for obs in decision_steps.obs:
        print(obs)

    input("\nUnity ekranına bak. Arrow'lar duruyor olmalı. Action göndermek için ENTER'a bas...")

    actions = ActionTuple(
        discrete=np.array([[2]], dtype=np.int32)
    )

    env.set_actions(behavior_name, actions)
    env.step()

    print("\nArrow 2 seçildi.")

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    print("Yeni observation:")
    for obs in decision_steps.obs:
        print(obs)

    input("\nUnity ekranına bak. Arrow_2 kaybolmuş olmalı. Kapatmak için ENTER'a bas...")

finally:
    env.close()