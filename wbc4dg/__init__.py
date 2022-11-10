from gymnasium.envs.registration import register

from gymnasium_robotics.core import GoalEnv

def register_robotics_envs():
    def _merge(a, b):
        a.update(b)
        return a


register(
    id=f"PandaPush",
    entry_point="wbc4dg.envs.robo:MujocoPandaReachEnv",
    kwargs={"reward_type" : ""},
    max_episode_steps=50,
)

# wbc4dg.envs.mujoco: 추가
