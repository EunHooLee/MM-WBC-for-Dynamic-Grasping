import gymnasium as gym

from wbc4dg.envs.mujoco import MujocoPandaReachEnv

env = MujocoPandaReachEnv(reward_type="sparse")

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # print(action)

env.close()
