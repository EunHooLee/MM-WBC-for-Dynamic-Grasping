import gymnasium as gym
from stable_baselines3.sac import sac

env = gym.make('Ant-v4',render_mode = "human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info, _ = env.step(action)

   if terminated :
      observation, info = env.reset()
# env.close()zip