# Display FetchReach-v3 example
import gymnasium as gym
import os 


# print(os.path.dirname(__file__))
env = gym.make("FetchReach-v3", max_episode_steps=100, render_mode="human")
# env = gym.make("FetchPush-v2", max_episode_steps=100, render_mode="human")
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # print(action)

env.close()
