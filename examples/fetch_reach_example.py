# Display FetchReach-v3 example
import gymnasium as gym


env = gym.make("FetchReach-v3", max_episode_steps=100, render_mode="human")

obs, info = env.reset()

for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    #print(obs)

env.close()
