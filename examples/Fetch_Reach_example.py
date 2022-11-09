
import gymnasium as gym

# from stable_baselines3 import PPO

env = gym.make("FetchReach-v3", max_episode_steps=100, render_mode="human")

# model = PPO("MlpPolicy", env=env, n_steps=4)

# total_timesteps = int(1e4)
# model.learn(total_timesteps=total_timesteps, log_interval=10)
# model.save("./ppo_fetch_reach")

# del model

# model = PPO.load("ppo_fetch_reach", env=env)

# obs, info = env.reset()
# episode_reward = 0
# for _ in range(300):
#     action, _states = model.predict(obs,deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action=action)
#     episode_reward += reward

#     if terminated:
#         print("Episode Reward: ",episode_reward)
#         obs = env.reset

# env.close()


