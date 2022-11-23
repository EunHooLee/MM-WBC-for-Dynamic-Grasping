import gymnasium as gym
# from stable_baselines3.stable_baselines3.ppo.ppo import PPO
env = gym.make("Ant-v4", render_mode="human")
observation, info = env.reset(seed=42)

model =  PPO('MlpPolicy', env=env, verbose=1)






# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()
