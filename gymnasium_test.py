import gymnasium as gym
from algorithm.stable import SAC

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

model = SAC("MlpPolicy",env=env)

for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
#    for the gymnasium
# observation, reward, terminated, truncated=false, {} is the result of the step
   observation, reward, terminated, truncated, info = env.step(action)
#  observation, reward, done, info

   if terminated or truncated:
      observation, info = env.reset()
env.close()
