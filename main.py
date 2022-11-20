import gymnasium as gym
from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv
from wbc4dg.envs.mujoco import mujoco_utils
# from algorithm.stable_baselines3 import SAC
# from algorithm.SAC_made.sac_torch import Agent
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_XML_PATH = "dynamic_grasping.xml"



env = MujocoMMDynamicGraspingEnv(reward_type="sparse",)
_model_name = mujoco_utils.MujocoModelNames(env.model)
obs,_ = env.reset()
print(obs)
print(obs)


for _ in range(1000):

    action = env.action_space.sample()
    # print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    # the original is as the followin
    # state_, reward, done, info

env.close()