import gymnasium as gym
from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv
from wbc4dg.envs.mujoco import mujoco_utils
from algorithm.stable.sac import SAC
# from algorithm.SAC_made.sac_torch import Agent
import torch
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


env = MujocoMMDynamicGraspingEnv(reward_type="sparse",)
_model_name = mujoco_utils.MujocoModelNames(env.model)

env.reset()
# print(env.observation_space['observation'])

model =  SAC('MlpPolicy', env=env,verbose=1)