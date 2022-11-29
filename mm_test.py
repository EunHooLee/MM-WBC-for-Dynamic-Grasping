import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from psac.sac import SAC
# from torch.utils.tensorboard import SummaryWriter

from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv
from wbc4dg.envs.mujoco import mujoco_utils
from psac.replay_memory import ReplayMemory

import math


# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make('InvertedPendulum-v4',render_mode='human')
# env = gym.make("FetchPickAndPlace-v2",max_episode_steps=500, render_mode="human")
# env = gym.make('CartPole-v1',render_mode='human')


env = MujocoMMDynamicGraspingEnv(reward_type="dense",)
_model_name = mujoco_utils.MujocoModelNames(env.model)
obs,_ = env.reset()

print("cuda" if torch.cuda.is_available() else "cpu")
# print(env._max_episode_steps)
state,_=env.reset()
print(env.observation_space["observation"])
# env.action_space.seed(123456)
# env = env.reset()
# print(env)
# torch.manual_seed(123456)
np.random.seed(123456)

# # Agent
agent = SAC(env.observation_space["observation"].shape[0], env.action_space, gamma=0.99, tau=0.005, alpha=0.2, policy="Gaussian", target_update_interval=True, automatic_entropy_tuning=False, hidden_size=256,lr=0.0005)
agent.load_model('/home/yeoma/code/mm-wbc_v2/models/sac_actor__','/home/yeoma/code/mm-wbc_v2/models/sac_critic__','/home/yeoma/code/mm-wbc_v2/models/sac_value__')
# Memory
memory = ReplayMemory(1000000, 123456)
# agent.load_model('/home/yeoma/code/Gymnasium-Robotics-develop/models/sac_actor_robotics_train','/home/yeoma/code/Gymnasium-Robotics-develop/models/sac_critic_robotics_train','/home/yeoma/code/Gymnasium-Robotics-develop/models/sac_value_robotics_train')
state,_=env.reset()
terminated = False
while not terminated:
    action = agent.select_action(state['observation'], eval=True)
    next_state, reward, truncated, terminated, _ = env.step(action)
#     env.render() 

env.close()