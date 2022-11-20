
import gymnasium as gym
import numpy as np
from algorithm.SAC_mine.sac_torch import Agent
from algorithm.SAC_mine.utils import plot_learning_curve
# from gym import wrappers
import torch
DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'

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
agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0],device=DEVICE)
print("========================   The main code  ==============================")
print("envobservation space:",env.observation_space)
print("env action space:",env.action_space.shape)
print("======================================================")
n_games = 10
# env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
filename = 'inverted_pendulum.png'
figure_file = 'plots/' + filename
best_score = env.reward_range[0]
score_history = []
load_checkpoint = False
if load_checkpoint:
    agent.load_models()
    env.render(mode='human')
for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.remember(observation, action, reward, observation_, done)
        if not load_checkpoint:
            agent.learn()
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()
    print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
if not load_checkpoint:
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)