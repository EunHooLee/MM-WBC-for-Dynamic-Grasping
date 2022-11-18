import gymnasium as gym
from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv
from wbc4dg.envs.mujoco import mujoco_utils
# from algorithm.stable_baselines3 import SAC
from algorithm.SAC_made.sac_torch import Agent
import torch
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


env = MujocoMMDynamicGraspingEnv(reward_type="sparse",)
_model_name = mujoco_utils.MujocoModelNames(env.model)

env.reset()
# print(env.observation_space['observation'])
print("======================================================")
agent = Agent(input_dims=env.observation_space['observation'].shape, env=env,
        n_actions=env.action_space.shape[0],device=DEVICE)


n_games = 250
# filename = 'inverted_pendulum.png'
# figure_file = 'plots/' + filename
best_score = env.reward_range[0]
score_history = []
load_checkpoint = False

if load_checkpoint:
    agent.load_models()
    env.render(mode='human')

for i in range(n_games):
    observation,_ = env.reset()
    terminated = False
    score = 0

    while not terminated or not truncated:
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        # print("obs:",observation_)
        score += reward
        agent.remember(observation, action, reward, observation_, terminated, truncated)
        
        if not load_checkpoint:
            agent.learn()
        observation= observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()
    print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
if not load_checkpoint:
    x = [i+1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, figure_file)