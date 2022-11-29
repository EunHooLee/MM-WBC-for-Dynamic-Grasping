import argparse
import datetime
import gymnasium as gym
import numpy as np
# import itertools
import torch
from psac.sac import SAC
# from torch.utils.tensorboard import SummaryWriter

from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv
from wbc4dg.envs.mujoco import mujoco_utils
from psac.replay_memory import ReplayMemory
from wbc4dg.envs.mujoco.utils import distance

# Environment
env = MujocoMMDynamicGraspingEnv(reward_type="dense",)
_model_name = mujoco_utils.MujocoModelNames(env.model)

print("cuda" if torch.cuda.is_available() else "cpu")
# print(env._max_episode_steps)
state,_=env.reset()



torch.manual_seed(123456)
np.random.seed(123456)

# # Agent
agent = SAC(env.observation_space["observation"].shape[0], env.action_space, gamma=0.99, tau=0.005, alpha=0.2, policy="Gaussian", target_update_interval=True, automatic_entropy_tuning=False, hidden_size=256,lr=0.0005)
# agent.load_model('/home/yeoma/code/Gymnasium-Robotics-develop/models/sac_actor_robotics_middle_check','/home/yeoma/code/Gymnasium-Robotics-develop/models/sac_critic_robotics_middle_check','/home/yeoma/code/Gymnasium-Robotics-develop/models/sac_value_robotics_middle_check`')
# Memory
memory = ReplayMemory(1000000, 123456)

# Training Loop
max_reward = 0.0
max_reward_train = 0.0
total_numsteps = 0
print('===========started=============')
updates = 0
for i_episode in range(100000):
    episode_reward = 0
    episode_steps = 0
    truncated = False
    terminated = False
    state,_ = env.reset() 

    
    while not terminated:
        if truncated:
            break

        if 1000 > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state['observation'])  # Sample action from policy


        if len(memory) > 256:
            # Number of updates per step in environment
            for i in range(1):
                # Update parameters of all the networks
                value_loss, critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, 256, updates)
                updates += 1
            
        next_state, reward, terminated, truncated, info = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        # print("pass",total_numsteps)
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask  = 1 if episode_steps == 200 else float(not terminated)
        # print(mask)
        # print(state)
        if episode_steps==200:
            truncated = True
        elif distance(state['observation'][:3],state['observation'][5:8])>4:
            truncated =True
        else:
            truncated =False
        # print(next_state)
        # terminated = True if (next_state[1]>0.2 or next_state[1]<-0.2) else False

        memory.push(state['observation'], action, reward, next_state['observation'], mask) # Append transition to memory
        # print("checking: ",terminated,",", truncated)
        state = next_state

    if max_reward_train < episode_reward:
        agent.save_model("","")
        max_reward_train=episode_reward
    
    if i_episode%1000==999:
            agent.save_model("","middle")

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
    # if i_episode%1000==999: 
    #     avg_reward = 0.
    #     episodes = 10
    #     episode_reward =0
    #     terminated=False
    #     truncated = False
    #     state, _ =env.reset()
    #     while not terminated:
    #         action = agent.select_action(state['observation'], eval=True)
    #         next_state, reward, truncated, terminated, _ = env.step(action)
    #         # env.render()
    #         episode_reward += reward
    #         state = next_state
            
    #     avg_reward += episode_reward


    #     # writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    #     print("----------------------------------------")
    #     print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    #     print("----------------------------------------")
    #     if avg_reward >max_reward:
    #         max_reward = avg_reward
    #         agent.save_model("robotics","eval")

        

env.close()

# agent.load_checkpoint('/home/yeoma/code/pytorch-soft-actor-critic/checkpoints/sac_checkpoint_hopper-v4_checking',True)
# state,_=env.reset()
# terminated = False
# while not terminated:
#     action = agent.select_action(state, evaluate=True)
#     next_state, reward, truncated, terminated, _ = env.step(action)


# env.close()