import numpy as np
from gymnasium import error, logger, spaces
import torch as T

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminated_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.truncated_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminated, truncated):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state["observation"]
        self.new_state_memory[index] = state_["observation"]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminated_memory[index] = terminated
        self.truncated_memory[index] = truncated
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminateds = self.terminated_memory[batch]
        truncateds = self.truncated_memory[batch]


        return states, actions, rewards, states_, terminateds, truncateds


