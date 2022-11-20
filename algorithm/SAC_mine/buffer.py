import numpy as np
from collections import deque
import random

class ReplayBuffer():
    def __init__(self, max_size, buffer_size):
        self.buffer_size = buffer_size
        self.mem_size = max_size
        self.buffer = deque()
        self.count=0

    def store_transition(self, state, action, reward, next_state, terminated, truncated, done):
    
        transition = (state, action, reward, next_state,terminated, truncated, done)

        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count +=1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample_buffer(self):
        max_mem = min(self.count, self.mem_size)
        batch = random.sample(self.buffer, max_mem)


        states = np.asarray([i[0] for i in batch],dtype=float)
        actions = np.asarray([i[1] for i in batch],dtype=float)
        rewards = np.asarray([i[2] for i in batch],dtype=float)
        next_states = np.asarray([i[3] for i in batch],dtype=float)
        terminateds = np.asarray([i[4] for i in batch],dtype=bool)
        truncateds = np.asarray([i[5] for i in batch],dtype=bool)
        dones = np.asarray([i[6] for i in batch],dtype=bool)

        return states, actions, rewards, next_states, terminateds, truncateds, dones

    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0


# if __name__=="__main__":
