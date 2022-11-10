import gymnasium as gym

from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv


env = MujocoMMDynamicGraspingEnv(reward_type="sparse")
