import gymnasium as gym

from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv


MODEL_XML_PATH = "dynamic_grasping.xml"

env = MujocoMMDynamicGraspingEnv(reward_type="sparse",model_path=)
