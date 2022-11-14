import gymnasium as gym

from wbc4dg.envs.mujoco import MujocoMMDynamicGraspingEnv
from gymnasium_robotics.utils import mujoco_utils


# MODEL_XML_PATH = "dynamic_grasping.xml"

env = MujocoMMDynamicGraspingEnv(reward_type="sparse",)
_model_name = mujoco_utils.MujocoModelNames(env.model)

# print(_model_name.joint_names)
obs, info = env.reset()

# action = env.action_space.sample()
# observation, reward, terminated, truncated, info = env.step(action)

for _ in range(1000):
    action = env.action_space.sample()*10
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation)
env.close()