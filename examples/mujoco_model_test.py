import gymnasium as gym
from gymnasium_robotics.utils import mujoco_utils
from gymnasium.envs.mujoco.mujoco_rendering import Viewer
import mujoco
from gymnasium_robotics.utils import rotations

import os
import numpy as np

MODEL_PATH = "../wbc4dg/envs/mujoco/assets/dynamic_grasping.xml"


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# viewer = Viewer(model=model, data=data)
_model_name = mujoco_utils.MujocoModelNames(model)
# print("Joint name: ",_model_name.joint_names)

# object_pose = mujoco_utils.get_joint_qpos(model,data,"object0")
# print(object_pose)
robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(model=model, data=data,joint_names=_model_name.joint_names)

# print("pos : ",robot_qpos)
# print(robot_qvel)
# grip_pos = mujoco_utils.get_site_xpos(model,data, "robot0:grip")

# print(grip_pos)

# action = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

# mujoco_utils.mocap_set_action(model, data, action)

def generate_mujoco_observations():
        # positions
    grip_pos = mujoco_utils.get_site_xpos(model, data, "robot0:grip")
    
    dt = 0.01 # dt 뭐냐? 
    grip_velp = (
        mujoco_utils.get_site_xvelp(model, data, "robot0:grip") * dt
    )   # self._utils.get_site_xvelp()는 뭐고 dt는 왜 곱하지?
    
    robot_qpos, robot_qvel = mujoco_utils.robot_get_obs(
        model, data, _model_name.joint_names
    )
    has_object = True
    if has_object:
        object_pos = mujoco_utils.get_site_xpos(model, data, "object0")
        # rotations
        object_rot = rotations.mat2euler(
            mujoco_utils.get_site_xmat(model,data, "object0")
        )
        # velocities
        object_velp = (
            mujoco_utils.get_site_xvelp(model, data, "object0") * dt
        )
        object_velr = (
            mujoco_utils.get_site_xvelr(model, data, "object0") * dt
        )
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
    else:
        object_pos = (
            object_rot
        ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:] # 배열의 뒤에서 부터 2개
    
    gripper_vel = (
        robot_qvel[-2:] * dt
    )  # change to a scalar if the gripper is made symmetric
    
    return (
        grip_pos,
        object_pos,
        object_rel_pos,
        gripper_state,
        object_rot,
        object_velp,
        object_velr,
        grip_velp,
        gripper_vel,
    )




while True:
    mujoco.mj_step(model, data)
    
    # action = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    # mujoco_utils.mocap_set_action(model=model, data=data, action=action)
    (grip_pos,
    object_pos,
    object_rel_pos,
    gripper_state,
    object_rot,
    object_velp,
    object_velr,
    grip_velp,
    gripper_vel,
    ) = generate_mujoco_observations()
    # print(object_pos)
    #print(object_velp)
    # print(mujoco_utils.get_site_xpos(model, data, "robot0:base_link"))
    # print(mujoco_utils.get_site_xpos(model, data, "target0"))
    print(object_rel_pos)
    # viewer.render()

