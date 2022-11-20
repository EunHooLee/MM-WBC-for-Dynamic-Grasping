from typing import Union

import numpy as np

from wbc4dg.envs.mujoco.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations

# the goal distance
# def compute_reward(self, achieved_goal, goal, info):
# def _set_action(self, action):
# def _get_obs(self):
# def _sample_goal(self):
# def _is_success(self, achieved_goal, desired_goal):

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def get_base_fetch_env(RobotEnvClass: MujocoRobotEnv):
    """Factory function that returns a BaseMMEnv class that inherits
    from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings.
    """
    
    class BaseMMEnv(RobotEnvClass):
        """Superclass for all Fetch environments."""

        def __init__(
            self,
            gripper_extra_height,
            block_gripper,
            has_object: bool,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            reward_type,
            **kwargs
        ):
            """Initializes a new Fetch environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                gripper_extra_height (float): additional height above the table when positioning the gripper
                block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
                has_object (boolean): whether or not the environment has an object
                target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
                target_offset (float or array with 3 elements): offset of the target
                obj_range (float): range of a uniform distribution for sampling initial object positions
                target_range (float): range of a uniform distribution for sampling a target
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            """

            self.gripper_extra_height = gripper_extra_height
            self.block_gripper = block_gripper
            self.has_object = has_object
            self.target_in_the_air = target_in_the_air
            self.target_offset = target_offset
            self.obj_range = obj_range
            self.target_range = target_range
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type
            # self.a = 0 
            # self.b = 0
            # self.k = 0
            super().__init__(n_actions=10, **kwargs)         # n_action : 2 DoF (base) + 7 DoF (arm) + 1DoF (gripper)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            # Compute distance between goal and the achieved goal.
            d = goal_distance(achieved_goal, goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (10,) # action shape가 4가 아니면 에러 발생 (action은 3DoF EE pose, 나머지 1개는 gripper)

            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            '''get the reward thought how are we gonna get the '''
            base_ctrl, ee_ctrl, gripper_ctrl = action[:2] ,action[2:9], action[9:]
            # self.a +=0.001  # acc = 0.001
            # self.a = 0.05     # vel 
            # self.b = 0.05     # vel 
            
            # base_ctrl = base_ctrl+

            # if self.k < 1000:
            #     base_ctrl =[ self.a, 0.0]
            #     self.k +=1
            # else:
            #     base_ctrl = [self.a, self.b]
            


            ee_ctrl = [0.00, 0.26 , 0.11, -1.62, 0.06, 1.42, 0.84]
            # gripper 대칭 제어
            gripper_ctrl = np.array([gripper_ctrl[0], gripper_ctrl[0]]) # gripper_ctrl[0] 원래 [0] 안해도 됬는데 왜그러지? 11/16
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper: #  gripper 잠김
                gripper_ctrl = np.zeros_like(gripper_ctrl)

            action = np.concatenate([base_ctrl, ee_ctrl, gripper_ctrl])
            return action
        '''get the obesrvations and set tot he'''
        def _get_obs(self):
            (
                base_pos,
                grip_pos,
                object_pos,
                object_rel_pos,
                gripper_state,
                object_rot,
                object_velp,
                object_velr,
                base_velp,
                grip_velp,
                gripper_vel,
            ) = self.generate_mujoco_observations()

            if not self.has_object: # 날라다니는 물체가 있는지 없는지
                achieved_goal = grip_pos.copy()
            else:
                # achieved_goal = np.squeeze(object_pos.copy()) # array 형태로 바꾼다. 
                # achieved_goal = ?  이 부분 수정!!
                achieved_goal = grip_pos.copy()
            
            obs = np.concatenate(
                [
                    base_pos,
                    grip_pos,
                    object_pos.ravel(),
                    object_rel_pos.ravel(),
                    gripper_state,
                    object_rot.ravel(),
                    object_velp.ravel(),
                    object_velr.ravel(),
                    base_velp,
                    grip_velp,
                    gripper_vel,
                ]
            )


            return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
            }

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def get_gripper_xpos(self):

            raise NotImplementedError

        def _viewer_setup(self):
            lookat = self.get_gripper_xpos()
            for idx, value in enumerate(lookat):
                self.viewer.cam.lookat[idx] = value
            self.viewer.cam.distance = 2.5
            self.viewer.cam.azimuth = 132.0
            self.viewer.cam.elevation = -14.0

        
        """
        11.16 - leh
        _sample_goal(), _is_success() 는 우리가 제시하는 방법에 맞게 수정해야 하는 부분이다. 
        """
        def _sample_goal(self):
            if self.has_object:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
            return goal.copy()

        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)

    return BaseMMEnv


class MujocoMMEnv(get_base_fetch_env(MujocoRobotEnv)):
    # gripper 가 block일 때만 사용하고, 왜사용하는지는 모르겠다. 일단 우리는 필요없다. 
    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)
        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        # self._utils.mocap_set_action(self.model, self.data, action)    # We don't use mocap.

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        base_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:base_link")
        
        dt = self.n_substeps * self.model.opt.timestep # dt 뭐냐?  # 우리에게 1 step이 simulation에서는 n_substeps 이다. 
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )
        base_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:base_link")
        )
        
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )

        
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            # rotations
            object_rot = rotations.mat2euler(
                self._utils.get_site_xmat(self.model, self.data, "object0")
            )
            # velocities
            object_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp

        # else 실행 안됨
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        
        gripper_state = robot_qpos[-2:] # 배열의 뒤에서 부터 2개 (object joint는 qpos에서 안뜬다.)
        
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        # print(base_velp)

        # grip_pos, grip_vel == EE , gripper_vel 은 gripper 가 좌우로 움직이는 속도인 것 같다.
        return (
            base_pos,
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,    # gripper 와 상대속도
            object_velr,    # world 좌표계에 대한 속도
            base_velp,
            grip_velp,
            gripper_vel,
        )

    def get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["panda_hand"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        # 이 부분도 수정 필요한 듯
        # 
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True




    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)

            # print(name)
        
        # self._utils.reset_mocap_welds(self.model, self.data)
        
        
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        # 이 부분은 처음 gripper 의 초기 위치인데 없어도 될 듯
        # gripper_target = np.array(
        #     [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        # ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip") # 현재 gripper 위치 + gripper_extra_height + 이상한 값

        # gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])

        # base_target = np.array([0.0, 0.0, 0.0])
        # base_rotation = np.array([1.0, 0.0, 0.0, 0.0])

        
        # self._utils.set_mocap_pos(self.model, self.data, "robot0_base:mocap", base_target)
        # self._utils.set_mocap_quat(
        #     self.model, self.data, "robot0_base:mocap", base_rotation
        # )
    

        # self._utils.set_mocap_pos(self.model, self.data, "robot0_gripper:mocap", gripper_target)
        # self._utils.set_mocap_quat(
        #     self.model, self.data, "robot0_gripper:mocap", gripper_rotation
        # )

        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.

        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        
        # self.height_offset 이건 뭐지
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]
