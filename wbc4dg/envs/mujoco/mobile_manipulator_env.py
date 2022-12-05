from typing import Union

import numpy as np
import random
from wbc4dg.envs.mujoco.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations

"""
object size: 0.04 x 0.04 x 0.04 -> xml에서 size는 half size를 나타낸다.
aheived goal: grip_pos (goal_a)
desired goal:  (goal_b)
distance_threshold: 0.02
"""
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

            super().__init__(n_actions=10, **kwargs)         # n_action : 2 DoF (base) + 7 DoF (arm) + 1DoF (gripper)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, observation, goal, info):
            # Compute distance between goal and the achieved goal.
           
            if self.reward_type == "sparse":
                d = goal_distance(observation[3:6], goal)
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                # Compute distance between goal and the achieved goal.
                r_dist = goal_distance(observation[3:6], goal)
                r_vel = goal_distance(observation[17:20], np.zeros(3))
                w_dist = 3
                w_vel = 2

                R_dense = -(w_dist*r_dist) - (w_vel*r_vel) + np.exp(-100*pow(r_dist,2))
                R_sparse = 0

                if info["is_success"]:
                    R_sparse = 200
                elif (r_vel<=0.2 and r_dist>0.5):
                    R_sparse = -5.0
                elif (r_vel>0.2 and r_dist<=0.5):
                    R_sparse = -10.0
                elif (r_vel<=0.2 and r_dist<=0.5):
                    R_sparse = 10.0

                return R_dense + R_sparse

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action,passed_substeps):
            assert action.shape == (10,) # action shape가 10이 아니면 에러 발생 (2 DoF (base) + 7 DoF (arm) + 1DoF (gripper))

            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            
            base_ctrl, ee_ctrl, gripper_ctrl = action[:2] ,action[2:9], action[9:]
            base_ctrl*=0.5
            ee_ctrl*=0.05

            # if passed_substeps <4:
            obj_ctrl=np.array([0.0, 0.0, 0.0, 0.0])
            
            # else:
                # obj_ctrl=np.array([0.0, 0.0, 0.0, 0.0])

            # gripper 대칭 제어
            gripper_ctrl = np.array([gripper_ctrl[0], gripper_ctrl[0]]) # gripper_ctrl[0] 원래 [0] 안해도 됬는데 왜그러지? 11/16
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper: #  gripper 잠김
                gripper_ctrl = np.zeros_like(gripper_ctrl)

            action = np.concatenate([base_ctrl, ee_ctrl, gripper_ctrl,obj_ctrl])
            return action

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

            if self.has_object:
                achieved_goal = grip_pos.copy()
            else:
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
                goal = self._utils.get_site_xpos(self.model, self.data, "object0")
            return goal.copy()

        def _is_success(self, observation, desired_goal):
            d = goal_distance(observation[3:6], desired_goal)
            v = goal_distance(observation[17:20],np.array([0,0,0]))
            d_gripper = abs(observation[12]-observation[13])

            if (d<0.02 and v<0.05 and d_gripper<0.045 and d_gripper>0.38):
                return True
            # return (d < self.distance_threshold).astype(np.float32)
            return False

    return BaseMMEnv


class MujocoMMEnv(get_base_fetch_env(MujocoRobotEnv)):


    def _set_action(self, action,passed_substeps):
        action = super()._set_action(action,passed_substeps)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        


    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        base_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:base_link")
        """
        dt: dx = x1 - x0 라고 하면 dx를 action으로 주게 되면 물체가 dx 만큼 순간이동하게 된다.
            이러한 물체의 움직임을 부드럽게 하기 위해 dt를 n_substeps로 나눠서 dt'마다 dx/n_substeps 씩 움직인다. 
        """
        
        dt = self.n_substeps * self.model.opt.timestep 

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
                self._utils.get_site_xvelp(self.model, self.data, "object0") * dt           # dt=0.04
            )
            object_velr = (
                self._utils.get_site_xvelr(self.model, self.data, "object0") * dt
            )
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
            # print("-----------------------")
            # print(object_pos,"   ",self._utils.get_mocap_xpos())
        # else 실행 안됨
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
        
        # 왜 state가 음수도 나오고 0.04 (JOINT MAX) 이상 값도 나오지?
        gripper_state = robot_qpos[-2:] # 배열의 뒤에서 부터 2개 (object joint는 qpos에서 안뜬다.)
        # print(gripper_state)
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric
        # print("site",self._utils.get_site_xpos(self.model, self.data, "object0:plate"))
        # print(object_pos)
        return (
            base_pos,
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,    # gripper 와 상대속도
            object_velr,    # world 좌표계에 대한 속도
            base_velp,      # world 좌표계에 대한 속도 인 것 같다.
            grip_velp,
            gripper_vel,
        )


    def get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["panda_hand"]
        return self.data.xpos[body_id]

    
    def get_body_xpos(self):
        body_id = self._model_names.body_name2id["robot0:base_link"]
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

        # random deployement of the robot
        self.base_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:base_link").copy()
        # robot_x = self.base_pos[1] + random.uniform(-1.0,0.0)
        # robot_y = self.base_pos[2] + random.uniform(-1.0,1.0)

        robot_x = self.base_pos[1] + np.array([-2.0])
        robot_y = self.base_pos[2] + np.array([-2.0])

        self._utils.set_joint_qpos(self.model, self.data, "robot0:base_joint1", robot_x)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:base_joint2", robot_y)
        self._mujoco.mj_forward(self.model, self.data)
        
        return True


    def _env_setup(self, initial_qpos):

        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        # 이거 뭔지 확인
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)


        self._mujoco.mj_forward(self.model, self.data)
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()

        obj_rot=np.array([1.0, 0.0, 0.0, 0.0])
        # self.initial_obj_xpos = np.concatenate([self.initial_obj_xpos, obj_rot])
        # self.height_offs et 이건 뭐지
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]