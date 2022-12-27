from typing import Union

import numpy as np

from wbc4dg.envs.mujoco.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
import random
import math

"""
object size: 0.04 x 0.04 x 0.04 -> xml에서 size는 half size를 나타낸다.
aheived goal: grip_pos (goal_a)
desired goal:  (goal_b)
distance_threshold: 0.02
"""
def distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def start_loc():
    x=random.uniform(-0.4,-0.7071*0.4)
    y=math.sqrt(0.16-math.pow(x,2))
    if random.random()>=0.5:
        y=-y
    return np.array([x]), np.array([y])

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

        def compute_reward(self, observation, goal, info, mu):
            # Compute distance between goal and the achieved goal.
           
            if self.reward_type == "sparse":
                d = distance(observation[3:6], goal)
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                # Compute distance between goal and the achieved goal.
                r_dist_xy = distance(observation[3:5], goal[:2])
                r_dist = abs(observation[3:6]- goal)
                r_vel = distance(observation[17:20], np.zeros(3))
                w_dist = 0.5
                w_vel = 0.2

                R_dense = -(w_dist*(r_dist_xy)) - (w_vel*(r_vel)) #+ np.exp(-100*pow(r_dist_xy,2))
                R_sparse = 0
                R_mani = 0
                R_line = 0 
                if info["is_success"]:
                    R_sparse = 10

                mu_next = self.compute_manipulability()
                
                if mu_next > mu:
                    # R_mani = mu_next * w_mani
                    R_mani = 0.3
                    self.mani_cnt +=1
                
                if observation[8]-self._utils.get_site_xpos(self.model, self.data, "plate0")[2]<-0.3:
                    fallen_penalty=-100
                else:
                    fallen_penalty=0

                cos_obj = self._utils.get_site_xmat(self.model, self.data, "object0:side")[0][0]
                sin_obj = self._utils.get_site_xmat(self.model, self.data, "object0:side")[1][0]
                cos_gri = self._utils.get_site_xmat(self.model, self.data, "robot0:grip")[0][0]
                sin_gri = self._utils.get_site_xmat(self.model, self.data, "robot0:grip")[1][0]
                
                R_line = cos_obj*cos_gri + sin_obj*sin_gri
                print("checking ",observation[3:6])
                return R_dense +R_mani +fallen_penalty

        # RobotEnv methods
        # ----------------------------
        def compute_manipulability(self):
            jacp = self._utils.get_site_jacp(self.model, self.data,2)
            jacr = self._utils.get_site_jacr(self.model, self.data,2)
            Jac = np.array([jacp[0][:9],
            jacp[1][:9],
            jacp[2][:9],
            jacr[0][:9],
            jacr[1][:9],
            jacr[2][:9]])
            
            eigenvalues, _ = np.linalg.eig(Jac @ np.transpose(Jac))
            
            eigenvalues_min = np.sqrt(min(eigenvalues))
            eigenvalues_max = np.sqrt(max(eigenvalues))
            
            
            if type(eigenvalues_min) != type(np.complex128(1.0+3.0j)):
                if math.isnan(eigenvalues_min):
                    eigenvalues_min = 0

            if type(eigenvalues_max) != type(np.complex128(1.0+3.0j)):
                if math.isnan(eigenvalues_max):
                    eigenvalues_max = 0.0000000001
            

            mu = eigenvalues_min/eigenvalues_max
            
            return mu
        def _set_action(self, action):
            assert action.shape == (10,) # action shape가 10이 아니면 에러 발생 (2 DoF (base) + 7 DoF (arm) + 1DoF (gripper))

            action = (
                action.copy()
            )  # ensure that we don't change the action outside of this scope
            
            base_ctrl, ee_ctrl, gripper_ctrl = action[:2] ,action[2:9], action[9:]
            # 아래 리미트 설정 한해주면 DOF ~~ Nan, inf 오류뜬다.
            base_ctrl *= 0.3
            ee_ctrl *= 0.05
            # print(self._utils.get_site_jacr(self.model, self.data,2))
            # gripper 대칭 제어
            gripper_ctrl = np.array([gripper_ctrl[0], gripper_ctrl[0]]) # gripper_ctrl[0] 원래 [0] 안해도 됬는데 왜그러지? 11/16
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper: #  gripper 잠김
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            
            action = np.concatenate([base_ctrl, ee_ctrl, gripper_ctrl])
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
            d = distance(observation[3:6], desired_goal)

            _d = observation[8]-self._utils.get_site_xpos(self.model, self.data, "plate0")[2]

            d_side_center=distance(self._utils.get_site_xpos(self.model, self.data, "object0:side"),self._utils.get_site_xpos(self.model, self.data, "robot0:grip"))
            d_side_center_c1=distance(self._utils.get_site_xpos(self.model, self.data, "object0:side_corner1"),self._utils.get_site_xpos(self.model, self.data, "robot0:grip"))
            d_side_center_c2=distance(self._utils.get_site_xpos(self.model, self.data, "object0:side_corner2"),self._utils.get_site_xpos(self.model, self.data, "robot0:grip"))
            d_side_center_c3=distance(self._utils.get_site_xpos(self.model, self.data, "object0:side_corner3"),self._utils.get_site_xpos(self.model, self.data, "robot0:grip"))
            d_side_center_c4=distance(self._utils.get_site_xpos(self.model, self.data, "object0:side_corner4"),self._utils.get_site_xpos(self.model, self.data, "robot0:grip"))

            if _d >= 0.42 and d <= 0.025:# and d_side_center<0.005 and d_side_center_c1<0.03 and d_side_center_c2<0.03 and d_side_center_c3<0.03 and d_side_center_c4<0.03:
                return True
            # return (d < self.distance_threshold).astype(np.float32)
            return False

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
        if self.has_object:
            # _object_plate_xpos = self._utils.get_site_xpos(self.model, self.data, "plate")
            
            object_plate_action = np.array([0.002, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            # object_plate_action = np.array([0.001, (np.sin(_object_plate_xpos[0] + 0.001 -1.5)-_object_plate_xpos[1]) ,0.0, 1.0, 0.0, 0.0, 0.0])
            self._utils.mocap_set_action(self.model, self.data, object_plate_action)


    def obj_is_fallen(self):
        if self._utils.get_site_xpos(self.model, self.data, "object0")[2]<=0.5:
            print(self._utils.get_site_xpos(self.model, self.data, "object0")[2])
            return True
        return False


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
            # print(object_pos)
        # else 실행 안됨
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)
     
        # 왜 state가 음수도 나오고 0.04 (JOINT MAX) 이상 값도 나오지?
        gripper_state = robot_qpos[-2:] # 배열의 뒤에서 부터 2개 (object joint는 qpos에서 안뜬다.)
        # print(object_pos)
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric
        
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


    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)


    def _reset_sim(self):
        self.mani_cnt = 0
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
    
        object_qpos = np.array([1.5, 0.0, 1.5, 1.0, 0.0, 0.0, 5.0])
        self._utils.set_joint_qpos(
            self.model, self.data, "object0:joint", object_qpos
        )
        # # print(self._utils.get_site_xpos(self.model, self.data, "plate"))
        
        self._utils.set_mocap_pos(self.model, self.data, "plate0:mocap", np.array([1.5 ,0.0, 1.0]))
        self._utils.set_mocap_quat(self.model, self.data, "plate0:mocap", np.array([1.0, 0.0, 0.0, 0.0]))
            

        
        # random deployement of the robot
        self.base_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:base_link").copy()
        # robot_x = self.base_pos[1] + self.np_random.uniform(-1.2,-0.8)
        # robot_y = self.base_pos[2] + self.np_random.uniform(-0.5, 0.5)
        robot_x, robot_y =start_loc()
        self._utils.set_joint_qpos(self.model, self.data, "robot0:base_joint1", robot_x)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:base_joint2", robot_y)

        self._mujoco.mj_forward(self.model, self.data)

        return True


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            if name == "robot0:base_joint1" or name == "robot0:base_joint2":
                value = value + self.np_random.uniform(-1.0,0.0)
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

        if self.has_object:
            self._utils.set_mocap_pos(self.model, self.data, "object0:plate", np.array([1.5 ,0.0, 0.68]))
            self._utils.set_mocap_quat(self.model, self.data, "object0:plate", np.array([1.0, 0.0, 0.0, 0.0]))



        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()


        # self.height_offset 이건 뭐지
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0"
            )[2]
