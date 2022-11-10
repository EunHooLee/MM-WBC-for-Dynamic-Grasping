# Header Summary
# ----------------------------------------------------
# reach.py

import os
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch_env import MujocoFetchEnv, MujocoPyFetchEnv

class MujocoFetchReachEnv(MujocoFetchEnv, EzPickle):
    NotImplementedError


# ----------------------------------------------------
# fetch_env.py

from typing import Union
import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoPyRobotEnv, MujocoRobotEnv
from gymnasium_robotics.utils import rotations

class BaseFetchEnv("""RobotEnvClass"""):          # MujocoRobotEnv를 상속받아서 BaseFetchEnv 로 바뚠뒤 get_base_fetch_env 함수를 이용해 MujocoFetchEnv 로 보낸다.
    def compute_reward(self, achieved_goal, goal, info):
        NotImplementedError
    def _set_action(self, action):          # 무시안됨!! (Overrided at MujocoFetchEnv BUT!!!) MujocoFetchEnv에서 super()._set_action()을 이용해 여기에 있는 함수를 직접적으로 이용한다.
        NotImplementedError
    def _get_obs(self):                     # 최종 사용
        NotImplementedError
    def generate_mujoco_observations(self): # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError
    def get_gripper_xpos(self):             # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError
    def _viewer_setup(self):                # 최종 사용
        NotImplementedError
    def _sample_goal(self):                 # 최종 사용
        NotImplementedError
    def _is_success(self, achieved_goal, desired_goal):     # 최종 사용
        NotImplementedError

class MujocoFetchEnv("""get_base_fetch_env(MujocoRobotEnv)"""):
    def _step_callback(self):               # 최종 사용
        NotImplementedError
    def _set_action(self, action):          # 최종 사용 아님 (super()._set_action() 이용하고 있다.)
        NotImplementedError
    def generate_mujoco_observations(self): # 최종 사용
        NotImplementedError
    def get_gripper_xpos(self):             # 최종 사용
        NotImplementedError
    def _render_callback(self):             # 최종 사용
        NotImplementedError
    def _reset_sim(self):                   # 최종 사용
        NotImplementedError
    def _env_setup(self, initial_qpos):     # 최종 사용
        NotImplementedError

# ----------------------------------------------------------------------
# robot_env.py

import copy
import os
from typing import Optional, Union
import gymnasium as gym
import numpy as np
from gymnasium import error, logger, spaces
from gymnasium_robotics import GoalEnv
try:
    import mujoco_py

    from gymnasium_robotics.utils import mujoco_py_utils
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

try:
    import mujoco

    from gymnasium_robotics.utils import mujoco_utils
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

DEFAULT_SIZE = 480


class BaseRobotEnv(GoalEnv):
    def compute_terminated(self, achieved_goal, desired_goal, info):
        NotImplementedError
    def compute_truncated(self, achievec_goal, desired_goal, info):
        NotImplementedError
    def step(self, action):                 # 사용됨 (Override Env)
        NotImplementedError
    def reset(self,*,seed: Optional[int] = None,options: Optional[dict] = None,):   # 사용됨 (Override GoalEnv)
        NotImplementedError
    def close(self):
        NotImplementedError
    def _mujoco_step(self, action):         # 무시됨 (Overrided at MujocoRobotEnv)
        NotImplementedError
    def _reset_sim(self):                   # 무시됨 (Overrided at MujocoRobotEnv)
        NotImplementedError
    def _initialize_simulation(self):       # 무시됨 (Overrided at MujocoRobotEnv)
        NotImplementedError
    def _get_obs(self):                     # 무시됨 (Overrided at BaseFetchEnv)
        NotImplementedError
    def _set_action(self, action):          # 무시됨 (Overrided at BaseFetchEnv, MujocoFetchEnv)
        NotImplementedError
    def _is_success(self, achieved_goal, desired_goal):     # 무시됨 (Overrided at BaseFetchEnv)
        NotImplementedError
    def _sample_goal(self):                 # 무시됨 (Overrided at BaseFetchEnv)
        NotImplementedError
    def _env_setup(self, initial_qpos):     # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError
    def _viewer_setup(self):                # 무시됨 (Overrided at BaseFetchEnv)
        NotImplementedError
    def _render_callback(self):             # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError
    def _step_callback(self):               # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError

class MujocoRobotEnv(BaseRobotEnv):
    def _initialize_simulation(self):       # 최종 사용
        NotImplementedError
    def _reset_sim(self):                   # 무시됨 (Overriding at MujocoFetchEnv)
        NotImplementedError
    def render(self):
        NotImplementedError
    def _get_viewer(self):
        NotImplementedError
    @property
    def dt(self):
        NotImplementedError
    def _mujoco_step(self, action):
        NotImplementedError

# ---------------------------------------------------------------
# gymnasium_robotics/core.py
# abstract method 로 구현됨 : method의 이름만 선언되있고 구현은 안되있음.

from abc import abstractmethod
from typing import Optional
import gymnasium as gym
from gymnasium import error

class GoalEnv(gym.Env):
    def reset(self, seed: Optional[int] = None):        # 무시됨 (Overrided at BaseRobotEnv)
        NotImplementedError
    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        NotImplementedError
    @abstractmethod
    def compute_terminated(self, achieved_goal, desired_goal, info):
        NotImplementedError
    @abstractmethod
    def compute_truncated(self, achieved_goal, desired_goal, info):
        NotImplementedError
    
# ----------------------------------------------------------------
# gymnasium/core.py

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from gymnasium import spaces
from gymnasium.logger import warn
from gymnasium.utils import seeding

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

class Env(Generic[ObsType, ActType]):
    def step(self, action: ActType):# 무시됨 (Overrided at BaseRobotEnv)
        NotImplementedError
    def reset(self):                # 무시됨 (Overrided at GoalEnv, BaseRobotEnv)
        NotImplementedError
    def render(self):               # 무시됨 (Overrided at MujocoRobotEnv)
        NotImplementedError
    def close(self):                # 무시됨 (Overrided at BaseRobotEnv)
        NotImplementedError
    @property
    def unwrapped(self):
        NotImplementedError
    def np_random(self):
        NotImplementedError
    @np_random.setter
    def np_random(self, value: np.random.Generator):
        NotImplementedError
    def __str__(self):
        NotImplementedError
    def __enter__(self):
        NotImplementedError
    def __exit__(self, *args):
        NotImplementedError


# --------------------------------------------------------------------------------------
# class : __init__() summary


class MujocoMMDynamicGraspingEnv(MujocoMMEnv, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoMMEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

class MujocoMMEnv(get_base_fetch_env(MujocoRobotEnv)):
    NotImplementedError                                     # __init__() 없음


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
            super().__init__(n_actions=4, **kwargs)

class BaseRobotEnv(GoalEnv):
    """Superclass for all MuJoCo robotic environments."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "rgb_array_list",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        model_path: str,                        # 위에 있음
        initial_qpos,                           # 위에 있음
        n_actions: int,                         # 위에서 따로 지정 (n_action=4)
        n_substeps: int,
        render_mode: Optional[str] = "human",   # 위에 없음, 여기서 지정
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ):

class GoalEnv(gym.Env): # __init__() 없음


class Env(Generic[ObsType, ActType]):

    # Set this in SOME subclasses
    metadata: Dict[str, Any] = {"render_modes": []}
    # define render_mode if your environment supports rendering
    render_mode: Optional[str] = None
    reward_range = (-float("inf"), float("inf"))
    spec: "EnvSpec" = None

    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    # Created
    _np_random: Optional[np.random.Generator] = None