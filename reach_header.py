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
    def _set_action(self, action):          # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError
    def _get_obs(self):
        NotImplementedError
    def generate_mujoco_observations(self): # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError
    def get_gripper_xpos(self):             # 무시됨 (Overrided at MujocoFetchEnv)
        NotImplementedError
    def _viewer_setup(self):
        NotImplementedError
    def _sample_goal(self):
        NotImplementedError
    def _is_success(self, achieved_goal, desired_goal):
        NotImplementedError

class MujocoFetchEnv("""get_base_fetch_env(MujocoRobotEnv)"""):
    def _step_callback(self):
        NotImplementedError
    def _set_action(self, action):          # 최종 사용
        NotImplementedError
    def generate_mujoco_observations(self): # 최종 사용
        NotImplementedError
    def get_gripper_xpos(self):             # 최종 사용
        NotImplementedError
    def _render_callback(self):
        NotImplementedError
    def _reset_sim(self):
        NotImplementedError
    def _env_setup(self, initial_qpos):
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
    def _get_obs(self):
        NotImplementedError
    def _set_action(self, action):
        NotImplementedError
    def _is_success(self, achieved_goal, desired_goal):
        NotImplementedError
    def _sample_goal(self):
        NotImplementedError
    def _env_setup(self, initial_qpos):
        NotImplementedError
    def _viewer_setup(self):
        NotImplementedError
    def _render_callback(self):
        NotImplementedError
    def _step_callback(self):
        NotImplementedError

class MujocoRobotEnv(BaseRobotEnv):
    def _initialize_simulation(self):
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
