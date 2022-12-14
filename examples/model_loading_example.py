#!/usr/bin/env python3

import mujoco_py
import math
import os
import numpy as np
import random


MODEL_PATH = "../wbc4dg/envs/mujoco/assets/dynamic_grasping.xml"
# MODEL_PATH = "../../.mujoco/mujoco-py/xmls/tosser.xml"

model = mujoco_py.load_model_from_path(MODEL_PATH)
sim = mujoco_py.MjSim(model)

grip_pos = sim.data.get_site_xpos("ee")
print(grip_pos)

# viewr = mujoco_py.MjViewer(sim=sim)

# state = sim.get_state()
# print("state 1: ", state)
# qpos = np.array(list(map(lambda x: x*random.random(),np.ones(shape=(25,)))))

# qvel = np.array(list(map(lambda x: x*random.random(),np.ones(shape=(25,)))))

# state = mujoco_py.MjSimState(state.time, qpos, qvel, state.act, state.udd_state)

# print("modified state : ", state)

# print(sim.mocap_pos)


# sim_state = sim.get_state()

# while True:
#     sim.set_state(sim_state)
    
#     for i in range(1000):
#         if i < 150:
#             sim.data.ctrl[:] = 0.0
#         else:
#             sim.data.ctrl[:] = -100.0
#         sim.step()
#         print(sim.data.ctrl[:],"     ", i)
#         # viewr.render()
#     if os.getenv('TESTING') is not None:
#         break
