import numpy as np
import math

def distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
    
# def zero_distance(goal_a):
#     return np.linalg.norm(goal_a, axis=-1)