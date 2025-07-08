import numpy as np
import matplotlib.pyplot as plt

from Quadrotor import Quadrotor
import Ref_Trajectory as rf
import Affine_LQR as LQR
import Cost_Function as cost
import Simulator as sim

Drone = Quadrotor()