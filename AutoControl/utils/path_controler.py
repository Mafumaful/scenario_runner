'''
calculate the control signal based on the current location and the target location

@input : target trajectory, it's a list of [[x, y, yaw], ...]
@input2: target velocity, float
@input3: current location, it's a point described by [x, y, yaw]
@input4: current velocity, float
@output: carla.VehicleControl()
'''

import math
from carla import VehicleControl

