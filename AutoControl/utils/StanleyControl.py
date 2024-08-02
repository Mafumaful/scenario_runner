"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Mafumaful, Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""

import carla
import numpy as np

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    return angle_mod(angle)

def StanleyController(local_route, current_state, current_speed):
    """
    Stanley Controller
    
    Params
    ------
    local_route: np.array()
        the route calculated by the local planner from start to goal, which contains the waypoints and the target speed
        [[x, y, yaw, v], ...]
    current_state: carla.transform()
        current state of the vehicle, which contains the location and orientation of the vehicle
        
    Returns
    -------
    control: carla.VehicleControl()
        control signal for the vehicle
    """
    control = carla.VehicleControl()

    if len(local_route) == 0:
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        return control
    
    current_location = current_state.location
    current_orientation = current_state.rotation.yaw
    
    # Find the nearest waypoint
    min_dist = float('inf')
    index_l, index_r = 0, len(local_route)-1
    
    for i in range(len(local_route)):
        dist = np.sqrt((current_location.x - local_route[i][0])**2 + (current_location.y - local_route[i][1])**2)
        if dist < min_dist:
            min_dist = dist
            index_l = i
            
    # Find the nearest waypoint in the forward direction
    wp = local_route[index_l]
    wp_orientation = wp[2]
    wp_location = carla.Location(x=wp[0], y=wp[1], z=0.0)
    wp_orientation = carla.Rotation(yaw=wp_orientation)
    
    # Calculate the heading error
    dx = wp_location.x - current_location.x
    dy = wp_location.y - current_location.y
    heading = np.arctan2(dy, dx)
    heading_error = heading - np.radians(current_orientation)
    heading_error = angle_mod(heading_error)
    
    # Calculate the cross track error
    front_axle = 2.0
    rear_axle = 0.0
    cross_track_error = np.sin(heading) * (current_location.x - wp_location.x) - np.cos(heading) * (current_location.y - wp_location.y)
    
    # Calculate the steering angle
    k_steer = 0.3
    steer_abs = min(abs(k_steer * heading_error + np.arctan2(2.0 * front_axle * cross_track_error, 10.0 * current_speed)), 0.8)
    sign_steer = np.sign(heading_error)
    control.steer = sign_steer * steer_abs
    
    # Calculate the speed error
    k_speed = 0.5
    speed_error = wp[3] - current_speed
    control.throttle = min(k_speed * speed_error, 0.8)
    
    return control