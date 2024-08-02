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
    print(f"current location: {current_location}, current orientation: {current_orientation}")
    
    L = 2.9 # length of the vehicle
    
    # calculate the next position
    fx = current_location.x + L * np.cos(np.deg2rad(current_orientation))
    fy = current_location.y + L * np.sin(np.deg2rad(current_orientation))
    
    
    # search the nearest point
    index_nearest = 0
    
    # project RMS error onto the front axle vector
    front_axle_vector = [-np.cos(np.deg2rad(current_orientation)), -np.sin(np.deg2rad(current_orientation))]
    error_front_axle = np.dot([local_route[index_nearest][0], local_route[index_nearest][1]], front_axle_vector)
    
    # stanley control
    k = 0.5
    k_throttle = 0.3
    control.steer = np.arctan2(2.0 * L * error_front_axle, k * current_speed)/5 # steering angle
    control.throttle = max(k_throttle*(local_route[index_nearest][4] - current_speed), 0.4)# throttle
    
    print(f"current speed: {current_speed}, target speed: {local_route[index_nearest][4]}, steer: {control.steer}, throttle: {control.throttle}")

    return control