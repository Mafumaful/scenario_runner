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

def StanleyController(local_route, current_state, current_speed, k=1.0, Kp=0.8):
    """
    Stanley Controller
    
    Params
    ------
    local_route: np.array()
        the route calculated by the local planner from start to goal, which contains the waypoints and the target speed
        [[x, y, yaw, v], ...]
    current_state: carla.Transform
        current state of the vehicle, which contains the location and orientation of the vehicle
    current_speed: float
        current speed of the vehicle (m/s)
    k: float
        control gain for the cross-track error
    Kp: float
        proportional control gain for the speed

    Returns
    -------
    control: carla.VehicleControl
        control signal for the vehicle
    """
    control = carla.VehicleControl()
    
    # Extract current position and orientation
    x = current_state.location.x
    y = current_state.location.y
    yaw = np.deg2rad(current_state.rotation.yaw)
    
    # Find the closest waypoint on the local route
    distances = np.linalg.norm(local_route[:, :2] - np.array([x, y]), axis=1)
    closest_index = np.argmin(distances)
    closest_point = local_route[closest_index]
    path_yaw = closest_point[2]
    
    # Calculate cross-track error
    dx = closest_point[0] - x
    dy = closest_point[1] - y
    cross_track_error = np.sin(path_yaw - np.arctan2(dy, dx)) * np.sqrt(dx**2 + dy**2)
    
    # Calculate heading error
    heading_error = path_yaw - yaw
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize angle
    
    # Stanley control law for steering
    steering_angle = heading_error + np.arctan2(k * cross_track_error, current_speed)
    steering_angle = np.clip(steering_angle, -1.0, 1.0)  # Assuming the steering angle range is [-1, 1]
    
    # Calculate throttle (simple proportional controller)
    # target_speed = closest_point[3]
    target_speed = 2.0  # Constant speed for now
    # print(f"Target Speed: {target_speed}, Current Speed: {current_speed}")
    throttle = Kp * (target_speed - current_speed)
    throttle = np.clip(throttle, 0.0, 1.0)
    
    # Set control signals
    control.steer = float(steering_angle)
    control.steer = 0.0
    control.throttle = float(throttle)
    control.throttle = 0.4
    control.brake = 0.0  
    # print(f"Steering: {control.steer}, Throttle: {control.throttle}")
    
    return control