import numpy as np
import sys, os
import carla

# add the $SCEANRIO_RUNNER_ROOT to the python path
SCENARI_RUNNER_ROOT = os.environ.get('SCENARIO_RUNNER_ROOT', None)
if SCENARI_RUNNER_ROOT is not None:
    sys.path.append(SCENARI_RUNNER_ROOT)

from AutoControl.utils.quintic_polynomials_planner import QuinticPolynomial
from AutoControl.utils.cubic_spline_planner import CubicSpline2D
import copy

MAX_ROAD_WIDTH = 5.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [s]
MIN_T = 4.0  # min prediction time [s]
TARGET_SPEED = 10.0 / 3.6  # target speed [m/s]
D_T_S = 1.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

K_J = 0.1  # weight of jerk
K_T = 0.1  # weight of time
K_D = 1.0  # weight of square of d

K_LAT = 1.0  # weight of lateral direction
K_LON = 1.0  # weight of longitudinal direction

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        
def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

def simple_planner(ego_vehicle, global_path, obs_predicted_path):
    '''
    Params
    ------
    ego_vehicle: carla.Actor()
        the ego vehicle
    
    global_path: cubic spline
        the path of the vehicle, which contains the position, orientation, velocity, etc.
        
    Returns
    -------
    candidate_routes: list
        a list of candidate routes
        [[x, y, z, yaw, v] ...] at the size of (N, 4) -> it's a list of np.array()
    choosed_route: list
        the choosed route
        [[x, y, z, yaw, v] ...] at the size of (N, 4)
    '''
    candidate_routes = []
    choosed_route = None
    
    tranform = ego_vehicle.get_transform()
    velocity = ego_vehicle.get_velocity()
    
    # get the tangent of the vector of velocity
    yaw = np.arctan2(velocity.y, velocity.x)
    yaws_candidates = [yaw+i*np.pi/10 for i in range(-3, 4)]
    vec_velocity_candidates = [np.array([np.cos(yaw), np.sin(yaw)]) for yaw in yaws_candidates]
    
    dt = 0.2
    velocity_scale = 60.0
    # predict
    for vec_velocity in vec_velocity_candidates:
        candidate = np.zeros((100, 5))
        x = tranform.location.x
        y = tranform.location.y
        z = tranform.location.z
        for i in range(100):
            x += vec_velocity[0] * velocity_scale * dt
            y += vec_velocity[1] * velocity_scale * dt
            z = tranform.location.z 
            yaw = np.arctan2(vec_velocity[1], vec_velocity[0])
            v = np.sqrt(vec_velocity[0]**2 + vec_velocity[1]**2)
            candidate[i] = [x, y, z, yaw, v]
        candidate_routes.append(candidate)
    
    choosed_index = 3
    # choose the best route
    choosed_route = candidate_routes[choosed_index]
    # delete from the candidate list
    del candidate_routes[choosed_index]
    
    return candidate_routes, choosed_route

def frenet_planner(ego_vehicle, global_path, obs_predicted_path):
    '''
    Params
    ------
    ego_vehicle: carla.Actor()
        the ego vehicle
    
    global_path: cubic spline
        the path of the vehicle, which contains the position, orientation, velocity, etc.
        
    Returns
    -------
    candidate_routes: list
        a list of candidate routes
        [[x, y, z, yaw, v] ...] at the size of (N, 4) -> it's a list of np.array()
    choosed_route: list
        the choosed route
        [[x, y, z, yaw, v] ...] at the size of (N, 4)
    '''
    candidate_routes = []
    choosed_route = None
    
    return candidate_routes, choosed_route

class FrenetOptimalPlanner(object):
    def __init__(self, csp_target: CubicSpline2D) -> None:
        # scalar states
        self.current_speed = 0.0
        self.current_accel = 0.0
        
        # Frenet states
        self.current_d = 0.0 # current lateral position [m]
        self.current_d_d = 0.0 # current lateral speed [m/s]
        self.current_d_dd = 0.0 # current lateral acceleration [m/s^2]
        self.current_s = 0.0 # current longitudinal position [m]
        
        self.csp = csp_target # the target path (spline)
        
    def update(self, ego_vehicle: carla.Actor):
        candidate_routes = []
        choosed_route = None
        
        return candidate_routes, choosed_route