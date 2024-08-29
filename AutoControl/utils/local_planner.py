import numpy as np
import math
import sys, os
import carla
import bisect

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

MAX_SPEED = 150.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 12.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 12.0  # maximum curvature [1/m]

K_J = 0.1  # weight of jerk
K_T = 0.1  # weight of time
K_D = 1.0  # weight of square of d

K_LAT = 1.0  # weight of lateral direction
K_LON = 1.0  # weight of longitudinal direction

def search_index(x, x_list):
    """
    search data segment index

    Params
    ------
    x: float
        the value to search
    x_list: list
        the list of values, accending order

    Returns
    -------
    i: int
        the index closest to the value
    """
    
    # Binary search initialization
    left, right = 0, len(x_list) - 1

    # Binary search loop
    while left < right:
        mid = (left + right) // 2
        if x_list[mid] < x:
            left = mid + 1
        else:
            right = mid

    # Determine the closest index
    if left == 0:
        return left
    if left == len(x_list):
        return left - 1

    if abs(x - x_list[left - 1]) <= abs(x - x_list[left]):
        return left - 1
    else:
        return left

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
        self.v = []
        
def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    """
    fplist_init = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    
    Params
    ------
    
    c_speed: float
        current speed [m/s]
    c_accel: float
        current acceleration [m/s^2]
    c_d: float
        current lateral position [m]
    c_d_d: float
        current lateral speed [m/s]
    c_d_dd: float
        current lateral acceleration [m/s^2]
    s0: float
        current longitudinal position [m]
        
    Returns
    -------
    fplist_init: list
        a list of frenet paths
    """
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
                
                tfp.v = [np.hypot(tfp.s_d[i], tfp.d_d[i]) for i in range(len(tfp.s_d))]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, csp, index):
    print_flag = True
    for fp in fplist:
        index_temp = index
        # calc global positions
        for i in range(len(fp.s)):
            index_temp = index_temp+i*1 if index_temp+i*1 < len(csp["s"]) else len(csp["s"])-1
            ix, iy = csp["x"][index_temp], csp["y"][index_temp]
            
            if ix is None:
                break
            iyaw = csp["yaw"][index] # csp is the cubic spline path
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)
            
            # if print_flag:
            #     print("iy: ", iy , "\r\ndy: ", di * math.sin(iyaw + math.pi / 2.0))
            #     print_flag = False
            
        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))
            
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])
        
        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
            
    return fplist

def check_collision(fp, ob):
    for x,y in zip(fp.x, fp.y):
        for ox, oy, _ in ob:
            dx = x - ox
            dy = y - oy
            d = dx**2 + dy**2
            if d <= ROBOT_RADIUS**2:
                return True
    return False

def check_paths(fplist, obs):
    ok_index = []
    for i, _ in enumerate(fplist):
        obs_check_pass = True
        if obs is not None:
            for ob in obs:
                if check_collision(fplist[i], ob):
                    obs_check_pass = False
                    continue
                
        if not obs_check_pass:
            continue
        elif any([v > MAX_SPEED for v in fplist[i].s_d]):
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):
            continue
        
        ok_index.append(i)
    
    return [fplist[i] for i in ok_index]

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

def frenet_optimal_planning(csp, index, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    '''
    temp_candidate_routes, temp_choosed_route = frenet_optimal_planning(
        self.csp, 
        self.current_s, 
        self.current_speed, 
        self.current_accel, 
        self.current_d, 
        self.current_d_d, 
        self.current_d_dd, 
        obs_predicted_path
        )
    
    Params
    ------
    csp: dictionary
        the target path (spline)
        ["x"]-> list(float) : x position of the path
        ["y"]-> list(float) : y position of the path
        ["yaw"]-> list(float) : yaw of the path
        ["s"]-> list(float) : s position of the path
    s0: float
        the initial position
    c_speed: float
        the current speed
    c_accel: float
        the current acceleration
        
    Returns
    -------
    candidate_routes:list of np.array()
        a list of candidate routes
        [[x, y, z, yaw, v] ...] at the size of (N, 4) -> it's a list of np.array()
    choosed_route:
        [[x, y, z, yaw, v] ...] at the size of (N, 4)
    '''
    s0 = csp["s"][index]
    fplist_init = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    x,y = csp["x"][index], csp["y"][index]
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(f"1 optimal planning spline location:{x:.2f}, {y:.2f}")
    fplist = calc_global_paths(fplist_init, csp, index)
    x,y = fplist[0].x[0], fplist[0].y[0]
    # print(f"2 optimal planning global location:{x:.2f}, {y:.2f}")
    fplist = check_paths(fplist, ob)
    x,y = fplist[0].x[0], fplist[0].y[0]
    # print(f"3 optimal planning global location:{x:.2f}, {y:.2f}")
    
    if len(fplist) == 0:
        fplist = fplist_init # if no path is found, use the initial path
    
    # choose the best path
    min_cost = float('inf')
    best_path_index = 0
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path_index = fplist.index(fp)
    
    # assert it's in the index if not print the info of the path and the index
    # choosed_route = fplist[best_path_index]    
    choosed_route = copy.deepcopy(fplist[best_path_index])
    del fplist[best_path_index]
    
    return fplist, choosed_route

class FrenetOptimalPlanner(object):
    def __init__(self, ego_vehicle: carla.Actor, csp_target) -> None:        
        self.csp = csp_target # the target path (spline)
        
        velocity = np.array([ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y])
        location = np.array([ego_vehicle.get_location().x, ego_vehicle.get_location().y])
        
        # calculate the current speed and acceleration
        self.current_speed = np.hypot(velocity[0], velocity[1])
        self.current_accel = 0.0
        
        print('location:', location)
        best_index = 0
        print("spline location:", self.csp["x"][best_index], self.csp["y"][best_index])
        print("lenth of s:", len(self.csp["s"]))
        self.current_d= self.calc_distance(location, csp_target, best_index) # current lateral position [m]
        self.current_d_d = 0.0 # current lateral speed [m/s] 
        self.current_d_dd = 0.0 # current lateral acceleration [m/s^2]
        self.current_s = self.csp["s"][best_index] # current longitudinal position [m]
        
        self.debug_prev_index = 0
        
    def update(self, ego_vehicle: carla.Actor, obs_predicted_path: np.array, rk) -> None:
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
        
        # update the current state of the vehicle    
        velocity = np.array([ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y])
        location = np.array([ego_vehicle.get_location().x, ego_vehicle.get_location().y])
        
        # calculate the current speed and acceleration
        self.current_speed = np.hypot(velocity[0], velocity[1])
        self.current_accel = 0.0
        
        best_index = self.calc_best_index(location, self.csp)
        x = self.csp["x"][best_index]
        y = self.csp["y"][best_index]
        # print("best index:", best_index)
        self.debug_prev_index = best_index
        # print("prev index:", self.debug_prev_index)
        # print(f"\r\n>>>>>>>>>>>>")
        # print(f"spline location:{x:.2f}, {y:.2f}")
        # print(f"location:{location[0]:.2f}, {location[1]:.2f}")
        self.current_d = self.calc_distance(location, self.csp, best_index) # current lateral position [m]
        self.current_d_d = 0.0 # current lateral speed [m/s]
        self.current_d_dd = 0.0 # current lateral acceleration [m/s^2]
        
        # update the candidate routes and the choosed route
        temp_candidate_routes, temp_choosed_route = frenet_optimal_planning(self.csp, best_index, self.current_speed, self.current_accel, self.current_d, self.current_d_d, self.current_d_dd, obs_predicted_path)
        
        return self.convert_format(temp_candidate_routes, temp_choosed_route)
    
    @staticmethod
    def calc_best_index(location, csp):
        '''
        Params
        ------
        location: np.array
            The location of the vehicle (x, y).
        csp: dictionary
            ["x"]-> list(float) : x position of the path
            ["y"]-> list(float) : y position of the path
            ["yaw"]-> list(float) : yaw of the path
            ["s"]-> list(float) : s position of the path

        Returns
        -------
        best_index: int
            The best index of the path closest to the vehicle location.
        '''

        path_x = csp["x"]
        path_y = csp["y"]

        # Helper function to calculate squared distance (avoid sqrt for efficiency)
        def squared_distance(i):
            return (path_x[i] - location[0])**2 + (path_y[i] - location[1])**2

        # Step 1: Binary search on a custom distance metric
        low, high = 0, len(path_x) - 1
        while low < high:
            mid = (low + high) // 2
            if squared_distance(mid) < squared_distance(mid + 1):
                high = mid
            else:
                low = mid + 1

        # Step 2: Fine-tune by checking surrounding points
        best_index = low

        return best_index
    
    @staticmethod
    def calc_distance(location, csp, index):
        '''
        Params
        ------
        location: np.array()
            the location of the vehicle
        csp: dictioanry
            ["x"]-> list(float) : x position of the path
            ["y"]-> list(float) : y position of the path
            ["yaw"]-> list(float) : yaw of the path
            ["s"]-> list(float) : s position of the path
            
        Returns
        -------
        distance: float
            the distance between the vehicle and the target path
        '''
        x, y = csp["x"][index], csp["y"][index]
        distance = np.hypot(location[0] - x, location[1] - y)
        
        return distance
        
    
    @staticmethod
    def convert_format(candidates, target):
        """
        Params
        ------
        candidates: list(FrenetPath)
            a list of candidate paths
        
        target: FrenetPath
            the target path
        
        Returns
        -------
        candidate_routes: list(np.array())
            a list of candidate routes
            [[x, y, z, yaw, v] ...] at the size of (N, 4) -> it's a list of np.array()
            
        choosed_route: np.array()
        """
        
        candidate_routes = []
        choosed_route = np.zeros((len(target.x), 5))
        
        for candidate in candidates:
            temp_candidate = np.zeros((len(candidate.x), 5))
            temp_candidate[:, 0] = np.array(candidate.x)
            temp_candidate[:, 1] = np.array(candidate.y)
            temp_candidate[:, 3] = np.array(candidate.yaw)
            temp_candidate[:, 4] = np.array(candidate.v)
            
            candidate_routes.append(temp_candidate)
            
        choosed_route[:, 0] = np.array(target.x)
        choosed_route[:, 1] = np.array(target.y)
        choosed_route[:, 3] = np.array(target.yaw)
        choosed_route[:, 4] = np.array(target.v)
        
        return candidate_routes, choosed_route