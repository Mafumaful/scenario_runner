#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
from carla import ColorConverter as cc

import argparse
import os
import sys
import time
import collections
import datetime
import logging
import math
import weakref
import numpy as np
import pygame
import bisect

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# for parse the global route
from agents.navigation.global_route_planner import GlobalRoutePlanner
import xml.etree.ElementTree as ET

# add the $SCEANRIO_RUNNER_ROOT to the python path
SCENARI_RUNNER_ROOT = os.environ.get('SCENARIO_RUNNER_ROOT', None)
if SCENARI_RUNNER_ROOT is not None:
    sys.path.append(SCENARI_RUNNER_ROOT)
    
# frenet, spline dependencies
# from AutoControl.utils.frenet import Frenet
from AutoControl.utils.StanleyControl import StanleyController
from AutoControl.utils.local_planner import simple_planner
from AutoControl.utils.local_planner import FrenetOptimalPlanner

# turn the route into the spline
from AutoControl.utils.cubic_spline_planner import calc_spline_course

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height) -> None:
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        
    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        self._info_text += [
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
                
    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    def __init__(self, parent_actor, hud) -> None:
        self.sensor = None
        self.history = []
        self.hud = hud
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        
    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    
    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
            
# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud) -> None:
        self.sensor = None
        
        # If the spawn object is not a vehicle, we cannot use the LaneInvasionSensor
        if parent_actor.type_id.startswith('vehicle'):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
            
    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self.hud.notification('Crossed line %s' % ' and '.join(text))
        
# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
        
# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            bp.set_attribute('gamma', '2.2')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, args) -> None:
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        
        # this is set for the visualization of the route
        self.target_route = None
        self.predicted_trajectories = None
        
        # this is set for the ego vehicle, which is a dictionary.
        # "candidate routes" is the candidate route
        # "planner route" is the route that the planner will follow
        # in order to distinguish the two routes, we will use the different color to represent them
        self.planner_route = None
    
    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        
        # Keep same camera config if the camera manager exists
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        # Get the ego vehicle
        while self.player is None:
            print("Waiting for ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    self.player = vehicle
                    break
        
        self.player_name = self.player.type_id
        
        # Set up sensors
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        
        self.world.wait_for_tick()
        
    def tick(self, clock, wait_for_repetitions):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            if not wait_for_repetitions:
                return False
            else:
                self.player = None
                self.destroy()
                self.restart()

        self.hud.tick(self, clock)
        return True
    
    def render(self, display):
        self.camera_manager.render(display)
        if self.target_route is not None:
            self.render_route(display)
        
        # render the predicted trajectories
        if self.predicted_trajectories is not None:
            for trajectory in self.predicted_trajectories:
                self.render_trajectory(display, trajectory)
            self.predicted_trajectories = None
        
        # render the planner route
        if self.planner_route is not None:
            for route in self.planner_route:
                #  if the dictionary contains the key "candidate routes", then render the candidate route
                candidate_routes = self.planner_route["candidate routes"]
                if candidate_routes is not None:
                    for candidate_route in candidate_routes:
                        self.render_trajectory(display, candidate_route, (0, 255, 0))
                
                choosed_route = self.planner_route["planner route"]
                if choosed_route is not None:
                    self.render_trajectory(display, choosed_route, (255, 0, 0))
                    
            self.planner_route = None
        
        self.hud.render(display)
        
    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None
    
    '''
    if the input of the function is a point array, then it will return the point in the camera view
    @input points: the points in the world coordinate[[x,y,z],...]
    '''
    @staticmethod
    def calc_point_fromW2F(self, points):
        # build projection matrix of the camera
        width =  int(self.camera_manager.sensor.attributes['image_size_x'])
        height = int(self.camera_manager.sensor.attributes['image_size_y'])
        fov = float(self.camera_manager.sensor.attributes['fov'])
        
        focal = width/(2*np.tan(fov*np.pi/360.0))
        
        K = np.identity(3)
        K[0,0] = K[1,1] = focal
        K[0,2] = width/2.0
        K[1,2] = height/2.0
        
        # calculate the world to camera matrix
        w2c = np.array(self.camera_manager.sensor.get_transform().get_inverse_matrix())
        
        # keep the first 3 columns
        points = points[:,:3]
        ones_column = np.ones((points.shape[0],1)) # x, y, z, 1
        points = np.hstack((points, ones_column)).T
        
        # build projection matrix of the camera
        points_in_camera = np.dot(w2c, points)

        # convert UE4 coordinate to standard coordinate
        points_in_camera = np.array([points_in_camera[1], -points_in_camera[2], points_in_camera[0]])
        
        # convert to pixel coordinate
        point_img = np.dot(K, points_in_camera)
        
        # normalize the pixel coordinate, divide by the third column
        point_img /= point_img[2,:]
        
        # delete the points that are not in the camera view
        # Define conditions for valid points
        valid = (point_img[0, :] >= 0) & (point_img[0, :] < width) & \
                (point_img[1, :] >= 0) & (point_img[1, :] < height) & \
                (points_in_camera[2, :] >= 0)

        # Keep only valid columns
        point_img = point_img[:, valid]
        
        target_route_incamera = point_img[:2]
        
        return target_route_incamera
        
    # calculate target route in the camera view
    def calc_tr_in_cam(self):
        if self.target_route is None:
            print("No target route is set")
            return
        
        points = self.target_route
        points = points[:,:3] # [[x, y, z],...] 
        target = self.calc_point_fromW2F(self, points)
        
        return target
    
    def render_route(self, display):
        route = self.calc_tr_in_cam()
        # if route array is empty, then return
        if route is None or not isinstance(route, np.ndarray):
            return
        
        max_size = route.shape[1]-1
        for i in range(max_size):
            start = (int(route[0][i]), int(route[1][i]))
            end = (int(route[0][i+1]), int(route[1][i+1]))
            pygame.draw.line(display, (190, 184, 220), start, end, int(6 - i/max_size*5))
    
    def render_trajectory(self, display, route, color=(255, 0, 0)):
        # if route is not a numpy array, then convert it to numpy array
        if not isinstance(route, np.ndarray):
            return
        
        route_in_camera = self.calc_point_fromW2F(self, route)
        
        if route_in_camera is None or not isinstance(route_in_camera, np.ndarray):
            return
        
        max_size = route_in_camera.shape[1]-1
        for i in range(max_size):
            start = (int(route_in_camera[0][i]), int(route_in_camera[1][i]))
            end = (int(route_in_camera[0][i+1]), int(route_in_camera[1][i+1]))
            pygame.draw.line(display, color, start, end, 2)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
            
# ==============================================================================
# -- vehicle agents ------------------------------------------------------------
# ==============================================================================

class VehicleAgent(object):
    def __init__(self) -> None:
        transform = None # carla.Transform
        predict_trajecotry = None # np.array
        velocity = None # carla.Vector3D


# ==============================================================================
# -- Simple Controller ---------------------------------------------------------
# ==============================================================================

class SimplePlanner(object):
    def __init__(self, world, display, args) -> None:
        self.world = world
        self.control = carla.VehicleControl()
        self.target_route = None
        self.display = display
        
        # initialize the other vehicle agents
        self.vehicle_agents = None
        self.predicted_trajectories = []
        self.csp_target = {} # cubic spline planner target

        self.parse_global_routes(args)
        # this is for the visualization of the route
        world.target_route = self.target_route
        # local planner
        self.lp = FrenetOptimalPlanner(world.player, self.csp_target)

    def parse_global_routes(self, args):
        # get the xml file
        xml_file = f"{SCENARI_RUNNER_ROOT}/AutoControl/config/routes.xml"
        _root = ET.parse(xml_file).getroot()
        start_point = carla.Location()
        end_point = carla.Location()
        scenario_name = args.scenario_name
        
        vehicle_location = self.world.player.get_transform().location
        
        # find the route element
        for _route in _root.findall("route"):
            if _route.get('id') == scenario_name:
                waypoints = _route.find("waypoints")
                start = waypoints.find("start")
                
                start_point.x = float(start.get('x'))
                start_point.y = float(start.get('y'))
                start_point.z = float(start.get('z'))
                
                dist = vehicle_location.distance(start_point)
                
                # calculate the distance between the vehicle and the start point
                print(f"Distance between the vehicle and the start point: {dist}")
                if dist > 100:
                    print("The vehicle is not in the start point")
                    return
                
                start_point = vehicle_location
                end = waypoints.find("end")
                
                end_point.x = float(end.get('x'))
                end_point.y = float(end.get('y'))
                end_point.z = float(end.get('z'))
        
        # set up the sampling resolution
        sampling_resolution = 2.0
        grp = GlobalRoutePlanner(self.world.map, sampling_resolution)
        
        # get the route
        routes = grp.trace_route(start_point, end_point)
        
        # x, y, z, yaw
        self.target_route = np.zeros((len(routes),4))
        for i,route in enumerate(routes):
            self.target_route[i][0] = route[0].transform.location.x
            self.target_route[i][1] = route[0].transform.location.y
            self.target_route[i][2] = route[0].transform.location.z
            self.target_route[i][3] = route[0].transform.rotation.yaw
        
        # calculate the csp target
        list_x = self.target_route[:,0].tolist()
        list_y = self.target_route[:,1].tolist()
        
        t_x, t_y, t_yaw, t_s, _= calc_spline_course(list_x, list_y, 0.01)
        
        self.csp_target["x"] = t_x
        self.csp_target["y"] = t_y
        self.csp_target["yaw"] = t_yaw
        self.csp_target["s"] = t_s
        
        # import matplotlib.pyplot as plt
        # plt.plot(list_x, list_y, "xb")
        # plt.show()
            
    def update_local_planner(self):
        '''
        local_planner should take consideration of the other vehicles
        input of the local planner: 
            predicted trajectory of the other vehicles
            global route
            current state of the ego vehicle
            
        output of the local planner:
            trajectory of the ego vehicle
        '''
        
        # initialize the local planner
        local_planner_route = {
            "candidate routes": None,
            "planner route": None
        }
        
        # candidate_routes, choosed_route = simple_planner(self.world.player, self.csp_target, self.predicted_trajectories)
        candidate_routes, choosed_route = self.lp.update(self.world.player, self.predicted_trajectories, self.csp_target)
        
        local_planner_route["candidate routes"] = candidate_routes
        local_planner_route["planner route"] = choosed_route
        
        return local_planner_route
        
    def compute_control(self, world, route):
        if self.local_planner_route is None:
            print("No local planner route is set")
            return
        
        control = StanleyController(self.local_planner_route, world.player.get_transform())
        return control
    
    def predict_other_vehicles(self, world):
        self.vehicle_agents = []
        trajectories = []
        for npc in world.world.get_actors().filter('vehicle.*'):
            # if the vehicle is the ego vehicle, then skip
            if npc.attributes['role_name'] == 'hero':
                continue
            
            npc_transform = npc.get_transform()
            dist = npc_transform.location.distance(world.player.get_transform().location)

            # if the distance is larger than 50, then skip
            if dist > 100:
                continue
            
            # predict the trajectory of the other vehicles for 20 steps(2s)
            predict_lenth = 20
            dt = 0.1
            
            agent = VehicleAgent()
            velocity = npc.get_velocity()
            
            positions = np.zeros((predict_lenth, 3))
            
            for i in range(predict_lenth):
                positions[i] = (npc_transform.location.x, npc_transform.location.y, npc_transform.location.z)
                npc_transform.location.x += velocity.x * dt
                npc_transform.location.y += velocity.y * dt
                npc_transform.location.z += velocity.z * dt
                
            agent.transform = npc_transform
            agent.predict_trajecotry = positions
            agent.velocity = velocity
            
            self.vehicle_agents.append(agent)
        
        # plot the predict trajectory
        if self.vehicle_agents is not None:
            for agent in self.vehicle_agents:
                trajectories.append(agent.predict_trajecotry)
    
        self.predicted_trajectories = trajectories
        return trajectories

    def parse_events(self, client, world, clock):        
        # this step is for the visualization of the route
        world.predicted_trajectories =  self.predict_other_vehicles(world)
        
        # local planner update
        world.planner_route = self.update_local_planner()
                
        # controller update 
        speed_abs = world.player.get_velocity().x**2 + world.player.get_velocity().y**2 + world.player.get_velocity().z**2
        self.control = StanleyController(world.planner_route["planner route"], world.player.get_transform(), speed_abs)
        
        # judge whether the vehicle has reached the destination
        target = carla.Location()
        target.x = self.target_route[-1][0]
        target.y = self.target_route[-1][1]
        target.z = self.target_route[-1][2]
        if world.player.get_transform().location.distance(target) < 2.0:
            print("The vehicle has reached the destination")
            return True

        # print gnss data
        self.world.player.apply_control(self.control)
    
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        sim_world = client.get_world()

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = SimplePlanner(world, display, args)

        sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            if not world.tick(clock, args.wait_for_repetitions):
                return
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            # prevent destruction of ego vehicle
            if args.keep_ego_vehicle:
                world.player = None
            world.destroy()

        pygame.quit()

# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--scenario_name',
        metavar='sn',
        # default='SingalizedJunctionLeftTurn_1',
        default='FreeRide1',
        help='The scenario name of the vehicle')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot. This does not autocomplete the scenario')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='role name of ego vehicle to control (default: "hero")')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--keep_ego_vehicle',
        action='store_true',
        help='do not destroy ego vehicle on exit')
    argparser.add_argument(
        '--wait-for-repetitions',
        action='store_true',
        help='Avoids stopping the manual control when the scenario ends.')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)