import socket
import json
import time
from typing import Any

# this udp sever is only used for processing IMU and GNSS data
# if you want to have more functions, you should modify this class
class udp_server():
    def __init__(self, name = None):
        self.dest_host = 'localhost'
        self.dest_port = 9870
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.name = name

        if name == None:
            self.runtime_data = {
                "timestamp": 0.0,
                "IMU": {
                    "accelerometer": [0.0, 0.0, 0.0],
                    "gyroscope": [0.0, 0.0, 0.0]
                },
                "GNSS": {
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "altitude": 0.0
                },
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0,
            }
        else:
            self.runtime_data = {
                "timestamp": 0.0,
                name:
                    {
                    "IMU": {
                        "accelerometer": [0.0, 0.0, 0.0],
                        "gyroscope": [0.0, 0.0, 0.0]
                    },
                    "GNSS": {
                        "latitude": 0.0,
                        "longitude": 0.0,
                        "altitude": 0.0
                    },
                    "control": {
                    "throttle": 0.0,
                    "brake": 0.0,
                    "steer": 0.0,
                    },
                }
            }

    def update_throttle(self, throttle):
        if self.name == None:
            self.runtime_data["control"]['throttle'] = throttle
        else:
            self.runtime_data[self.name]["control"]['throttle'] = throttle
    
    def update_brake(self, brake):
        if self.name == None:
            self.runtime_data["control"]['brake'] = brake
        else:
            self.runtime_data[self.name]["control"]['brake'] = brake
    
    def update_steer(self, steer):
        if self.name == None:
            self.runtime_data["control"]['steer'] = steer
        else:
            self.runtime_data[self.name]["control"]['steer'] = steer
    
    def update_control(self, control):
        self.update_throttle(control.throttle)
        self.update_brake(control.brake)
        self.update_steer(control.steer)

    def update_IMU(self, acc_gyro):
        acc = acc_gyro[0]
        gyro = acc_gyro[1]
        if self.name == None:
            self.runtime_data['IMU']['accelerometer'] = acc
            self.runtime_data['IMU']['gyroscope'] = gyro
        else:
            self.runtime_data[self.name]['IMU']['accelerometer'] = acc
            self.runtime_data[self.name]['IMU']['gyroscope'] = gyro
    
    def update_GNSS(self, gnss):
        if self.name == None:
            self.runtime_data['GNSS']['latitude'] = gnss[0]
            self.runtime_data['GNSS']['longitude'] = gnss[1]
            self.runtime_data['GNSS']['altitude'] = gnss[2]
        else:
            self.runtime_data[self.name]['GNSS']['latitude'] = gnss[0]
            self.runtime_data[self.name]['GNSS']['longitude'] = gnss[1]
            self.runtime_data[self.name]['GNSS']['altitude'] = gnss[2]

    def update(self, snap = None):
        # update timestamp
        if snap is None:
            self.runtime_data['timestamp'] = time.time()
        else:
            self.runtime_data['timestamp'] = snap
        # send data
        self.sock.sendto(json.dumps(self.runtime_data).encode(), (self.dest_host, self.dest_port))

def send_custom_data(data):
    dest_host = 'localhost'
    dest_port = 9870
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(data).encode(), (dest_host, dest_port))

class custom_data_sender:
    def __init__(self):
        self.data = dict()

    def update_data(self, data):
        # if key is not in the dict, add it
        for key in data.keys():
            if key not in self.data.keys():
                self.data[key] = data[key]
            else:
                self.data[key] = data[key]

        # if key is in the dict, update it
        for key in data.keys():
            if key in self.data.keys():
                self.data[key] = data[key]
    
    def send_data(self):
        if self.data is not None:
            send_custom_data(self.data)