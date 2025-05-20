import os

from data_collection.collect_data.simple_sensors import RealSensor, ReplaySensor, SimSensor
from tactile_sim.setup_pybullet_env import setup_pybullet_env

from cri.robot import SyncRobot
from cri.controller import SimController, Controller


def setup_real_embodiment(
    env_params,
    sensor_params,
):
    # setup real robot
    robot = SyncRobot(Controller[env_params['robot']]())
    sensor = RealSensor(sensor_params)

    robot.controller.servo_delay = env_params.get('servo_delay', 0.0)
    robot.coord_frame = env_params['work_frame']
    robot.tcp = env_params['tcp_pose']
    robot.speed = env_params.get('speed', 10)

    return robot, sensor


def setup_sim_embodiment(
    env_params,
    sensor_params,
):
    # setup simulated robot
    embodiment = setup_pybullet_env(**env_params, **sensor_params)
    robot = SyncRobot(SimController(embodiment.arm))
    sensor = SimSensor(sensor_params, embodiment)

    robot.speed = env_params.get('speed', float('inf'))
    robot.controller.servo_delay = env_params.get('servo_delay', 0.0)
    robot.coord_frame = env_params['work_frame']
    robot.tcp = env_params['tcp_pose']

    return robot, sensor


def setup_embodiment(
    env_params,
    sensor_params,
):
    if 'sim' in env_params['robot']:
        robot, sensor = setup_sim_embodiment(env_params, sensor_params)
    else:
        robot, sensor = setup_real_embodiment(env_params, sensor_params)

    # if replay overwrite sensor
    if sensor_params['type'] == 'replay':
        sensor = ReplaySensor(sensor_params)   

    return robot, sensor


if __name__ == '__main__':

    env_params = {
        'robot': 'sim_ur',
        'stim_name': 'mixed_probe_shan',
        'speed': 50,
        'work_frame': (600, 0, 200, 0, 0, 0),
        'tcp_pose': (600, 0, 0, 0, 0, 0),
        'stim_pose': (600, 0, 0, 0, 0, 0),
        'show_gui': True
    }

    sensor_params = {
        "type": "standard_tactip",
        "image_size": (256, 256),
        "show_tactile": False
    }

    robot = setup_embodiment(
        env_params,
        sensor_params
    )
