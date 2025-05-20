import os
import numpy as np

from tactile_sim.utils.setup_pb_utils import connect_pybullet, load_standard_environment
from tactile_sim.utils.setup_pb_utils import load_stim, set_debug_camera, simple_pb_loop
from tactile_sim.embodiments.create_embodiment import create_embodiment
from tactile_sim.assets.default_rest_poses import rest_poses_dict


def setup_pybullet_env(
    embodiment_type='tactile_arm',
    arm_type='ur5',
    sensor_type='standard_tactip',
    image_size=(128, 128),
    show_tactile=False,
    stim_name='circle',
    stim_path=os.path.dirname(__file__)+'/stimuli',
    stim_pose=(600, 0, 12.5, 0, 0, 0),
    show_gui=True,
    **kwargs
):

    timestep = 1/240.0

    # define sensor parameters
    robot_arm_params = {
        "type": arm_type,
        "rest_poses": rest_poses_dict[arm_type],
        "tcp_lims": np.column_stack([-np.inf*np.ones(6), np.inf*np.ones(6)]),
    }

    tactile_sensor_params = {
        "type": sensor_type,
        "core": "no_core",
        "dynamics": {},  # {'stiffness': 50, 'damping': 100, 'friction': 10.0},
        "image_size": image_size,
        "turn_off_border": False,
        "show_tactile": show_tactile,
    }

    # set debug camera position
    visual_sensor_params = {
        'image_size': [128, 128],
        'dist': 0.25,
        'yaw': 90.0,
        'pitch': -25.0,
        'pos': [0.6, 0.0, 0.0525],
        'fov': 75.0,
        'near_val': 0.1,
        'far_val': 100.0,
        'show_vision': False
    }

    pb = connect_pybullet(timestep, show_gui)
    load_standard_environment(pb)
    stim_name = os.path.join(stim_path, stim_name, stim_name+'.urdf')
    load_stim(pb, stim_name, np.array(stim_pose)/1e3, fixed_base=True)
    embodiment = create_embodiment(
        pb,
        embodiment_type,
        robot_arm_params,
        tactile_sensor_params,
        visual_sensor_params
    )
    set_debug_camera(pb, visual_sensor_params)
    return embodiment


if __name__ == '__main__':
    pass
