import os
import numpy as np
import argparse

from common.utils import save_json_obj

SPHERE_LABEL_NAMES = [
    '2mm', '3mm', '4mm',
    '5mm', '6mm', '7mm',
    '8mm', '9mm', '10mm',
]

MIXED_LABEL_NAMES = [
    'cone', 'cross_lines', 'curved_surface', 'cylinder', 'cylinder_shell', 'cylinder_side', 'dot_in',
    'dots', 'flat_slab', 'hexagon', 'line', 'moon', 'pacman', 'parallel_lines',
    'prism', 'random', 'sphere', 'sphere2', 'torus', 'triangle', 'wave1'
]


def setup_parse(
    transfer='',
    robot='sim',
    sensor='tactip',
    datasets=['edge_2d'],
    data_dirs=['data_train', 'data_val'],
    sample_nums=[80, 20],
    device='cuda'
):
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--transfer', nargs='+', help="Options: ['', 'ur_tactip', 'sim_tactip']", default=transfer)
    parser.add_argument('-r', '--robot', type=str, help="Options: ['sim', 'mg400', 'cr']", default=robot)
    parser.add_argument('-s', '--sensor', type=str, help="Options: ['tactip', 'tactip_127']", default=sensor)
    parser.add_argument('-ds', '--datasets', nargs='+', help="Options: ['surface_3d', 'edge_2d', 'spherical_probe']", default=datasets)
    parser.add_argument('-dd', '--data_dirs', nargs='+', help="Default: ['data_train', 'data_val']", default=data_dirs)
    parser.add_argument('-n', '--sample_nums', type=int, help="Default [80, 20]", default=sample_nums)
    parser.add_argument('-d', '--device', type=str, help="Options: ['cpu', 'cuda']", default=device)

    return parser.parse_args()


def setup_sensor_image_params(robot, sensor, save_dir=None):

    bbox_dict = {
        'mini': (320-160,    240-160+25, 320+160,    240+160+25),
        'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
    }
    sensor_type = 'midi'  # TODO: Fix hardcoded sensor type

    if 'sim' in robot:
        sensor_image_params = {
            "type": "standard_tactip",
            "image_size": (256, 256),
            "show_tactile": True
        }

    else:
        sensor_image_params = {
            'type': sensor_type,
            'source': 1,
            'exposure': -7,
            'gray': True,
            'bbox': bbox_dict[sensor_type]
        }

    if save_dir:
        save_json_obj(sensor_image_params, os.path.join(save_dir, 'sensor_image_params'))

    return sensor_image_params


def setup_collect_params(robot, dataset, save_dir=None):

    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    object = dataset.split('_')[0]
    object_poses = '_'.join(dataset.split('_')[:2])
    
    pose_lims_dict = {
        'surface_0':     [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)],
        'surface_zRxRy': [(0, 0, 2.5, -15, -15, 0), (0, 0, 5.5, 15, 15, 0)],
        'edge_xRz':      [(-6, 0, 3, 0, 0, -180),   (6, 0, 5, 0, 0, 180)],
        'edge_yRz':      [(0, -6, 3, 0, 0, -180),   (0, 6, 5, 0, 0, 180)],
        'edge_zzRxRyRz': [(-6, 0, 3, -15, -15, -180),   (6, 0, 5, 15, 15, 180)],
        'edge_yzRxRyRz': [(0, -6, 3, -15, -15, -180),   (0, 6, 5, 15, 15, 180)],
        'spheres_xy':    [(-12.5, -12.5, 4, 0, 0, 0), (12.5, 12.5, 5, 0, 0, 0)],
        'mixed_xy':      [(-5, -5, 4, 0, 0, 0),       (5, 5, 5, 0, 0, 0)],
    }

    shear_lims_dict = {
        'surface_0':     [(-5, -5, 0, 0, 0, 0), (5, 5, 0, 0, 0, 0)],
        'surface_zRxRy': [(-5, -5, 0, -5, -5, -5), (5, 5, 0, 5, 5, 5)],
        'edge_xRz':      [(-5, -5, 0, 0, 0, -5), (5, 5, 0, 0, 0, 5)],
        'edge_yRz':      [(-5, -5, 0, 0, 0, -5), (5, 5, 0, 0, 0, 5)],
        'edge_xzRxRyRz': [(-5, -5, 0, -5, -5, -5), (5, 5, 0, 5, 5, 5)],
        'edge_yzRxRyRz': [(-5, -5, 0, -5, -5, -5), (5, 5, 0, 5, 5, 5)],
    }

    object_poses_dict = {
        'surface':       {'surface': (0, 0, 0, 0, 0, 0)},
        'edge':          {'edge':    (0, 0, 0, 0, 0, 0)},
        'spheres':       {
            SPHERE_LABEL_NAMES[3*i+j]: (60*(1-j), 60*(1-i), 0, 0, 0, -48)
            for i, j in np.ndindex(3, 3)
        },
        'mixed':         {
            MIXED_LABEL_NAMES[7*i+j]: (25*(i-1), 25*(3-j), 0, 0, 0, 0)
            for i, j in np.ndindex(3, 7)
        }
    }

    collect_params = {
        'object_poses': object_poses_dict[object],
        'pose_llims': pose_lims_dict[object_poses][0],
        'pose_ulims': pose_lims_dict[object_poses][1],
        'shear_llims': shear_lims_dict[object_poses][0],
        'shear_ulims': shear_lims_dict[object_poses][1],
        'sample_disk': False,
        'sort': False,
        'seed': 0
    }

    if dataset.split('_')[-1] != 'shear':
        collect_params['shear_llims'] = [0, 0, 0, 0, 0, 0]
        collect_params['shear_ulims'] = [0, 0, 0, 0, 0, 0]

    if robot == 'sim':
        collect_params['sort'] = True

    if save_dir:
        save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))

    return collect_params


def setup_env_params(robot, dataset, save_dir=None):

    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    # pick the correct stimuli dependent on the dataset
    if 'surface' in dataset:
        stim_name = 'square'
        stim_pose = (650, 0, 12.5, 0, 0, 0)
        work_frame_dict = {
            'sim': (650, 0,  50, -180, 0, 90),
            'ur':  (0, -500, 54, -180, 0, 0)
        }
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 101, 0, 0, 0)
        }
    if 'edge' in dataset:
        stim_name = 'square'
        stim_pose = (600, 0, 12.5, 0, 0, 0)
        work_frame_dict = {
            'sim': (650, 0,  50, -180, 0, 90),
            'ur':  (0, -451, 54, -180, 0, 0),
            'mg400': (374, 15, -125, 0, 0, 0),
        }
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 101, 0, 0, 0),
            'mg400': (0, 0, 0, 0, 0, 0)
        }
    if 'spheres' in dataset:
        stim_name = 'spherical_probes'
        stim_pose = (650, 0, 0, 0, 0, 0)
        work_frame_dict = {
            'sim': (650, 0, 42.5, -180, 0, 90),
            'ur': (-15.75, -462, 47.0, -180, 0, 0)
        }
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 88.5, 0, 0, 0)
        }
    if 'mixed' in dataset:
        stim_name = 'mixed_probes'
        stim_pose = (650, 0, 0, 0, 0, 0)
        work_frame_dict = {
            'sim': (650, 0, 20, -180, 0, 0)
        }
        tcp_pose_dict = {
            'sim':   (0, 0, -85, 0, 0, 0),
        }
    if 'alphabet' in dataset or 'arrows' in dataset:
        stim_name = 'static_keyboard'
        stim_pose = (600, 0, 0, 0, 0, 0)
        work_frame_dict = {
            'sim': (593, -7, 25-2, -180, 0, 0),
            'ur':  (111, -473.5, 28.6, -180, 0, -90)
        }
        tcp_pose_dict = {
            'sim':   (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 125, 0, 0, 0)
        }
    if dataset == 'tap_shear_surface':
        stim_name = 'square'
        stim_pose = (600, 0, 12.5, 0, 0, 0)
        work_frame_dict = {
            'sim': (600, 0,  50, -180, 0, 90), #This puts the end effector in the center of the stim
            'ur':  (0, -451, 54, -180, 0, 0),
            'mg400': (318, 15, -125, 0, 0, 0)
        }
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 101, 0, 0, 0),
            'mg400': (0, 0, 0, 0, 0, 0)
        }
    if dataset == 'tap_shear_edge':
        stim_name = 'square'
        stim_pose = (600, 0, 12.5, 0, 0, 0)
        work_frame_dict = {
            'sim': (650, 0, 50, -180, 0, 90),  # This puts the end effector in the center of the TOP EDGE of the stim
            'ur':  (0, -451, 54, -180, 0, 0),
            'mg400': (368, 15, -125, 0, 0, 0)
        }
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 101, 0, 0, 0),
            'mg400': (0, 0, 0, 0, 0, 0)
        }
    if dataset == 'tap_shear_surface_deep':
        stim_name = 'square'
        stim_pose = (600, 0, 12.5, 0, 0, 0)
        work_frame_dict = {
            'sim': (600, 0,  50, -180, 0, 90), #This puts the end effector in the center of the stim
            'ur':  (0, -451, 54, -180, 0, 0),
            'mg400': (318, 15, -126, 0, 0, 0)
        }
        tcp_pose_dict = {
            'sim': (0, 0, -85, 0, 0, 0),
            'ur':  (0, 0, 101, 0, 0, 0),
            'mg400': (0, 0, 0, 0, 0, 0)
        }

    env_params = {
        'robot': robot,
        'stim_name': stim_name,
        'speed': 50,
        'work_frame': work_frame_dict[robot],
        'tcp_pose': tcp_pose_dict[robot],
    }

    if 'sim' in robot:
        env_params['speed'] = float('inf')
        env_params['stim_pose'] = stim_pose

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_collect_data(robot, sensor, dataset, save_dir=None):
    sensor_image_params = setup_sensor_image_params(robot, sensor, save_dir)
    env_params = setup_env_params(robot, dataset, save_dir)

    return env_params, sensor_image_params


if __name__ == '__main__':
    pass
