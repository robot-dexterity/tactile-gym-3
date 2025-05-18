import os
import numpy as np
import argparse

from common.utils import save_json_obj


def setup_parse(
    robot='sim',
    sensor='tactip',
    datasets=['edge_2d'],
    tasks=['servo_2d'],
    models=['simple_cnn'],
    objects=['circle'],
    sample_nums=[100],
    run_version=[''],
    device='cuda'
):
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--robot', type=str, help="Options: ['sim', 'mg400', 'cr']", default=robot)
    parser.add_argument('-s', '--sensor', type=str, help="Options: ['tactip', 'tactip_127']", default=sensor)
    parser.add_argument('-ds', '--datasets', nargs='+', help="Options: ['surface_3d', 'edge_2d', 'spherical_probe']", default=datasets)
    parser.add_argument('-dt', '--train_dirs', nargs='+', help="Default: ['train']", default=['train'])
    parser.add_argument('-t', '--tasks', nargs='+', help="Options: ['servo_2d', 'servo_3d', 'servo_5d', 'track_2d', 'track_3d', 'track_4d']", default=tasks)
    parser.add_argument('-m', '--models', nargs='+', help="Options: ['simple_cnn', 'nature_cnn', 'posenet', 'resnet', 'vit']", default=models)
    parser.add_argument('-o', '--objects', nargs='+', help="Options: ['circle', 'square']", default=objects)
    parser.add_argument('-n', '--sample_nums', type=int, help="Default [100]", default=sample_nums)
    parser.add_argument('-rv', '--run_version', nargs='+', help="Default: ['']", default=run_version)
    parser.add_argument('-d', '--device', type=str, help="Options: ['cpu', 'cuda']", default=device)
    
    return parser.parse_args()


def setup_control_params(task, save_dir=None):

    if task == 'surface_3d':
        control_params = {
            'kp': [1, 1, 0.5, 0.5, 0.5, 1],
            'ki': [0, 0, 0.3, 0.1, 0.1, 0],
            'ei_clip': [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 1, 3, 0, 0, 0]
        }

    elif task == 'edge_2d':
        control_params = {
            'kp': [0.5, 1, 0, 0, 0, 0.5],
            'ki': [0.3, 0, 0, 0, 0, 0.1],
            'ei_clip': [[-5, 0, 0, 0, 0, -45], [5, 0, 0, 0, 0,  45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 0, 0, 0, 0]
        }

    elif task == 'edge_3d':
        control_params = {
            'kp': [0.5, 1, 0.5, 0, 0, 0.5],
            'ki': [0.3, 0, 0.3, 0, 0, 0.1],
            'ei_clip': [[-5, 0, -2.5, 0, 0, -45], [5, 0, 2.5, 0, 0, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 3.5, 0, 0, 0]
        }

    elif task == 'edge_5d':
        control_params = {
            'kp': [0.5, 1, 0.5, 0.5, 0.5, 0.5],
            'ki': [0.3, 0, 0.3, 0.1, 0.1, 0.1],
            'ei_clip': [[-5, 0, -2.5, -15, -15, -45], [5, 0, 2.5, 15, 15, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, -2, 3.5, 0, 0, 0]
        }

    if save_dir:
        save_json_obj(control_params, os.path.join(save_dir, 'control_params'))

    return control_params


def update_env_params(env_params, object, save_dir=None):

    wf_offset_dict = {
        'saddle':  (-10, 0, 18.5, 0, 0, 0),
        'default': (0, 0, 3.5, 0, 0, 0)
    }
    wf_offset = np.array(wf_offset_dict.get(object, wf_offset_dict['default']))

    env_params.update({
        'stim_name': object,
        'work_frame': tuple(env_params['work_frame'] - wf_offset),
        'speed': 20,
    })

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_task_params(sample_num, model_dir, save_dir=None):

    task_params = {
        'num_iterations': sample_num,
        'show_plot': True,
        'show_slider': False,
        'model': model_dir
        # 'servo_delay': 0.0,
    }

    if save_dir:
        save_json_obj(task_params, os.path.join(save_dir, 'task_params'))

    return task_params


def setup_servo_control(sample_num, task, object, model_dir, env_params, save_dir=None):
    env_params = update_env_params(env_params, object, save_dir)
    control_params = setup_control_params(task, save_dir)
    task_params = setup_task_params(sample_num, model_dir, save_dir)

    return control_params, env_params, task_params


if __name__ == '__main__':
    pass
