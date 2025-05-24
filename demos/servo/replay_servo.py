"""
python replay_servo_control.py -r sim -s tactip -t edge_2d -o circle
"""
import os
import itertools as it

BASE_DATA_PATH = './../tactile_data'

from common.utils import load_json_obj
from data_collection.collect.setup_embodiment import setup_embodiment
from demos.servo.launch_servo import servo
from demos.servo.servo_utils.labelled_model import LabelledModel
from demos.servo.servo_utils.controller import PIDController
from demos.servo.setup_servo import setup_parse
from learning.supervised.image_to_feature.cnn.label_encoder import LabelEncoder
from learning.supervised.image_to_feature.cnn.setup_model import setup_model


def replay(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.dataset, args.predict, args.model, args.object in it.product(args.datasets, args.predicts, args.models, args.objects):

        run_dir_name = '_'.join(filter(None, [args.object, *args.run_version]))

        # setup save dir
        run_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, 'run_' + run_dir_name)
        image_dir = os.path.join(run_dir, "processed_images")

        # load model and preproc parameters from model dir
        model_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, 'predict_' + args.predict, args.model)
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
        label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
        sensor_params = {'type': 'replay'}

        # load control, environment and task parameters from run_dir
        control_params = load_json_obj(os.path.join(run_dir, 'control_params'))
        env_params = load_json_obj(os.path.join(run_dir, 'env_params'))
        task_params = load_json_obj(os.path.join(run_dir, 'task_params'))
        # env_params['work_frame'] += np.array([0, 0, 2, 0, 0, 0])

        # setup the robot and sensor
        robot, sensor = setup_embodiment(
            env_params,
            sensor_params
        )

        # setup the controller
        pid_controller = PIDController(**control_params)

        # setup the model
        label_encoder = LabelEncoder(label_params, device=args.device)

        model = setup_model(
            in_dim=model_image_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=model_dir,
            device=args.device
        )
        model.eval()

        pose_model = LabelledModel(
            model,
            model_image_params['image_processing'],
            label_encoder,
            device=args.device
        )

        # run the servo control
        servo(
            robot,
            sensor,
            pose_model,
            pid_controller,
            image_dir,
            task_params
        )


if __name__ == "__main__":

    args = setup_parse(
        robot='sim',
        sensor='tactip',
        datasets=['edge_yRz_shear'],
        predicts=['pose_yRz'],
        models=['simple_cnn'],
        objects=['circle', 'square'],
        # run_version=['test'],
        device='cuda'
    )

    replay(args)
