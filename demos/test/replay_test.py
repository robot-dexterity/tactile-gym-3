"""
python test_model.py -r sim -m simple_cnn -t edge_2d
"""
import os
import itertools as it
import pandas as pd

BASE_DATA_PATH = './../tactile_data'

from common.utils import load_json_obj, make_dir
# from common.utils_plots import RegressionPlotter
from data_collection.collect.setup_embodiment import setup_embodiment
from demos.servo.servo_utils.labelled_model import LabelledModel
from demos.servo.setup_servo import setup_parse
from demos.test.launch_test import test
from learning.supervised.image_to_feature.cnn.label_encoder import LabelEncoder
from learning.supervised.image_to_feature.cnn.setup_model import setup_model


def replay(args):

    output_dir = '_'.join([args.robot, args.sensor])

     # test the trained networks
    for args.dataset, args.predict in it.product(args.datasets, args.predicts):
        for args.model, args.sample_num in zip(args.models, args.sample_nums):

            run_dir_name = '_'.join(filter(None, ['test', args.model, *args.run_version]))

            # setup save dir
            run_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, 'predict_' + args.predict, run_dir_name)
            image_dir = os.path.join(run_dir, "processed_images")

            # load model and preproc parameters from model dir
            model_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, 'predict_' + args.predict, args.model)
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
            label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
            sensor_params = {'type': 'replay'}
            
            # load collect, environment params and target_df from run_dir
            collect_params = load_json_obj(os.path.join(run_dir, 'collect_params'))
            env_params = load_json_obj(os.path.join(run_dir, 'env_params'))
            target_df = pd.read_csv(os.path.join(run_dir, 'targets.csv'))

            # setup any plotters
            # error_plotter = RegressionPlotter(model_label_params, run_dir, name='test_plot.png')

            # setup the robot and sensor
            robot, sensor = setup_embodiment(
                env_params,
                sensor_params
            )

            # setup the model
            label_encoder = LabelEncoder(label_params, args.device)

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

            test(
                robot,
                sensor,
                pose_model,
                collect_params,
                label_params,
                target_df,
                image_dir
            )


if __name__ == "__main__":

    args = setup_parse(
        robot='sim',
        sensor='tactip',
        datasets=['edge_xRz_shear'],
        predicts=['pose_xRz'],
        models=['simple_cnn_test'],
        # run_version=['test'],
        device='cuda'
    )

    replay(args)
