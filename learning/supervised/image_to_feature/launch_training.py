"""
python launch_training.py -r sim -s tactip -m simple_cnn -t edge_2d
"""
import os
import itertools as it

from common.utils import make_dir, seed_everything
from common.utils_plots import RegressionPlotter
from learning.supervised.image_to_feature.cnn.image_generator import ImageGenerator
from learning.supervised.image_to_feature.cnn.label_encoder import LabelEncoder

from learning.supervised.image_to_feature.cnn.setup_model import setup_model as setup_cnn_model
from learning.supervised.image_to_feature.cnn.evaluate_model import evaluate_model as evaluate_cnn_model
from learning.supervised.image_to_feature.cnn.train_model import train_model as train_cnn_model
from learning.supervised.image_to_feature.mdn.setup_model import setup_model as setup_mdn_model
from learning.supervised.image_to_feature.mdn.evaluate_model import evaluate_model as evaluate_mdn_model
from learning.supervised.image_to_feature.mdn.train_model import train_model as train_mdn_model

from learning.supervised.image_to_feature.setup_training import setup_training, csv_row_to_label
from learning.supervised.image_to_feature.parse_args import parse_args

BASE_DATA_PATH = "./tactile_data"


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.dataset, args.task, args.model in it.product(args.datasets, args.tasks, args.models):

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.dataset, d) for d in args.train_dirs
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.dataset, d) for d in args.val_dirs
        ]

        # setup save dir
        save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, args.task, args.model)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, label_params, image_params = setup_training(
            args.model,
            args.task,
            train_data_dirs,
            save_dir
        )

        # configure dataloaders
        train_generator = ImageGenerator(
            train_data_dirs,
            csv_row_to_label,
            **{**image_params['image_processing'], **image_params['augmentation']}
        )
        val_generator = ImageGenerator(
            val_data_dirs,
            csv_row_to_label,
            **image_params['image_processing']
        )

        # create the label encoder/decoder and plotter
        label_encoder = LabelEncoder(label_params, args.device)
        error_plotter = RegressionPlotter(label_params, save_dir, final_only=False)

        # create the model
        seed_everything(learning_params['seed'])
        model = setup_model(
            in_dim=image_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=args.device
        )
        train_model(
            prediction_mode='regression',
            model=model,
            label_encoder=label_encoder,
            train_generator=train_generator,
            val_generator=val_generator,
            learning_params=learning_params,
            save_dir=save_dir,
            error_plotter=error_plotter,
            device=args.device
        )
        
        # perform a final evaluation using the last model
        evaluate_model(
            model,
            label_encoder,
            val_generator,
            learning_params,
            error_plotter,
            device=args.device
        )

def setup_model(**kwargs):
    if "_mdn" not in args.model:
        model = setup_cnn_model(**kwargs)
    else:
        model = setup_mdn_model(**kwargs)
    return model

def train_model(**kwargs):
    if "_mdn" not in args.model:
        train_cnn_model(**kwargs)
    else:
        train_mdn_model(**kwargs)

def evaluate_model(**kwargs):
    if "_mdn" not in args.model:
        evaluate_cnn_model(**kwargs)
    else:
        evaluate_mdn_model(**kwargs)


if __name__ == "__main__":

    args = parse_args(
        robot='sim_ur',
        sensor='tactip',
        datasets=['surface_3d_shear'],
        tasks=['servo_3d'],
        models=['simple_cnn_mdn_test','simple_cnn_test','posenet_mdn_test','posenet_test'],
        train_dirs=['train'],
        val_dirs=['val'],
        device='cuda'
    )

    launch(args)
