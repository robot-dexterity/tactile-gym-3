"""
python launch_training.py -r sim -s tactip -m simple_cnn -t edge_2d
"""
import os
import itertools as it

BASE_DATA_PATH = './../tactile_data'

from common.utils import make_dir, seed_everything
from learning.supervised.image_to_feature.cnn.evaluate_model import evaluate_model as evaluate_cnn_model
from learning.supervised.image_to_feature.cnn.image_generator import ImageGenerator
from learning.supervised.image_to_feature.cnn.label_encoder import LabelEncoder
from learning.supervised.image_to_feature.cnn.setup_model import setup_model as setup_cnn_model
from learning.supervised.image_to_feature.cnn.train_model import train_model as train_cnn_model
from learning.supervised.image_to_feature.cnn.utils_plots import RegressionPlotter
from learning.supervised.image_to_feature.mdn.evaluate_model import evaluate_model as evaluate_mdn_model
from learning.supervised.image_to_feature.mdn.setup_model import setup_model as setup_mdn_model
from learning.supervised.image_to_feature.mdn.train_model import train_model as train_mdn_model
from learning.supervised.image_to_feature.setup_training import setup_training, setup_parse, csv_row_to_label


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.dataset, args.predict, args.model in it.product(args.datasets, args.predicts, args.models):

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.dataset, d) for d in args.train_dirs
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.dataset, d) for d in args.val_dirs
        ]

        # setup save dir
        save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, 'predict_' + args.predict, args.model)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, label_params, image_params = setup_training(
            args.model,
            args.predict,
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

        # setup plotters
        error_plotter = RegressionPlotter(save_dir, label_params)#, final_only=True)

        # setup the model
        label_encoder = LabelEncoder(label_params, args.device)

        seed_everything(learning_params['seed'])
        model = setup_model(
            in_dim=image_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=args.device
        )

        # run the training
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
        
        # run an evaluation using the last model
        evaluate_model(
            model=model,
            label_encoder=label_encoder,
            generator=val_generator,
            learning_params=learning_params,
            error_plotter=error_plotter,
            device=args.device
        )

def setup_model(**kwargs):
    if '_mdn' not in args.model:
        model = setup_cnn_model(**kwargs)
    else:
        model = setup_mdn_model(**kwargs)
    return model

def train_model(**kwargs):
    if '_mdn' not in args.model:
        train_cnn_model(**kwargs)
    else:
        train_mdn_model(**kwargs)

def evaluate_model(**kwargs):
    if '_mdn' not in args.model:
        evaluate_cnn_model(**kwargs)
    else:
        evaluate_mdn_model(**kwargs)


if __name__ == "__main__":

    args = setup_parse(
        robot='sim',
        sensor='tactip',
        datasets=['edge_yRz_shear'],
        predicts=['pose_yRz'],
        models=['simple_cnn_test'],
        train_dirs=['data_train'],
        val_dirs=['data_val'],
        device='cuda'
    )

    launch(args)
