"""
python launch_training.py -r sim -s tactip -m simple_cnn -t edge_2d
"""
import os
import itertools as it
# import matplotlib
# matplotlib.use('agg')

from learning.supervised.image_to_feature.cnn.evaluate_model import evaluate_model, evaluate_mdn_model
from learning.supervised.image_to_feature.cnn.image_generator import ImageDataGenerator
from learning.supervised.image_to_feature.cnn.label_encoder import LabelEncoder
from learning.supervised.image_to_feature.models.models import create_model
from learning.supervised.image_to_feature.mdn.train_model import train_mdn_model
from learning.supervised.image_to_feature.cnn.train_model import train_model
from learning.supervised.image_to_image.utils_learning.utils_learning import seed_everything
from common.utils_plots import RegressionPlotter

from common.utils import make_dir

from learning.supervised.image_to_feature.setup_training import setup_training, csv_row_to_label
from learning.supervised.image_to_feature.parse_args import parse_args

BASE_DATA_PATH = "./tactile_data"


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task, args.model in it.product(args.tasks, args.models):

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, d) for d in args.train_dirs
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, d) for d in args.val_dirs
        ]

        # setup save dir
        save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.task, model_dir_name)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, label_params, image_params = setup_training(
            args.model,
            args.task,
            train_data_dirs,
            save_dir
        )

        # configure dataloaders
        train_generator = ImageDataGenerator(
            train_data_dirs,
            csv_row_to_label,
            **{**image_params['image_processing'], **image_params['augmentation']}
        )
        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label,
            **image_params['image_processing']
        )

        # create the label encoder/decoder and plotter
        label_encoder = LabelEncoder(label_params, args.device)
        error_plotter = RegressionPlotter(label_params, save_dir, final_only=False)

        # create the model
        seed_everything(learning_params['seed'])
        model = create_model(
            in_dim=image_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=args.device
        )

        if "_mdn" not in args.model:
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
        else:
            train_mdn_model(
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
        if "_mdn" not in args.model:
            evaluate_model(
                model,
                label_encoder,
                val_generator,
                learning_params,
                error_plotter,
                device=args.device
            )
        else:
            evaluate_mdn_model(
                model,
                label_encoder,
                val_generator,
                learning_params,
                error_plotter,
                device=args.device
            )


if __name__ == "__main__":

    args = parse_args(
        robot='ur',
        sensor='tactip',
        tasks=['edge_2d'],
        train_dirs=['train_shear'],
        val_dirs=['val_shear'],
        # models=['simple_cnn'],
        models=["simple_cnn_mdn_jl"],
        model_version=[''],
        device='cuda'
    )

    launch(args)
