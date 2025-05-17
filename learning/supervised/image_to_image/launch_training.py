"""
python launch_training.py -i ur_tactip -o sim_tactip -t edge_2d -m pix2pix_128 -v tap
"""
import os
import itertools as it

BASE_DATA_PATH = "./tactile_data"

from common.utils import make_dir
from learning.supervised.image_to_image.train_model.image_generator import shPix2PixImageGenerator
from learning.supervised.image_to_image.train_model.models import create_model
from learning.supervised.image_to_image.train_model.train_model import train_shpix2pix
from learning.supervised.image_to_image.utils_learning.utils_learning import seed_everything

from learning.supervised.image_to_image.setup_training import setup_training
from learning.supervised.image_to_image.parse_args import parse_args


def launch(args):

    input_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.inputs, args.tasks, args.train_dirs)
    ]
    target_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.targets, args.tasks, args.train_dirs)
    ]
    input_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.inputs, args.tasks, args.val_dirs)
    ]
    target_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.targets, args.tasks, args.val_dirs)
    ]

    for args.model in args.models:

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))
        output_dir = "_to_".join([*args.inputs, *args.targets])
        task_dir = "_".join(args.tasks)

        # setup save dir
        save_dir = os.path.join(BASE_DATA_PATH, output_dir, task_dir, model_dir_name)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, image_params = setup_training(
            args.model,
            input_train_data_dirs,
            save_dir
        )

        # configure dataloaders
        train_generator = shPix2PixImageGenerator(
            input_train_data_dirs,
            target_train_data_dirs,
            **{**image_params['image_processing'], **image_params['augmentation']}
        )
        val_generator = shPix2PixImageGenerator(
            input_val_data_dirs,
            target_val_data_dirs,
            **image_params['image_processing']
        )

        # create the model
        seed_everything(learning_params['seed'])
        generator, discriminator = create_model(
            image_params['image_processing']['dims'],
            model_params,
            device=args.device
        )

        # run training
        train_shpix2pix(
            generator,
            discriminator,
            train_generator,
            val_generator,
            learning_params,
            image_params['image_processing'],
            save_dir,
            device=args.device,
            debug=True
        )


if __name__ == "__main__":

    args = parse_args(
        inputs=['sim_ur_tactip'],
        targets=['ur_tactip'],
        tasks=['edge_2d'],
        train_dirs=['train_shear'],
        val_dirs=['val_shear'],
        models=['pix2pix_128'],
        # model_version=['']
        device="cuda"
    )

    launch(args)
