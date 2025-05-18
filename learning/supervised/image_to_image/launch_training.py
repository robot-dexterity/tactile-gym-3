"""
python launch_training.py -i ur_tactip -o sim_tactip -t edge_2d -m pix2pix_128 -v tap
"""
import os
import itertools as it

BASE_DATA_PATH = "./tactile_data"

from common.utils import make_dir, seed_everything
from learning.supervised.image_to_image.pix2pix.image_generator import Image2ImageGenerator
from learning.supervised.image_to_image.pix2pix.models import create_model
from learning.supervised.image_to_image.pix2pix.train_model import train_model
from learning.supervised.image_to_image.pix2pix.image_generator import Image2ImageGenerator as ShImage2ImageGenerator
from learning.supervised.image_to_image.pix2pix.models import create_model as create_sh_model
from learning.supervised.image_to_image.pix2pix.train_model import train_model as train_sh_model

from learning.supervised.image_to_image.setup_training import setup_training
from learning.supervised.image_to_image.parse_args import parse_args


def launch(args):

    input_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.inputs, args.datasets, args.train_dirs)
    ]
    target_train_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.targets, args.datasets, args.train_dirs)
    ]
    input_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.inputs, args.datasets, args.val_dirs)
    ]
    target_val_data_dirs = [
        os.path.join(BASE_DATA_PATH, *i) for i in it.product(args.targets, args.datasets, args.val_dirs)
    ]

    for args.model in args.models:

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))
        output_dir = "_to_".join([*args.inputs, *args.targets])
        task_dir = "_".join(args.datasets)

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
        if 'shpix2pix' in args.model:
            train_generator = ShImage2ImageGenerator(
                input_train_data_dirs,
                target_train_data_dirs,
                **{**image_params['image_processing'], **image_params['augmentation']}
            )
            val_generator = ShImage2ImageGenerator(
                input_val_data_dirs,
                target_val_data_dirs,
                **image_params['image_processing']
            )
        else:
            train_generator = Image2ImageGenerator(
                input_train_data_dirs,
                target_train_data_dirs,
                **{**image_params['image_processing'], **image_params['augmentation']}
            )
            val_generator = Image2ImageGenerator(
                input_val_data_dirs,
                target_val_data_dirs,
                **image_params['image_processing']
            )

        # create the model
        seed_everything(learning_params['seed'])
        if 'shpix2pix' in args.model:
            generator, discriminator = create_sh_model(
                image_params['image_processing']['dims'],
                model_params,
                device=args.device
            )
        else:
            generator, discriminator = create_model(
                image_params['image_processing']['dims'],
                model_params,
                device=args.device
            )

        # run training
        if 'shpix2pix' in args.model:
            train_sh_model(
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
        else:
            train_model(
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
        datasets=['edge_2d_shear'],
        train_dirs=['train'],
        val_dirs=['val'],
        models=['shpix2pix_128'],
        # model_version=['']
        device='cuda'
    )

    launch(args)
