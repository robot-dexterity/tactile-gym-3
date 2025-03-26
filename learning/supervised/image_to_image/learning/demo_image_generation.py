"""
python demo_image_generation.py -i sim_tactip -o cr_tactip -t edge_2d -iv tap -tv data
"""
import os
import itertools as it

from tactile_data.braille_classification import BASE_DATA_PATH as INPUT_DATA_PATH
from tactile_data.tactile_sim2real import BASE_DATA_PATH as TARGET_DATA_PATH
from tactile_learning.pix2pix.image_generator import demo_image_generation

from tactile_sim2real.learning.setup_training import setup_learning, setup_model_image
from tactile_sim2real.utils.parse_args import parse_args


if __name__ == '__main__':

    args = parse_args(
        inputs=['ur_tactip_small'],
        targets=['sim_ur_tactip'],
        tasks=['alphabet'],
        data_dirs=['train', 'val']
    )

    learning_params = setup_learning()
    image_params = setup_model_image()


    # combine the data directories
    input_data_dirs = [
        os.path.join(INPUT_DATA_PATH, *i) for i in it.product(args.inputs, args.tasks, args.data_dirs)
    ]
    target_data_dirs = [
        os.path.join(TARGET_DATA_PATH, *i) for i in it.product(args.targets, args.tasks, args.data_dirs)
    ]

    demo_image_generation(
        input_data_dirs,
        target_data_dirs,
        learning_params,
        image_params['image_processing'],
        image_params['augmentation'],
    )
