"""
python launch_collect_data.py -i cr_tactip -r sim_cr -s tactip -t edge_5d
"""
import os
import itertools as it
import pandas as pd

from tactile_data.braille_classification import BASE_DATA_PATH as INPUT_DATA_PATH
from tactile_data.tactile_sim2real import BASE_DATA_PATH as TARGET_DATA_PATH
from tactile_image_processing.collect_data.collect_data import collect_data
from tactile_image_processing.collect_data.setup_targets import setup_targets
from tactile_image_processing.process_data.process_image_data import process_image_data, partition_data
from tactile_image_processing.utils import make_dir, load_json_obj, save_json_obj

from tactile_sim2real.collect_data.setup_collect_data import setup_collect_data, setup_collect_params
from tactile_sim2real.utils.setup_embodiment import setup_embodiment
from tactile_sim2real.utils.parse_args import parse_args


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task, args.input in it.product(args.tasks, args.inputs):
        for args.data_dir, args.sample_num in zip(args.data_dirs, args.sample_nums):

            # setup save dir
            save_dir = os.path.join(TARGET_DATA_PATH, output_dir, args.task, args.data_dir)
            image_dir = os.path.join(save_dir, "sensor_images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            env_params, sensor_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_params
            )

            if args.input:
                # load and save targets to collect
                load_dir = os.path.join(INPUT_DATA_PATH, args.input, args.task, args.data_dir)
                collect_params = load_json_obj(os.path.join(load_dir, 'collect_params'))              
                target_df = pd.read_csv(os.path.join(load_dir, 'targets_images.csv'))

                save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))
                target_df.to_csv(os.path.join(save_dir, "targets.csv"), index=False)

            else:
                # setup targets to collect
                collect_params = setup_collect_params(args.robot, args.task, save_dir)
                target_df = setup_targets(
                    collect_params,
                    args.sample_num,
                    save_dir
                )

            # collect
            collect_data(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


def process_images(args, image_params, split=None):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        path = os.path.join(TARGET_DATA_PATH, output_dir, args.task)

        dir_names = partition_data(path, args.data_dirs, split)
        process_image_data(path, dir_names, image_params)


if __name__ == "__main__":

    args = parse_args(
        inputs=['ur_tactip_small'],
        robot='sim_ur',
        sensor='tactip',
        tasks=['alphabet'],
        data_dirs=['train', 'val'],
        # sample_nums=[10]
    )
    launch(args)

    image_params = {
        "bbox": (12, 12, 240, 240)  
    }
    process_images(args, image_params)  # , split=0.8)
