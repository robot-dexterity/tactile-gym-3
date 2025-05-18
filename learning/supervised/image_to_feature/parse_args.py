import argparse


def parse_args(
        robot='sim',
        sensor='tactip',
        datasets=['edge_2d'],
        tasks=['servo_2d'],
        train_dirs=['train'],
        val_dirs=['val'],
        models=['simple_cnn'],
        model_version=[],
        sample_nums=[100],
        device='cuda'
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r', '--robot',
        type=str,
        help="Choose robot from ['sim', 'mg400', 'cr', 'ur']",
        default=robot
    )
    parser.add_argument(
        '-s', '--sensor',
        type=str,
        help="Choose sensor from ['tactip', 'tactip_127']",
        default=sensor
    )
    parser.add_argument(
        '-ds', '--datasets',
        nargs='+',
        help="Choose datasets from ['edge_2d', 'edge_2d_shear', 'surface_3d', 'surface_3d_shear', 'spherical_probe']",
        default=datasets
    )
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose tasks from ['servo_2d', 'servo_3d', 'servo_5d', 'track_2d', 'track_3d', 'track_4d']",
        default=tasks
    )
    parser.add_argument(
        '-dt', '--train_dirs',
        nargs='+',
        help="Specify train data directories (default ['train').",
        default=train_dirs
    )
    parser.add_argument(
        '-dv', '--val_dirs',
        nargs='+',
        help="Specify validation data directories (default ['val']).",
        default=val_dirs
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose models from ['simple_cnn', 'nature_cnn', 'posenet', 'resnet', 'vit']",
        default=models
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda']",
        default=device
    )

    return parser.parse_args()
