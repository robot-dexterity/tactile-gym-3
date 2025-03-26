import argparse


def parse_args(
    robot='sim',
    sensor='tactip',
    tasks=['edge_2d'],
    inputs=[''],
    data_dirs=['train', 'val'],
    sample_nums=[80, 20],
    train_dirs=['train'],
    val_dirs=['val'],
    targets=['sim_tactip'],
    models=['pix2pix_128'],
    model_version=[],
    device='cuda'
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r', '--robot',
        type=str,
        help="Choose robot from ['sim', 'mg400', 'cr']",
        default=robot
    )
    parser.add_argument(
        '-s', '--sensor',
        type=str,
        help="Choose sensor from ['tactip', 'tactip_127']",
        default=sensor
    )
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose tasks from ['surface_3d', 'edge_2d', 'spherical_probe']",
        default=tasks
    )
    parser.add_argument(
        '-i', '--inputs',
        nargs='+',
        help="Choose input directory from ['', 'ur_tactip', 'sim_tactip'].",
        default=inputs
    )
    parser.add_argument(
        '-dd', '--data_dirs',
        nargs='+',
        help="Specify data directories (default ['train', 'val']).",
        default=data_dirs
    )
    parser.add_argument(
        '-n', '--sample_nums',
        type=int,
        help="Choose numbers of samples (default [80, 20]).",
        default=sample_nums
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
        '-o', '--targets',
        nargs='+',
        help="Choose target directory from ['ur_tactip', 'sim_tactip'].",
        default=targets
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['pix2pix'].",
        default=models
    )
    parser.add_argument(
        '-mv', '--model_version',
        type=str,
        help="Choose model version]",
        default=model_version
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda']",
        default=device
    )

    return parser.parse_args()
