import argparse


def parse_args(
    robot='sim',
    sensor='tactip',
    inputs=[''],
    datasets=['edge_2d'],
    data_dirs=['train', 'val'],
    sample_nums=[80, 20],
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
        '-i', '--inputs',
        nargs='+',
        help="Choose input directory from ['', 'ur_tactip', 'sim_tactip'].",
        default=inputs
    )
    parser.add_argument(
        '-ds', '--datasets',
        nargs='+',
        help="Choose datasets from ['surface_3d', 'edge_2d', 'spherical_probe']",
        default=datasets
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
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda']",
        default=device
    )

    return parser.parse_args()