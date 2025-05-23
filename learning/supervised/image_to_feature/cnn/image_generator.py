import numpy as np
import os
import cv2
import itertools as it
import pandas as pd
import torch

from common.utils import numpy_collate
from data_collection.process_data.image_transforms import process_image, augment_image
from data_collection.setup_collect_data import setup_parse
from learning.supervised.image_to_feature.setup_training import setup_model_image, csv_row_to_label

BASE_DATA_PATH = "./tactile_data"


class ImageGenerator(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dirs,
        csv_row_to_label,
        dims=(128, 128),
        bbox=None,
        stdiz=False,
        normlz=False,
        thresh=None,
        rshift=None,
        rzoom=None,
        brightlims=None,
        noise_var=None,
        gray=True
    ):

        # check if data dirs are lists
        assert isinstance(data_dirs, list), "data_dirs should be a list!"

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var
        self._gray = gray

        self._csv_row_to_label = csv_row_to_label

        # load csv file
        self._label_df = self.load_data_dirs(data_dirs)

        new_shear_x = (self._label_df['shear_x'] * np.cos(np.deg2rad(-self._label_df['pose_Rz']))) - (
                self._label_df['shear_y'] * np.sin(np.deg2rad(-self._label_df['pose_Rz'])))
        new_shear_y = (self._label_df['shear_x'] * np.sin(np.deg2rad(-self._label_df['pose_Rz']))) + (
                self._label_df['shear_y'] * np.cos(np.deg2rad(-self._label_df['pose_Rz'])))
        self._label_df.loc[:, "shear_x"] = new_shear_x
        self._label_df.loc[:, "shear_y"] = new_shear_y

    def load_data_dirs(self, data_dirs):

        # check if images or processed images; use for all dirs
        is_processed = os.path.isdir(os.path.join(data_dirs[0], 'processed_images'))

        # add collumn for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:

            # use processed images or fall back on standard images
            if is_processed:
                image_dir = os.path.join(data_dir, 'processed_images')
                df = pd.read_csv(os.path.join(data_dir, 'targets_images.csv'))
            else: 
                image_dir = os.path.join(data_dir, 'images')
                df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))

            df['image_dir'] = image_dir
            df_list.append(df)

        # concat all df
        full_df = pd.concat(df_list)

        return full_df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        row = self._label_df.iloc[index]
        image_filename = os.path.join(row['image_dir'], row['sensor_image'])
        raw_image = cv2.imread(image_filename)

        # preprocess/augment image
        processed_image = process_image(
            raw_image,
            gray=self._gray,
            bbox=self._bbox,
            dims=self._dims,
            stdiz=self._stdiz,
            normlz=self._normlz,
            thresh=self._thresh,
        )

        processed_image = augment_image(
            processed_image,
            rshift=self._rshift,
            rzoom=self._rzoom,
            brightlims=self._brightlims,
            noise_var=self._noise_var
        )

        # put the channel into first axis because pytorch
        processed_image = np.rollaxis(processed_image, 2, 0)

        # get label
        target = self._csv_row_to_label(row)
        sample = {'inputs': processed_image, 'labels': target}

        return sample


def demo_image_generation(
    data_dirs,
    csv_row_to_label,
    learning_params,
    image_processing_params,
    augmentation_params
):

    # Configure dataloaders
    generator_args = {**image_processing_params, **augmentation_params}
    generator = ImageGenerator(
        data_dirs=data_dirs,
        csv_row_to_label=csv_row_to_label,
        **generator_args
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
        collate_fn=numpy_collate
    )

    # iterate through
    for (i_batch, sample_batched) in enumerate(loader, 0):

        # shape = (batch, n_frames, width, height)
        images = sample_batched['inputs']
        labels = sample_batched['labels']
        cv2.namedWindow("example_images")

        for i in range(images.shape[0]):
            for key, item in labels.items():
                print(key, ': ', item[i])
            print('')

            # convert image to opencv format, not pytorch
            image = np.moveaxis(images[i], 0, -1)
            cv2.imshow("example_images", image)
            k = cv2.waitKey(500)
            if k == 27:    # Esc key to stop
                exit()


if __name__ == '__main__':

    args = setup_parse(
        robot='sim_ur',
        sensor='tactip',
        datasets=['edge_2d_shear'],
        data_dirs=['train', 'val']
    )

    output_dir = '_'.join([args.robot, args.sensor])

    data_dirs = [
        os.path.join(BASE_DATA_PATH, output_dir, *i) for i in it.product(args.datasets, args.data_dirs)
    ]

    learning_params = {
        'batch_size': 16,
        'shuffle': True,
        'n_cpu': 1
    }

    image_params = setup_model_image()

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        image_params['image_processing'],
        image_params['augmentation'],
    )