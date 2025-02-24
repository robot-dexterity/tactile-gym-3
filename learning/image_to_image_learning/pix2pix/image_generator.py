import numpy as np
import os
import cv2
import pandas as pd
import torch

from tactile_learning.supervised.image_generator import numpy_collate
from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.image_transforms import augment_image


class Pix2PixImageGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data_dirs,
        target_data_dirs,
        dims=(128, 128),
        bbox=None,
        stdiz=False,
        normlz=False,
        thresh=None,
        rshift=None,
        rzoom=None,
        brightlims=None,
        noise_var=None,
        joint_aug=False,
    ):

        # check if data dirs are lists
        assert isinstance(
            input_data_dirs, list
        ), "input_data_dirs should be a list!"
        assert isinstance(
            target_data_dirs, list
        ), "target_data_dirs should be a list!"

        self._dims = dims
        self._bbox = bbox
        self._stdiz = stdiz
        self._normlz = normlz
        self._thresh = thresh
        self._rshift = rshift
        self._rzoom = rzoom
        self._brightlims = brightlims
        self._noise_var = noise_var
        self._joint_aug = joint_aug

        # load csv file
        self.input_label_df = self.load_data_dirs(input_data_dirs)
        self.target_label_df = self.load_data_dirs(target_data_dirs)

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
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.input_label_df)))

    def __getitem__(self, index):
        "Generate one batch of data"

        # Generate data
        input_image_filename = os.path.join(
            self.input_label_df.iloc[index]["image_dir"],
            self.input_label_df.iloc[index]["sensor_image"],
        )
        target_image_filename = os.path.join(
            self.target_label_df.iloc[index]["image_dir"],
            self.target_label_df.iloc[index]["sensor_image"],
        )

        raw_input_image = cv2.imread(input_image_filename)
        raw_target_image = cv2.imread(target_image_filename)

        # preprocess/augment images separetly
        processed_input_image = process_image(
            raw_input_image,
            gray=True,
            bbox=self._bbox,
            dims=self._dims,
            stdiz=self._stdiz,
            normlz=self._normlz,
            thresh=self._thresh,
        )

        processed_target_image = process_image(
            raw_target_image,
            gray=True,
            bbox=None,
            dims=self._dims,
            stdiz=self._stdiz,
            normlz=self._normlz,
        )

        if self._joint_aug:
            # stack images to apply the same data augmentations to both
            stacked_image = np.dstack(
                [processed_input_image, processed_target_image],
            )

            # apply shift/zoom augs
            augmented_images = augment_image(
                stacked_image,
                rshift=self._rshift,
                rzoom=self._rzoom,
                brightlims=self._brightlims,
                noise_var=self._noise_var
            )

            # print(augmented_images.shape)
            # unstack the images
            processed_input_image = augmented_images[..., 0]
            processed_target_image = augmented_images[..., 1]

            # put the channel into first axis because pytorch
            processed_input_image = processed_input_image[np.newaxis, ...]
            processed_target_image = processed_target_image[np.newaxis, ...]

        else:
            processed_input_image = augment_image(
                processed_input_image,
                rshift=self._rshift,
                rzoom=self._rzoom,
                brightlims=self._brightlims,
                noise_var=self._noise_var
            )
            processed_target_image = augment_image(
                processed_target_image,
                rshift=self._rshift,
                rzoom=self._rzoom,
                brightlims=self._brightlims,
                noise_var=self._noise_var
            )

            # put the channel into first axis because pytorch
            processed_input_image = np.rollaxis(processed_input_image, 2, 0)
            processed_target_image = np.rollaxis(processed_target_image, 2, 0)

        return {"input": processed_input_image, "target": processed_target_image}


def demo_image_generation(
    input_data_dirs,
    target_data_dirs,
    learning_params,
    image_processing_params,
    augmentation_params
):
    # Configure dataloaders
    generator_args = {**image_processing_params, **augmentation_params}
    generator = Pix2PixImageGenerator(
        input_data_dirs=input_data_dirs,
        target_data_dirs=target_data_dirs,
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
        input_images = sample_batched["input"]
        target_images = sample_batched["target"]

        cv2.namedWindow("training_images")

        for i in range(learning_params['batch_size']):

            # convert image to opencv format, not pytorch
            input_image = np.moveaxis(input_images[i], 0, -1)
            target_image = np.moveaxis(target_images[i], 0, -1)
            overlay_image = cv2.addWeighted(
                input_image, 0.5,
                target_image, 0.5,
                0
            )[..., np.newaxis]

            disp_image = np.concatenate(
                [input_image, target_image, overlay_image], axis=1
            )

            cv2.imshow("training_images", disp_image)
            k = cv2.waitKey(500)
            if k == 27:    # Esc key to stop
                exit()
