import numpy as np
import os
import cv2
import pandas as pd
import torch

from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.supervised.image_generator import numpy_collate

from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.image_transforms import augment_image


class FrameDataGenerator(ImageDataGenerator):

    def __init__(
        self,
        data_dirs,
        csv_row_to_label,
        n_stack=1,
        dims=(128, 128),
        bbox=None,
        stdiz=False,
        normlz=False,
        thresh=None,
        rshift=None,
        rzoom=None,
        brightlims=None,
        noise_var=None,
    ):

        # check if data dirs are lists
        assert isinstance(data_dirs, list), "data_dirs should be a list!"
        super(FrameDataGenerator, self).__init__(
            data_dirs,
            csv_row_to_label,
            dims,
            bbox,
            stdiz,
            normlz,
            thresh,
            rshift,
            rzoom,
            brightlims,
            noise_var,
        )
        self._n_stack = n_stack

    def load_data_dirs(self, data_dirs):

        # add collumn for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:
            df = pd.read_csv(os.path.join(data_dir, 'per_frame_targets.csv'))

            # add frame dir
            frame_dir = os.path.join(data_dir, 'processed_frames', str(self._dims[0]))

            if not os.path.isdir(frame_dir):
                raise ValueError('Frame directory %s not found.', str(frame_dir))

            df['frame_dir'] = frame_dir
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

        processed_frames = []
        for i in reversed(range(self._n_stack)):

            video_index = int(row['frame_id'].split('_')[1])
            frame_index = int(row['frame_id'].split('_')[3].split('.')[0])
            frame_index = np.clip(frame_index - i, row['first_frame_index'], row['last_frame_index'])

            frame_filename = os.path.join(
                row['frame_dir'],
                f'video_{video_index}_frame_{frame_index}.png'
            )
            loaded_frame = cv2.imread(frame_filename)

            processed_frame = process_image(
                loaded_frame,
                gray=True,
                bbox=self._bbox,
                dims=self._dims,
                stdiz=self._stdiz,
                normlz=self._normlz,
                thresh=self._thresh,
            )

            processed_frames.append(processed_frame)

        # combine frame stack + channel dimensions
        # TODO: extend to include n_channels (currently only works for n_channels = 1)
        processed_frames = np.moveaxis(processed_frames, 0, -1)
        processed_frames = processed_frames.reshape(*self._dims, self._n_stack)

        # applt image augmentation across frames
        processed_frames = augment_image(
            processed_frames,
            rshift=self._rshift,
            rzoom=self._rzoom,
            brightlims=self._brightlims,
            noise_var=self._noise_var
        )

        # put the channel into first axis because pytorch
        processed_frames = np.moveaxis(processed_frames, -1, 0)

        # get label
        target = self._csv_row_to_label(row)
        sample = {'images': processed_frames, 'labels': target}

        return sample


def demo_frame_generation(
    data_dirs,
    csv_row_to_label,
    learning_params,
    image_processing_params,
    augmentation_params
):

    # Configure dataloaders
    generator_args = {**image_processing_params, **augmentation_params}
    generator = FrameDataGenerator(
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
        images = sample_batched['images']
        labels = sample_batched['labels']
        cv2.namedWindow("example_frames")

        for i in range(learning_params['batch_size']):
            for key, item in labels.items():
                print(key.split('_')[0], ': ', item[i])
            print('')

            # convert image to opencv format, not pytorch
            frames = np.moveaxis(images[i], 0, -1)

            for j in range(frames.shape[2]):
                frame = frames[..., j]
                # show image
                cv2.imshow("example_frames", frame)
                k = cv2.waitKey(500)
                if k == 27:    # Esc key to stop
                    exit()
