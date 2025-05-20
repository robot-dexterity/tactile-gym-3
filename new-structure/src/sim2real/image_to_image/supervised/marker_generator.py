import numpy as np
import os
import pandas as pd
import torch

from tactile_learning.supervised.image_generator import numpy_collate


class MarkerDataGenerator(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dirs,
        csv_row_to_label,
        num_markers
    ):
        assert isinstance(data_dirs, list), "data_dirs must be a list"

        self._csv_row_to_label = csv_row_to_label

        # load labels and select by marker number
        label_df = self.load_labels(data_dirs)
        self._label_df = label_df[label_df['num_markers']==num_markers]

    def load_labels(self, data_dirs):

        # combine and add column for data dir
        df_list = []
        for data_dir in data_dirs:
            df = pd.read_csv(os.path.join(data_dir, 'targets_markers.csv'))
            df['keypoints_dir'] = os.path.join(data_dir, 'processed_markers')
            df_list.append(df)

        return pd.concat(df_list)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        row = self._label_df.iloc[index]
        keypoints_filename = os.path.join(row['keypoints_dir'], row['markers_file'])
        keypoints = np.load(keypoints_filename)

        # get label
        target = self._csv_row_to_label(row)
        sample = {'inputs': keypoints, 'labels': target}

        return sample


def demo_marker_generation(
    data_dirs,
    csv_row_to_label,
    learning_params,
    num_markers
):

    # Configure dataloaders
    generator = MarkerDataGenerator(
        data_dirs,
        csv_row_to_label,
        num_markers
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
        collate_fn=numpy_collate
    )

    # iterate through
    for _, sample_batched in enumerate(loader, 0):

        # shape = (batch, n_frames, width, height)
        inputs = sample_batched['inputs']
        labels = sample_batched['labels']

        for i in range(inputs.shape[0]):
            for key, item in labels.items():
                print(key, ': ', item[i])

            print('')
            print('Extracted Keypoints: ', inputs.shape)
            print('')
