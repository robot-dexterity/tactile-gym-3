"""
python evaluate_model.py -r abb -m simple_cnn -t edge_2d
"""
import os
import itertools as it
import pandas as pd
from torch.autograd import Variable
import torch

from utils.utils import load_json_obj
from learning.supervised.image_to_image.supervised.models import create_model
from learning.supervised.image_to_image.supervised.image_generator import ImageDataGenerator
from learning.supervised.image_to_image.utils.utils_plots import RegressionPlotter

from learning.supervised.image_to_feature.learning.setup_training import csv_row_to_label
from learning.supervised.image_to_feature.utils.label_encoder import LabelEncoder
from learning.supervised.image_to_feature.utils.parse_args import parse_args


def evaluate_model(
    model,
    label_encoder,
    generator,
    learning_params,
    error_plotter,
    device='cpu'
):

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    target_label_names = list(filter(None, label_encoder.target_label_names))
    pred_df = pd.DataFrame(columns=target_label_names)
    targ_df = pd.DataFrame(columns=target_label_names)

    for _, batch in enumerate(loader):

        # get inputs
        inputs, targ_dict = batch['inputs'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)

        # count correct for accuracy metric
        pred_dict = label_encoder.decode_label(outputs)

        # append predictions and labels to dataframes
        batch_pred_df = pd.DataFrame.from_dict(pred_dict)
        batch_targ_df = pd.DataFrame.from_dict(targ_dict)
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

    # reset indices to be 0 -> test set size
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)

    print("Metrics")
    metrics = label_encoder.calc_metrics(pred_df, targ_df)
    err_df, acc_df = metrics['err'], metrics['acc']
    print("evaluated_acc:")
    print(acc_df[[*target_label_names, 'overall_acc']].mean())
    print("evaluated_err:")
    print(err_df[target_label_names].mean())

    # plot full error graph
    error_plotter.name = 'error_plot_best'
    error_plotter.final_plot(
        pred_df, targ_df, metrics
    )

def evaluate_mdn_model(
    model,
    label_encoder,
    generator,
    learning_params,
    error_plotter,
    device='cpu'
):

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    target_label_names = list(filter(None, label_encoder.target_label_names))
    pred_df = pd.DataFrame(columns=target_label_names)
    targ_df = pd.DataFrame(columns=target_label_names)

    for _, batch in enumerate(loader):

        # get inputs
        inputs, targ_dict = batch['inputs'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)[1].squeeze()

        # count correct for accuracy metric
        pred_dict = label_encoder.decode_label(outputs)

        # append predictions and labels to dataframes
        batch_pred_df = pd.DataFrame.from_dict(pred_dict)
        batch_targ_df = pd.DataFrame.from_dict(targ_dict)
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

    # reset indices to be 0 -> test set size
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)

    print("Metrics")
    metrics = label_encoder.calc_metrics(pred_df, targ_df)
    err_df, acc_df = metrics['err'], metrics['acc']
    print("evaluated_acc:")
    print(acc_df[[*target_label_names, 'overall_acc']].mean())
    print("evaluated_err:")
    print(err_df[target_label_names].mean())

    # plot full error graph
    error_plotter.name = 'error_plot_best'
    error_plotter.final_plot(
        pred_df, targ_df, metrics
    )

if __name__ == "__main__":
    pass
