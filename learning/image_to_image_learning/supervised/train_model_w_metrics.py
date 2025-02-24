import os
import warnings
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from tactile_learning.utils.utils_learning import get_lr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_model_w_metrics(
    prediction_mode,
    model,
    label_encoder,
    train_generator,
    val_generator,
    learning_params,
    save_dir,
    error_plotter=None,
    calculate_train_metrics=False,
    device='cpu'
):
    # tensorboard writer for tracking vars
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_runs'))

    train_loader = torch.utils.data.DataLoader(
        train_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    val_loader = torch.utils.data.DataLoader(
        val_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # define optimizer and loss
    if prediction_mode == 'classification':
        loss = nn.CrossEntropyLoss()
    elif prediction_mode == 'regression':
        loss = nn.MSELoss()
    else:
        raise Warning("Incorrect prediction mode provided, falling back on MSEloss")
        loss = nn.MSELoss()

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_params['lr'],
        betas=(learning_params["adam_b1"], learning_params["adam_b2"]),
        weight_decay=learning_params['adam_decay']
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience'],
        verbose=True
    )

    def run_epoch(loader, n_batches_per_epoch, training=True):

        epoch_batch_loss = []
        epoch_batch_acc = []

        # complete dateframe of predictions and targets
        pred_df = pd.DataFrame()
        targ_df = pd.DataFrame()

        for i, batch in enumerate(loader):

            # run set number of batches per epoch
            if n_batches_per_epoch is not None and i >= n_batches_per_epoch:
                break

            # get inputs
            inputs, labels_dict = batch['inputs'], batch['labels']

            # wrap them in a Variable object
            inputs = Variable(inputs).float().to(device)

            # get labels
            labels = label_encoder.encode_label(labels_dict)

            # set the parameter gradients to zero
            if training:
                optimizer.zero_grad()

            # forward pass, backward pass, optimize
            outputs = model(inputs)
            loss_size = loss(outputs, labels)
            epoch_batch_loss.append(loss_size.item())

            if prediction_mode == 'classification':
                epoch_batch_acc.append((outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item())
            else:
                epoch_batch_acc.append(0.0)

            if training:
                loss_size.backward()
                optimizer.step()

            # calculate metrics that are useful to keep track of during training
            # this can slow learning noticably, particularly if train metrics are tracked
            if not training or calculate_train_metrics:

                # decode predictions into label
                predictions_dict = label_encoder.decode_label(outputs)

                # append predictions and labels to dataframes
                batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
                batch_targ_df = pd.DataFrame.from_dict(labels_dict)
                pred_df = pd.concat([pred_df, batch_pred_df])
                targ_df = pd.concat([targ_df, batch_targ_df])

        # reset indices to be 0 -> test set size
        pred_df = pred_df.reset_index(drop=True).fillna(0.0)
        targ_df = targ_df.reset_index(drop=True).fillna(0.0)
        return epoch_batch_loss, epoch_batch_acc, pred_df, targ_df

    # get time for printing
    training_start_time = time.time()

    # for tracking metrics across training
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # for saving best model
    lowest_val_loss = np.inf

    with tqdm(total=learning_params['epochs']) as pbar:

        # Main training loop
        for epoch in range(1, learning_params['epochs'] + 1):

            # ========= Training =========
            train_epoch_loss, train_epoch_acc, train_pred_df, train_targ_df = run_epoch(
                train_loader, learning_params['n_train_batches_per_epoch'], training=True
            )

            # ========= Validation =========
            model.eval()
            val_epoch_loss, val_epoch_acc, val_pred_df, val_targ_df = run_epoch(
                val_loader, learning_params['n_val_batches_per_epoch'], training=False
            )
            model.train()

            # append loss and acc
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            # print metrics
            print("")
            print("")
            print("Epoch: {}".format(epoch))
            print("Train Loss: {:.6f}".format(np.mean(train_epoch_loss)))
            print("Train Acc:  {:.6f}".format(np.mean(train_epoch_acc)))
            print("Val Loss:   {:.6f}".format(np.mean(val_epoch_loss)))
            print("Val Acc:    {:.6f}".format(np.mean(val_epoch_acc)))
            print("")

            # write vals to tensorboard
            writer.add_scalar('loss/train', np.mean(train_epoch_loss), epoch)
            writer.add_scalar('loss/val', np.mean(val_epoch_loss), epoch)
            writer.add_scalar('accuracy/train', np.mean(train_epoch_acc), epoch)
            writer.add_scalar('accuracy/val', np.mean(val_epoch_acc), epoch)
            writer.add_scalar('learning_rate', get_lr(optimizer), epoch)

            # calculate task metrics
            if calculate_train_metrics:
                train_metrics = label_encoder.calc_metrics(train_pred_df, train_targ_df)
            val_metrics = label_encoder.calc_metrics(val_pred_df, val_targ_df)

            # print task metrics
            if calculate_train_metrics:
                print("Train Metrics")
                label_encoder.print_metrics(train_metrics)
                print("")
            print("Validation Metrics")
            label_encoder.print_metrics(val_metrics)

            # write task_metrics to tensorboard
            if calculate_train_metrics:
                label_encoder.write_metrics(writer, train_metrics, epoch, mode='train')
            label_encoder.write_metrics(writer, val_metrics, epoch, mode='val')

            # track weights on tensorboard
            try:
                for name, weight in model.named_parameters():
                    full_name = f'{os.path.basename(os.path.normpath(save_dir))}/{name}'
                    writer.add_histogram(full_name, weight, epoch)
                    writer.add_histogram(f'{full_name}.grad', weight.grad, epoch)
            except ValueError:
                warnings.warn("Unable to save weights/gradients to tensorboard.")

            # update plots
            if error_plotter:
                if not error_plotter.final_only:
                    error_plotter.update(
                        val_pred_df, val_targ_df, val_metrics
                    )

            # save the model with lowest val loss
            if np.mean(val_epoch_loss) < lowest_val_loss:
                lowest_val_loss = np.mean(val_epoch_loss)

                print('Saving Best Model')
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'best_model.pth')
                )

                # save loss and acc, save val
                save_vars = [train_loss, val_loss, train_acc, val_acc]
                with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'bw') as f:
                    pickle.dump(save_vars, f)

            # decay the lr
            lr_scheduler.step(np.mean(val_epoch_loss))

            # update epoch progress bar
            pbar.update(1)

    total_training_time = time.time() - training_start_time
    print("Training finished, took {:.6f}s".format(total_training_time))

    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, 'final_model.pth')
    )

    return lowest_val_loss, total_training_time


if __name__ == "__main__":
    pass
