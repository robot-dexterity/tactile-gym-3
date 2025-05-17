import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter

from common.utils import get_lr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_mdn_model(
    prediction_mode,
    model,
    label_encoder,
    train_generator,
    val_generator,
    learning_params,
    save_dir,
    device='cpu',
    error_plotter=None
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

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # define optimizer
    optimizer = optim.Adam(model.parameters())

    if 'cyclic_base_lr' in learning_params:
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=learning_params['cyclic_base_lr'],
            max_lr=learning_params['cyclic_max_lr'],
            step_size_up=learning_params['cyclic_half_period'] * n_train_batches,
            mode=learning_params['cyclic_mode'],
            cycle_momentum=False
        )

    elif 'lr' in learning_params:
        optimizer.lr = learning_params['lr']
        optimizer.betas = (learning_params.get("adam_b1", None), learning_params.get("adam_b2", None))
        optimizer.weight_decay = learning_params.get('adam_decay', None)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=learning_params.get('lr_factor', None),
            patience=learning_params.get('lr_patience', None),
            verbose=True
        )

    def run_epoch(loader, n_batches, training=True):

        pred_df = pd.DataFrame()
        targ_df = pd.DataFrame()

        epoch_batch_loss = []
        epoch_batch_acc = []

        for batch in loader:

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
            loss_size = model.loss(inputs, labels).mean()
            epoch_batch_loss.append(loss_size.item())
            epoch_batch_acc.append(0.0)

            if training:
                loss_size.backward()
                optimizer.step()
                if 'cyclic_base_lr' in learning_params:
                    lr_scheduler.step()

            if not training:
                # decode predictions into label
                outputs = model(inputs)[1].squeeze()
                predictions_dict = label_encoder.decode_label(outputs)

                # append predictions and labels to dataframes
                batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
                batch_targ_df = pd.DataFrame.from_dict(labels_dict)
                pred_df = pd.concat([pred_df, batch_pred_df])
                targ_df = pd.concat([targ_df, batch_targ_df])

        pred_df = pred_df.reset_index(drop=True).fillna(0.0)
        targ_df = targ_df.reset_index(drop=True).fillna(0.0)
        return epoch_batch_loss, epoch_batch_acc, pred_df, targ_df

    # for tracking overall train time
    training_start_time = time.time()

    # for tracking metrics across training
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # for saving best model
    lowest_val_loss = np.inf

    with tqdm(total=learning_params['epochs']) as pbar:

        # Main training loop
        for epoch in range(1, learning_params['epochs'] + 1):

            train_epoch_loss, train_epoch_acc, train_pred_df, train_targ_df = run_epoch(
                train_loader, n_train_batches, training=True
            )

            # ========= Validation =========
            model.eval()
            val_epoch_loss, val_epoch_acc, val_pred_df, val_targ_df = run_epoch(
                val_loader, n_val_batches, training=False
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

            # train_metrics = label_encoder.calc_metrics(train_pred_df, train_targ_df)
            val_metrics = label_encoder.calc_metrics(val_pred_df, val_targ_df)
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

            # decay the lr
            if 'lr' in learning_params:
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