import os
import warnings
import time
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tactile_learning.utils.utils_learning import get_lr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def simple_train_model(
    prediction_mode,
    model,
    label_encoder,
    train_generator,
    val_generator,
    learning_params,
    save_dir,
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

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # define optimizer and loss
    if prediction_mode == 'classification':
        loss = nn.CrossEntropyLoss()
    elif prediction_mode == 'regression':
        loss = nn.MSELoss()
    else:
        raise Warning("Incorrect prediction mode provided, falling back on MSEloss")
        loss = nn.MSELoss()

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

    def run_epoch(loader, n_batches, training=True):

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

        return epoch_batch_loss, epoch_batch_acc

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

            train_epoch_loss, train_epoch_acc = run_epoch(
                train_loader, n_train_batches, training=True
            )

            # ========= Validation =========
            model.eval()
            val_epoch_loss, val_epoch_acc = run_epoch(
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

            # track weights on tensorboard
            try:
                for name, weight in model.named_parameters():
                    full_name = f'{os.path.basename(os.path.normpath(save_dir))}/{name}'
                    writer.add_histogram(full_name, weight, epoch)
                    writer.add_histogram(f'{full_name}.grad', weight.grad, epoch)
            except ValueError:
                warnings.warn("Unable to save weights/gradients to tensorboard.")

            # save the model with lowest val loss
            if np.mean(val_epoch_loss) < lowest_val_loss:
                lowest_val_loss = np.mean(val_epoch_loss)

                print('Saving Best Model')
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'best_model.pth')
                )

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
