import os
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision.utils import save_image, make_grid

from common.utils import get_lr, make_dir


def train_model(
    generator,
    discriminator,
    train_generator,
    val_generator,
    learning_params,
    image_processing_params,
    save_dir,
    device='cpu',
    debug=False
):

    # tensorboard writer for tracking vars
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_runs'))

    # make an image dir for saving images
    image_dir = os.path.join(save_dir, 'val_images')
    make_dir(image_dir)

    training_loader = torch.utils.data.DataLoader(
        train_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
    )

    val_loader = torch.utils.data.DataLoader(
        val_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
    )

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()  # .to(device)
    criterion_pixelwise = torch.nn.L1Loss()  # .to(device)

    # Calculate output of image discriminator (PatchGAN)
    patch = (
        1,
        image_processing_params["dims"][0] // 2 ** 4,
        image_processing_params["dims"][1] // 2 ** 4,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=learning_params["lr"],
        betas=(learning_params["adam_b1"], learning_params["adam_b2"]),
        weight_decay=learning_params['adam_decay']
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_params["lr"],
        betas=(learning_params["adam_b1"], learning_params["adam_b2"]),
        weight_decay=learning_params['adam_decay']
    )
    G_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience'],
        verbose=True
    )
    D_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience'],
        verbose=True
    )

    # for tracking overall train time
    training_start_time = time.time()

    # for saving best models
    lowest_pixel_loss = np.inf

    if debug:
        plt.ion()
        LOSS_DATA = []
        VAL_LOSS_DATA = []
        VAL_LOSS_DATA_X = []
        fig, ax = plt.subplots(1, 4, figsize=(1, 1))
        ax0 = ax[0].imshow(np.zeros((128, 128)), vmin=0, vmax=1)
        ax[0].set_title("GAN Input (Sim TacTip)")
        ax1 = ax[1].imshow(np.zeros((128, 128)), vmin=0, vmax=1)
        ax[1].set_title("True Output (Real TacTip)")
        ax2 = ax[2].imshow(np.zeros((128, 128)), vmin=0, vmax=1)
        ax[2].set_title("GAN Output")
        ax3, = ax[3].plot([], [], linewidth=0.25)
        ax3val, = ax[3].plot([], [], linewidth=1, c="red")
        ax[3].set_xlabel("Training Step")
        ax[3].set_ylabel("Loss")
        ax[3].set_xlim((0, 80 * 250 * 2 * 2))
        # ax[3].set_ylim((0,0.25))
        ax[3].set_ylim((0, .99))
        ax[3].set_title("Pixel Loss")
        plt.suptitle("ok")
        frame_counter = 0

    # Main training loop
    with tqdm(total=learning_params['epochs']) as pbar:
        for epoch in range(1, learning_params['epochs'] + 1):

            # tracking epoch metrics
            epoch_loss_D = []
            epoch_loss_real = []
            epoch_loss_fake = []
            epoch_loss_G = []
            epoch_loss_GAN = []
            epoch_loss_pixel = []

            # Run training epoch
            for i, batch in enumerate(training_loader):

                input_images = Variable(batch["input"]).float().to(device)
                target_images = Variable(batch["target"]).float().to(device)

                # Adversarial ground truths
                valid = torch.tensor(np.ones((input_images.size(0), *patch), dtype=np.float32),
                                     requires_grad=False, device=device)
                fake = torch.tensor(np.zeros((input_images.size(0), *patch), dtype=np.float32),
                                    requires_grad=False, device=device)

                #  Train Generators
                optimizer_G.zero_grad()

                # GAN loss
                gen_images = generator(input_images, batch["shear"].to(device))
                pred_gen = discriminator(gen_images, input_images)
                loss_GAN = criterion_GAN(pred_gen, valid)

                # Generator losses
                loss_pixel = criterion_pixelwise(gen_images, target_images)
                ssim_L = loss_pixel
                loss_G = ssim_L

                loss_G.backward()

                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()

                # Real loss
                pred_real = discriminator(target_images, input_images)
                loss_real = criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = discriminator(gen_images.detach(), input_images)
                loss_fake = criterion_GAN(pred_fake, fake)

                # Total loss
                loss_disc = 0.5 * (loss_real + loss_fake)
                loss_D = loss_disc

                loss_D.backward()
                optimizer_D.step()

                # log batch losses
                epoch_loss_D.append(loss_D.item())
                epoch_loss_real.append(loss_real.item())
                epoch_loss_fake.append(loss_fake.item())
                epoch_loss_G.append(loss_G.item())
                epoch_loss_GAN.append(loss_GAN.item())
                epoch_loss_pixel.append(loss_pixel.item())

                if debug:
                    LOSS_DATA.append(loss_G.item())
                    if i >= 0:
                        ax0.set_data(input_images.cpu()[0,0,:,:])
                        ax1.set_data(target_images.cpu()[0,0,:,:])
                        ax2.set_data(gen_images.detach().cpu().numpy()[0,0,:,:])
                        plt.suptitle(f"GAN Training. Epoch {epoch}: Loss {np.round(np.mean(epoch_loss_GAN), 4)}")
                        ax3.set_data(range(len(LOSS_DATA)), LOSS_DATA)
                        fig.canvas.flush_events()
                        plt.pause(0.01)
                        frame_counter += 1

            # perform validation on single batch
            generator.eval()
            discriminator.eval()

            val_loss_pixel = []
            for i in range(learning_params['n_val_batches']):
                val_batch = next(iter(val_loader))
                val_input_images = Variable(val_batch["input"]).float().to(device)
                val_target_images = Variable(val_batch["target"]).float().to(device)
                val_gen_images = generator(val_input_images, val_batch["shear"].to(device))
                val_loss_pixel.append(criterion_pixelwise(val_gen_images, val_target_images).item())
                val_pix_loss = criterion_pixelwise(val_gen_images, val_target_images)#.item()
                val_loss_G = val_pix_loss

                val_loss_pixel.append(val_loss_G.item())
                if debug:
                    VAL_LOSS_DATA.append(np.mean(val_loss_pixel))
                    VAL_LOSS_DATA_X.append(len(LOSS_DATA))
                    ax3val.set_data(VAL_LOSS_DATA_X, VAL_LOSS_DATA)
                    fig.canvas.flush_events()
                    plt.pause(0.01)

            # save example images
            if epoch % learning_params['save_every'] == 0:
                val_gen_images = torch.clamp(val_gen_images, 0, 1)
                val_img_sample = torch.cat([
                    val_input_images.data[:learning_params['n_save_images'], ...],
                    val_gen_images.data[:learning_params['n_save_images'], ...],
                    val_target_images.data[:learning_params['n_save_images'], ...]],
                    -1,
                )
                grid = make_grid(val_img_sample, nrow=4, normalize=False, pad_value=1.0)
                name = f'{os.path.basename(os.path.normpath(save_dir))}/val_images/epoch_{epoch}'
                writer.add_image(name, grid)
                save_image(
                    val_img_sample,
                    os.path.join(image_dir, f"epoch_{epoch}.png"),
                    nrow=4, normalize=False, pad_value=1.0
                )

            generator.train()
            discriminator.train()

            #  print metrics
            print("")
            print("")
            print("Epoch: {}".format(epoch))
            print("Disc Loss:      {:.6f}".format(np.mean(epoch_loss_D)))
            print("Real Loss:      {:.6f}".format(np.mean(epoch_loss_real)))
            print("Fake Loss:      {:.6f}".format(np.mean(epoch_loss_fake)))
            print("Gen Loss:       {:.6f}".format(np.mean(epoch_loss_G)))
            print("GAN Loss:       {:.6f}".format(np.mean(epoch_loss_GAN)))
            print("Pixel Loss:     {:.6f}".format(np.mean(epoch_loss_pixel)))
            print("Val Pixel Loss: {:.6f}".format(np.mean(val_loss_pixel)))
            print("")

            # write vals to tensorboard
            writer.add_scalar('loss/discriminator', np.mean(epoch_loss_D), epoch)
            writer.add_scalar('loss/discriminator.real', np.mean(epoch_loss_real), epoch)
            writer.add_scalar('loss/discriminator.fake', np.mean(epoch_loss_fake), epoch)
            writer.add_scalar('loss/generator', np.mean(epoch_loss_G), epoch)
            writer.add_scalar('loss/gan', np.mean(epoch_loss_GAN), epoch)
            writer.add_scalar('loss/pixel', np.mean(epoch_loss_pixel), epoch)
            writer.add_scalar('loss/val_pixel', np.mean(val_loss_pixel), epoch)
            writer.add_scalar('learning_rate/generator', get_lr(optimizer_G), epoch)
            writer.add_scalar('learning_rate/discriminator', get_lr(optimizer_D), epoch)

            # track weights on tensorboard
            for name, weight in generator.named_parameters():
                full_name = f'{os.path.basename(os.path.normpath(save_dir))}/generator/{name}'
                try:
                    writer.add_histogram(full_name, weight, epoch)
                    writer.add_histogram(f'{full_name}.grad', weight.grad, epoch)
                except:
                    print(f"failed for {name}...")
            for name, weight in discriminator.named_parameters():
                full_name = f'{os.path.basename(os.path.normpath(save_dir))}/discriminator/{name}'
                try:
                    writer.add_histogram(full_name, weight, epoch)
                    writer.add_histogram(f'{full_name}.grad', weight.grad, epoch)
                except:
                    print(f"failed for {name}...")

            # save the model with lowest val loss
            if np.mean(val_loss_pixel) < lowest_pixel_loss:
                print('Saving Best Model')
                lowest_pixel_loss = np.mean(val_loss_pixel)
                torch.save(generator.state_dict(), os.path.join(save_dir, 'best_generator.pth'))
                torch.save(discriminator.state_dict(), os.path.join(save_dir, 'best_discriminator.pth'))

            # update lrs
            G_lr_scheduler.step(np.mean(epoch_loss_G))
            D_lr_scheduler.step(np.mean(epoch_loss_D))

            # update epoch progress bar
            pbar.update(1)

    # print total training time
    total_training_time = time.time() - training_start_time
    print("Training finished, took {:.6f}s".format(total_training_time))

    # save final model
    torch.save(generator.state_dict(), os.path.join(save_dir, 'final_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'final_discriminator.pth'))

if __name__ == "__main__":
    pass
