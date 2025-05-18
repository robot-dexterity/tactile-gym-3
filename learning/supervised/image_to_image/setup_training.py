import os
import shutil

from common.utils import save_json_obj


def setup_learning(save_dir=None):

    # Parameters
    learning_params = {
        'seed': 42,
        'batch_size': 32,
        'epochs': 50,
        'n_val_batches': 10,
        'lr': 2e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 0.0,
        'adam_b1': 0.5,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 1,
        'sample_interval': 5,
        'lambda_gan': 1.0,
        'lambda_pixel': 100.0,
        'n_save_images': 16,
        'save_every': 5,
    }

    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))

    return learning_params


def setup_model_image(save_dir=None):

    image_processing_params = {
        #'dims': (128, 128),
        'dims': (256, 256),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': None,  # (0.025, 0.025),
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    model_image_params = {
        'image_processing': image_processing_params,
        'augmentation': augmentation_params
    }

    if save_dir:
        save_json_obj(model_image_params, os.path.join(save_dir, 'model_image_params'))

    return model_image_params


def setup_model(model_type, save_dir):

    if 'pix2pix' in model_type:
        model_params = {
            'model_type': model_type,
            'generator_kwargs': {
                'in_channels': 1,
                'out_channels': 1,
            },
            'discriminator_kwargs': {
                'in_channels': 1,
                'disc_block': [64, 128, 256, 512],
                'normalise_disc': [False, True, True, True],
            }
        }

        if model_type == 'pix2pix_64':
            model_params['generator_kwargs'].update({
                'unet_down': [64, 128, 256, 512, 512, 512],
                'dropout_down': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                'normalise_down': [False, True, True, True, True, False],
                'unet_up': [0, 512, 512, 256, 128, 64],
                'dropout_up': [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            })

        elif model_type == 'pix2pix_128':
            model_params['generator_kwargs'].update({
                'unet_down': [64, 128, 256, 512, 512, 512, 512],
                'dropout_down': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
                'normalise_down': [False, True, True, True, True, True, False],
                'unet_up': [0, 512, 512, 512, 256, 128, 64],
                'dropout_up': [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            })

        elif model_type == 'pix2pix_256':
            model_params['generator_kwargs'].update({
                'unet_down': [64, 128, 256, 512, 512, 512, 512, 512],
                'dropout_down': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
                'normalise_down': [False, True, True, True, True, True, True, False],
                'unet_up': [0, 512, 512, 512, 512, 256, 128, 64],
                'dropout_up': [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            }) #..todo: kipp i edited this
        # elif model_type == 'pix2pix_256':
        #     model_params['generator_kwargs'].update({
        #         'unet_down': [64, 128, 256, 512, 1024, 2048, 2048],
        #         'dropout_down': [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        #         'normalise_down': [False, True, True, True, True, True, False],
        #         'unet_up': [0, 2048, 1024, 512, 256, 128, 64],
        #         'dropout_up': [0.5, 0.5, 0.5, 0.5, 0.5, 0., 0.],
        #     })
        # elif model_type == 'pix2pix_256':
        #     model_params['generator_kwargs'].update({
        #         'unet_down': [64, 128, 256, 512, 1024, 2048, 2048, 2048],
        #         'dropout_down': [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        #         'normalise_down': [False, True, True, True, True, True, True, False],
        #         'unet_up': [0, 2048, 2048, 1024, 512, 256, 128, 64],
        #         'dropout_up': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0., 0.],
        #     })
        else:
            print('dimension of pix2pix not recognized')

    else:
        print('model not recognized')

    # save parameters
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_training(model_type, data_dirs, save_dir=None):
    learning_params = setup_learning(save_dir)
    model_image_params = setup_model_image(save_dir)
    model_params = setup_model(model_type, save_dir)

    is_processed = os.path.isdir(os.path.join(data_dirs[0], 'processed_images'))

    # retain data parameters
    if save_dir:
        shutil.copy(os.path.join(data_dirs[0], 'env_params.json'), save_dir)
        if is_processed:
            shutil.copy(os.path.join(data_dirs[0], 'processed_image_params.json'), save_dir)
        else:
            shutil.copy(os.path.join(data_dirs[0], 'sensor_image_params.json'), save_dir)

    return learning_params, model_params, model_image_params