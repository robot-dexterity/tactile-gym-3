import os
import torch
import torch.nn as nn
from pytorch_model_summary import summary

from learning.supervised.image_to_image.pix2pix.setup_model import Discriminator
from learning.supervised.image_to_image.pix2pix.setup_model import UNetDown, UNetUp  


def setup_model(
    in_dim,
    model_params,
    saved_model_dir=None,
    device='cpu'
):

    if 'shpix2pix' in model_params['model_type']:
        generator = GeneratorUNet(**model_params['generator_kwargs']).to(device)
        discriminator = Discriminator(**model_params['discriminator_kwargs']).to(device)
    else:
        raise ValueError('Incorrect model_type specified:  %s' % (model_params['model_type'],))

    if saved_model_dir is not None:
        generator.load_state_dict(torch.load(os.path.join(
            saved_model_dir, 'best_generator.pth'), map_location='cpu')
        )
        discriminator.load_state_dict(torch.load(os.path.join(
            saved_model_dir, 'best_discriminator.pth'), map_location='cpu')
        )

    print(summary(
        generator,
        torch.zeros((1, 1, *in_dim)).to(device),
        torch.zeros((1, 6)).to(device),
        show_input=True
    ))
    print(summary(
        discriminator,
        torch.zeros((1, 1, *in_dim)).to(device),
        torch.zeros((1, 1, *in_dim)).to(device),
        show_input=True
    ))

    return generator, discriminator


class GeneratorUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        unet_down=[],
        dropout_down=[],
        normalise_down=[],
        unet_up=[],
        dropout_up=[],
    ):
        super(GeneratorUNet, self).__init__()

        assert len(unet_down) > 0, "unet_down must contain values"
        assert len(unet_up) > 0, "unet_up must contain values"
        assert len(unet_down) == len(dropout_down), "unet_down must be same len as dropout_down"
        assert len(unet_up) == len(dropout_up), "unet_up must be same len as dropout_up"
        assert len(unet_down) == len(normalise_down), "unet_down must be same len as normalise_down"

        # add input channels to unet_down for simplicity
        unet_down.insert(0, in_channels)

        # add the remaining unet_down layers by iterating through params
        unet_down_modules = []
        for idx in range(len(unet_down) - 1):
            unet_down_modules.append(
                UNetDown(
                    unet_down[idx],
                    unet_down[idx+1],
                    normalize=normalise_down[idx],
                    dropout=dropout_down[idx],
                )
            )
        self.unet_down = nn.Sequential(*unet_down_modules)

        # add the unet_up layers by iterating through params
        unet_up_modules = []
        for idx in range(len(unet_up)-1):
            unet_up_modules.append(
                UNetUp(
                    unet_up[idx] + unet_down[-(idx+1)],
                    unet_up[idx+1],
                    dropout=dropout_up[idx],
                )
            )
        self.unet_up = nn.Sequential(*unet_up_modules)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(unet_down[1] + unet_up[-1], out_channels, 4, padding=1),
            nn.Tanh(),
        )
        # ..todo: we need to get 512 automaticaly, not hardcode it. This way shpix2pix will be invariant to image size
        # self.fc = nn.Linear(512+6,512) # pix2pix256
        self.fc = nn.Linear(2048+6,2048) # pix2pix128
        # self.fc = nn.Linear(8192+6,8192) # pix2pix64

    def forward(self, x, shear):
        down_outs = []
        down_outs.append(x)
        for i, layer in enumerate(self.unet_down.children()):
            x = layer(down_outs[i])
            down_outs.append(x)
        xshape = x.shape
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, shear), dim=1)
        try:
            x = self.fc(x)
        except:
            raise(BaseException("..todo: The FC linear layer is the wrong size - change it in init."))
        x = x.unsqueeze(2).unsqueeze(3)
        x = torch.reshape(x, xshape)
        x = nn.ReLU(inplace=True)(x)
        for i, layer in enumerate(self.unet_up.children()):
            x = layer(x, down_outs[-(i+2)])
        return self.final(x)
