import os
import torch
import torch.nn as nn
from pytorch_model_summary import summary


def create_model(
    in_dim,
    model_params,
    saved_model_dir=None,
    device='cpu'
):

    if 'pix2pix' in model_params['model_type']:
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
        show_input=True
    ))
    print(summary(
        discriminator,
        torch.zeros((1, 1, *in_dim)).to(device),
        torch.zeros((1, 1, *in_dim)).to(device),
        show_input=True
    ))

    return generator, discriminator


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        if normalize:
            layers = [
                nn.utils.spectral_norm(
                    nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)
                )
            ]
        else:
            layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]

        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False)
            ),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


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

    def forward(self, x):
        down_outs = []
        down_outs.append(x)
        for i, layer in enumerate(self.unet_down.children()):
            x = layer(down_outs[i])
            down_outs.append(x)

        for i, layer in enumerate(self.unet_up.children()):
            x = layer(x, down_outs[-(i+2)])

        return self.final(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, disc_block=[], normalise_disc=[]):
        super(Discriminator, self).__init__()

        assert len(disc_block) > 0, "disc_block must contain values"
        assert len(disc_block) == len(normalise_disc), "disc_block must be same len as normalise_disc"

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            if normalization:
                layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            else:
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        cnn_modules = []

        # add first cnn layer
        cnn_modules.extend(
            discriminator_block(
                in_channels * 2,
                disc_block[0],
                normalization=normalise_disc[0]
            )
        )

        # add remaining layers
        for idx in range(len(disc_block) - 1):
            cnn_modules.extend(
                discriminator_block(
                    disc_block[idx],
                    disc_block[idx+1],
                    normalization=normalise_disc[idx+1]
                )
            )
        self.cnn = nn.Sequential(*cnn_modules)

        self.disc = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(disc_block[-1], 1, 4, padding=1, bias=False),
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        cnn_out = self.cnn(img_input)
        disc_out = self.disc(cnn_out)
        return disc_out
