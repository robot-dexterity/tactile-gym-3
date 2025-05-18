import os
import torch
import torch.nn as nn
from pytorch_model_summary import summary


def create_model(
    in_dim,
    in_channels,
    out_dim,
    model_params,
    saved_model_dir=None,
    display_model=True,
    device='cpu'
):
    if model_params['model_type'] in ['fcn']:
        model = FCN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    if saved_model_dir is not None:
        print("LOADING MODEL")
        model.load_state_dict(torch.load(os.path.join(
            saved_model_dir, 'best_model.pth'), map_location='cpu')
        )

    if display_model:
        dummy_input = torch.zeros((1, in_dim)).to(device)

    return model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class FCN(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        fc_layers=[128, 128],
        activation='relu',
        dropout=0.0,
        apply_batchnorm=False,
    ):
        super(FCN, self).__init__()

        assert len(fc_layers) > 0, "fc_layers must contain values"

        fc_modules = []

        # add first layer
        fc_modules.append(nn.Linear(in_dim, fc_layers[0]))
        if apply_batchnorm:
            fc_modules.append(nn.BatchNorm1d(fc_layers[0]))
        if activation == 'relu':
            fc_modules.append(nn.ReLU())
        elif activation == 'elu':
            fc_modules.append(nn.ELU())

        # add remaining layers
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            if apply_batchnorm:
                fc_modules.append(nn.BatchNorm1d(fc_layers[idx + 1]))
            if activation == 'relu':
                fc_modules.append(nn.ReLU())
            elif activation == 'elu':
                fc_modules.append(nn.ELU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

