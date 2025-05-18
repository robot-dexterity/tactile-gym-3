import os
import numpy as np
import torch
from pytorch_model_summary import summary
from vit_pytorch.vit import ViT
import torch.nn as nn
from torch.distributions.normal import Normal

from learning.supervised.image_to_feature.cnn.models import weights_init_normal, CNN, NatureCNN, ResNet, ResidualBlock


def create_model(
    in_dim,
    in_channels,
    out_dim,
    model_params,
    saved_model_dir=None,
    display_model=True,
    device='cpu'
):
    # for mdn head, base layers have model_out_dim
    if '_mdn' in model_params['model_type']:
        mdn_out_dim = out_dim
        out_dim = model_params['mdn_kwargs']['model_out_dim']

    if 'simple_cnn' in model_params['model_type']:
        model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif 'posenet_cnn' in model_params['model_type']:
        model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif 'nature_cnn' in model_params['model_type']:
        model = NatureCNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif 'resnet' in model_params['model_type']:
        model = ResNet(
            ResidualBlock,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs'],
        ).to(device)

    elif 'vit' in model_params['model_type']:
        model = ViT(
            image_size=in_dim[0],
            channels=in_channels,
            num_classes=out_dim,
            **model_params['model_kwargs']
        ).to(device)

    else:
        raise ValueError('Incorrect model_type specified:  %s' % (model_params['model_type'],))

    if '_mdn' in model_params['model_type']:
    #     model = MDN_AC(
    #         model=model,
    #         out_dim=mdn_out_dim,
    #         **model_params['mdn_kwargs']
    #     ).to(device)
        model = MDN_JL(
            model=model,
            out_dim=mdn_out_dim,
            **model_params['mdn_kwargs']
        ).to(device)

    if saved_model_dir is not None:
        print("LOADING MODEL")
        model.load_state_dict(torch.load(os.path.join(
            saved_model_dir, 'best_model.pth'), map_location='cpu')
        )

    if display_model:
        dummy_input = torch.zeros((1, in_channels, *in_dim)).to(device)
        print(summary(
            model,
            dummy_input,
            show_input=True
        ))

    return model


def softbound(x, x_min, x_max):
    return (torch.log1p(torch.exp(-torch.abs(x - x_min)))
            - torch.log1p(torch.exp(-torch.abs(x - x_max)))) \
        + torch.maximum(x, x_min) + torch.minimum(x, x_max) - x


class MDN_AC(nn.Module):
    """
    Implementation of Mixture Density Networks in Pytorch from

    https://github.com/tonyduan/mixture-density-network

    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """

    def __init__(
        self,
        model,
        out_dim,
        model_out_dim,
        hidden_dims,
        activation,
        n_mdn_components,
        noise_type='diagonal',
        fixed_noise_level=None
    ):
        super(MDN_AC, self).__init__()

        assert (fixed_noise_level is not None) == (noise_type == 'fixed')

        num_sigma_channels = {
            'diagonal': out_dim * n_mdn_components,
            'isotropic': n_mdn_components,
            'isotropic_across_clusters': 1,
            'fixed': 0,
        }[noise_type]

        self.out_dim, self.n_components = out_dim, n_mdn_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level

        # init pi and normal heads
        pi_network_modules = [model]
        normal_network_modules = [model]

        # add the first layer
        pi_network_modules.append(nn.ReLU())
        pi_network_modules.append(nn.Linear(model_out_dim, hidden_dims[0]))
        normal_network_modules.append(nn.ReLU())
        normal_network_modules.append(nn.Linear(model_out_dim, hidden_dims[0]))
        if activation == 'relu':
            pi_network_modules.append(nn.ReLU())
            normal_network_modules.append(nn.ReLU())
        elif activation == 'elu':
            pi_network_modules.append(nn.ELU())
            normal_network_modules.append(nn.ELU())

        # add the remaining hidden layers
        for idx in range(len(hidden_dims) - 1):
            pi_network_modules.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            normal_network_modules.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            if activation == 'relu':
                pi_network_modules.append(nn.ReLU())
                normal_network_modules.append(nn.ReLU())
            elif activation == 'elu':
                pi_network_modules.append(nn.ELU())
                normal_network_modules.append(nn.ELU())

        # add the final layers
        pi_network_modules.append(nn.Linear(hidden_dims[-1], n_mdn_components))
        normal_network_modules.append(nn.Linear(hidden_dims[-1], out_dim * n_mdn_components + num_sigma_channels))

        self.pi_network = nn.Sequential(*pi_network_modules)
        self.normal_network = nn.Sequential(*normal_network_modules)

    def forward(self, x, eps=1e-6):
        """
        Returns
        -------
        log_pi: (bsz, n_components)
        mu: (bsz, n_components, dim_out)
        sigma: (bsz, n_components, dim_out)
        """

        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.out_dim * self.n_components]
        sigma = normal_params[..., self.out_dim * self.n_components:]

        if self.noise_type == 'diagonal':
            sigma = torch.exp(sigma + eps)
        if self.noise_type == 'isotropic':
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type == 'isotropic_across_clusters':
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type == 'fixed':
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)

        mu = mu.reshape(-1, self.n_components, self.out_dim)
        sigma = sigma.reshape(-1, self.n_components, self.out_dim)

        return log_pi, mu, sigma

    def loss(self, x, y):
        """
        Calculates negative log_likelihood.
        """
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def predict(self, x, deterministic=True):
        """
        Samples from the predicted distribution.
        """
        log_pi, mu, sigma = self.forward(x)
        pi = torch.exp(log_pi)
        pred_mean = torch.sum(pi.unsqueeze(dim=-1) * mu, dim=1)
        pred_stddev = torch.sqrt(torch.sum(pi.unsqueeze(dim=-1) * (sigma**2 + mu**2), dim=1) - pred_mean**2).squeeze()
        return pred_mean, pred_stddev

    def sample(self, x, deterministic=True):
        """
        Samples from the predicted distribution.
        """
        log_pi, mu, sigma = self.forward(x)
        pi = torch.exp(log_pi)
        pred_mean = torch.sum(pi.unsqueeze(dim=-1) * mu, dim=1)
        pred_stddev = torch.sqrt(torch.sum(pi.unsqueeze(dim=-1) * (sigma**2 + mu**2), dim=1) - pred_mean**2).squeeze()
        if deterministic:
            return pred_mean
        else:
            pi_distribution = Normal(pred_mean, pred_stddev)
            return pi_distribution.rsample()


class MDN_JL(nn.Module):
    def __init__(
        self,
        model,
        out_dim,
        model_out_dim,
        n_mdn_components,
        pi_dropout=0.1,
        mu_dropout=0.1,
        sigma_inv_dropout=0.1,
        mu_min=-np.inf,
        mu_max=np.inf,
        sigma_inv_min=1e-6,
        sigma_inv_max=1e6,
    ):
        super(MDN_JL, self).__init__()

        self.out_dim, self.n_mdn_components = torch.tensor(out_dim), torch.tensor(n_mdn_components)
        self.mu_min, self.mu_max = torch.tensor(np.resize(mu_min, out_dim)), torch.tensor(np.resize(mu_max, out_dim))
        self.sigma_inv_min, self.sigma_inv_max = \
            torch.tensor(np.resize(sigma_inv_min, out_dim)), torch.tensor(np.resize(sigma_inv_max, out_dim))

        self.base_model = model

        # mixture weights
        pi_modules = []
        pi_modules.append(nn.Dropout(pi_dropout))
        pi_modules.append(nn.Linear(model_out_dim, n_mdn_components))
        pi_modules.append(nn.Softmax(dim=-1))
        self.pi_head = nn.Sequential(*pi_modules)

        # component means and (inverse) stdevs
        self.mu_heads, self.sigma_inv_heads = [], []
        mu_dropout, sigma_inv_dropout = np.resize(mu_dropout, out_dim), np.resize(sigma_inv_dropout, out_dim)
        for i in range(out_dim):
            mu_modules_i = []
            mu_modules_i.append(nn.Dropout(mu_dropout[i]))
            mu_modules_i.append(nn.Linear(model_out_dim, n_mdn_components))
            self.mu_heads.append(nn.Sequential(*mu_modules_i))

            sigma_inv_modules_i = []
            sigma_inv_modules_i.append(nn.Dropout(sigma_inv_dropout[i]))
            sigma_inv_modules_i.append(nn.Linear(model_out_dim, n_mdn_components))
            self.sigma_inv_heads.append(nn.Sequential(*sigma_inv_modules_i))
        self.mu_heads, self.sigma_inv_heads = nn.ModuleList(self.mu_heads), nn.ModuleList(self.sigma_inv_heads)

    def _apply(self, fn):
        super(MDN_JL, self)._apply(fn)
        self.out_dim, self.n_mdn_components = fn(self.out_dim), fn(self.n_mdn_components)
        self.mu_min, self.mu_max, self.sigma_inv_min, self.sigma_inv_max = \
            fn(self.mu_min), fn(self.mu_max), fn(self.sigma_inv_min), fn(self.sigma_inv_max)
        return self

    def forward(self, x):
        x = self.base_model(x)
        pi = self.pi_head(x)
        mu, sigma_inv = [], []
        for i in range(self.out_dim):
            mu.append(softbound(self.mu_heads[i](x), self.mu_min[i], self.mu_max[i]))
            sigma_inv.append(softbound(self.sigma_inv_heads[i](x), self.sigma_inv_min[i], self.sigma_inv_max[i]))
        mu, sigma_inv = torch.stack(mu, dim=2), torch.stack(sigma_inv, dim=2)
        return pi, mu, sigma_inv

    def loss(self, x, y):
        pi, mu, sigma_inv = self.forward(x)
        squared_err = torch.sum(torch.square((torch.unsqueeze(y, dim=1) - mu) * sigma_inv), dim=2)
        log_pdf_comp = - (squared_err / 2) - (self.out_dim * np.log(2 * np.pi) / 2) \
            + torch.sum(torch.log(sigma_inv), dim=2)
        log_pdf = torch.logsumexp(torch.log(pi) + log_pdf_comp, dim=-1)
        nll = -torch.mean(log_pdf)
        return nll

    def predict(self, x):
        pi, mu, sigma_inv = self.forward(x)
        pred_mean = torch.sum(pi.unsqueeze(dim=-1) * mu, dim=1)
        pred_stddev = torch.sqrt(torch.sum(pi.unsqueeze(dim=-1) * (1/(sigma_inv**2) + mu**2), dim=1)
                                 - pred_mean**2).squeeze()
        return pred_mean, pred_stddev
