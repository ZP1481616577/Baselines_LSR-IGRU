import torch
import torch.nn as nn

from factorVAE.basic_net import MLP


class FactorPredictor(nn.Module):
    def __init__(self, latent_size, factor_size, stock_size):
        super(FactorPredictor, self).__init__()

        self.multi_head_attention = nn.MultiheadAttention(latent_size, factor_size)

        self.distribution_network_mu = MLP(
            input_size=stock_size * latent_size, output_size=factor_size, hidden_size=64
        )

        self.distribution_network_sigma = MLP(
            input_size=stock_size * latent_size,
            output_size=factor_size,
            hidden_size=64,
            out_activation=nn.Softplus(),
        )

    def forward(self, latent_features):

        h = self.multi_head_attention(
            latent_features, latent_features, latent_features
        )[0]

        h = h.reshape(h.shape[0], -1) # concatenate

        mu_prior = self.distribution_network_mu(h).unsqueeze(-1)
        sigma_prior = self.distribution_network_sigma(h).unsqueeze(-1)
        # (bs, factor_size, 1)
        return mu_prior, sigma_prior
