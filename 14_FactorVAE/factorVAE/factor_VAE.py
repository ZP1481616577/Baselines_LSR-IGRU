import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from factorVAE.feature_extractor import FeatureExtractor
from factorVAE.factor_encoder import FactorEncoder
from factorVAE.factor_decoder import FactorDecoder
from factorVAE.factor_predictor import FactorPredictor

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, mu_pred, sigma_pred, mu_true, sigma_true):
        # 创建正态分布
        dist_pred = Normal(mu_pred, sigma_pred)
        dist_true = Normal(mu_true, sigma_true)

        # 从正态分布中抽取样本
        y_pred_tensor = dist_pred.sample()
        y_true_tensor = dist_true.sample()

        # 计算相关系数
        mean_pred = torch.mean(y_pred_tensor)
        mean_true = torch.mean(y_true_tensor)
        cov_pred_true = torch.mean((y_pred_tensor - mean_pred) * (y_true_tensor - mean_true))
        var_pred = torch.mean((y_pred_tensor - mean_pred)**2)
        var_true = torch.mean((y_true_tensor - mean_true)**2)
        rho = 2 * cov_pred_true / (var_pred + var_true + (mean_pred - mean_true)**2)

        # 计算标准差
        sigma_pred_mean = torch.mean(sigma_pred)
        sigma_true_mean = torch.mean(sigma_true)

        # 计算 CCC 损失
        ccc_loss = 1 - rho + (sigma_pred_mean / sigma_true_mean + sigma_true_mean / sigma_pred_mean) / 2

        return ccc_loss

class FactorVAE(nn.Module):
    def __init__(
        self,
        characteristic_size,
        stock_size,
        latent_size,
        factor_size,
        time_span,
        gru_input_size,
        hidden_size=64,
        alpha_h_size=64,
    ):
        super(FactorVAE, self).__init__()

        self.characteristic_size = characteristic_size
        self.stock_size = stock_size
        self.latent_size = latent_size

        self.feature_extractor = FeatureExtractor(
            time_span=time_span,
            characteristic_size=characteristic_size,
            latent_size=latent_size,
            stock_size=stock_size,
            gru_input_size=gru_input_size,
        )

        self.factor_encoder = FactorEncoder(
            latent_size=latent_size,
            stock_size=stock_size,
            factor_size=factor_size,
            hidden_size=hidden_size,
        )

        self.factor_decoder = FactorDecoder(
            latent_size=latent_size,
            factor_size=factor_size,
            stock_size=stock_size,
            alpha_h_size=alpha_h_size,
            hidden_size=hidden_size,
        )

        self.factor_predictor = FactorPredictor(
            latent_size=latent_size, factor_size=factor_size, stock_size=stock_size
        )

    def run_model(self, characteristics, future_returns, gamma=1):
        latent_features = self.feature_extractor(characteristics)
        # (batch_size, stock_size, latent_size)

        mu_post, sigma_post = self.factor_encoder(latent_features, future_returns)
        # (batch_size, factor_size)

        m_encoder = Normal(mu_post, sigma_post)
        factors_post = m_encoder.sample()

        # (batch_size, factor_size, 1)

        reconstruct_returns, mu_alpha, sigma_alpha, beta = self.factor_decoder(
            factors_post, latent_features
        )

        mu_dec, sigma_dec = self.get_decoder_distribution(
            mu_alpha, sigma_alpha, mu_post, sigma_post, beta
        )

        loss_negloglike = (
            Normal(mu_dec, sigma_dec).log_prob(future_returns.unsqueeze(-1)).sum()
        )

        loss_negloglike = loss_negloglike * (-1 / (self.stock_size * latent_features.shape[0]))
        # latent_features.shape[0] is the batch_size

        mu_prior, sigma_prior = self.factor_predictor(latent_features)
        m_predictor = Normal(mu_prior, sigma_prior)

        loss_KL = kl_divergence(m_encoder, m_predictor).sum()

        loss = loss_negloglike + gamma * loss_KL
        #loss_fn = CCCLoss()
        #ccc_loss = loss_fn(mu_prior, sigma_prior, mu_post, sigma_post).sum()
        #loss = loss_negloglike + gamma * ccc_loss

        return loss

    def prediction(self, characteristics):
        with torch.no_grad():

            latent_features = self.feature_extractor(characteristics)

            mu_prior, sigma_prior = self.factor_predictor(latent_features)

            m_prior = Normal(mu_prior, sigma_prior)
            factor_prior = m_prior.sample()

            pred_returns, mu_alpha, sigma_alpha, beta = self.factor_decoder(
                factor_prior, latent_features
            )

            mu_dec, sigma_dec = self.get_decoder_distribution(
                mu_alpha, sigma_alpha, mu_prior, sigma_prior, beta
            )

        return pred_returns, mu_dec, sigma_dec

    def get_decoder_distribution(
        self, mu_alpha, sigma_alpha, mu_factor, sigma_factor, beta
    ):
        # print(mu_alpha.shape, mu_factor.shape, sigma_factor.shape, beta.shape)
        mu_dec = mu_alpha + torch.bmm(beta, mu_factor)

        sigma_dec = torch.sqrt(
            torch.square(sigma_alpha)
            + torch.bmm(torch.square(beta), torch.square(sigma_factor))
        )

        return mu_dec, sigma_dec
