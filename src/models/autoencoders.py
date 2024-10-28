import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Define the VAE class
class SimpleVAE(nn.Module):
    def __init__(self, hp):
        super(SimpleVAE, self).__init__()
        self.hp = hp
        logger.info("Initializing model layers...")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=hp["in"], out_channels=16, kernel_size=hp["kernel_size"]),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=hp["kernel_size"]),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=hp["kernel_size"]),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=hp["kernel_size"]),
            nn.LeakyReLU(),
        )

        # Mapping to latent mean and variance
        self.mean_map = nn.Sequential(
            nn.Linear(16 * (hp["time_series_length"] - 4 * (hp["kernel_size"] - 1)), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, hp["embedding_dimension"])
        )
        self.std_map = nn.Sequential(
            nn.Linear(16 * (hp["time_series_length"] - 4 * (hp["kernel_size"] - 1)), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, hp["embedding_dimension"])
        )

        # Decoder
        self.linear2 = nn.Sequential(
            nn.Linear(hp["embedding_dimension"], 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16 * (hp["time_series_length"] - 4 * (hp["kernel_size"] - 1))),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=hp["kernel_size"]),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=hp["kernel_size"]),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=hp["kernel_size"]),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=hp["in"], kernel_size=hp["kernel_size"]),
        )

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, X):
        pre_code = self.encoder(X)
        B, C, L = pre_code.shape
        flattened = pre_code.view(B, C * L)

        # Latent space
        mu = self.mean_map(flattened)
        log_var = self.std_map(flattened)
        code = self.sample(mu, log_var)

        # Decode
        post_code = self.linear2(code)
        post_code_reshaped = post_code.view(B, C, L)
        X_hat = self.decoder(post_code_reshaped)

        return X_hat, code, mu, log_var

    @staticmethod
    def vae_loss(x_hat, x, mu, log_var, alpha, gamma=0):
        BCE = F.mse_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (x.shape[0] * x.shape[1])
        SSL = F.mse_loss(x_hat[:, :, -1], x[:, :, -1])
        return BCE + alpha * KLD + gamma * SSL, BCE, KLD

