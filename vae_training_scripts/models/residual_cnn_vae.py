import torch.nn as nn
import torch.nn.functional as F
from .base_vae import BaseVAE

class ResidualCNNEncoder(nn.Module):
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(ResidualCNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Residual connection
        self.residual = nn.Conv1d(1, 128, kernel_size=1)  # Match dimensions

        self.final_encoder = nn.Sequential(
            nn.Conv1d(128, latent_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(latent_channel),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(latent_channel * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(latent_channel * seq_length, latent_dim)

    def forward(self, x):
        residual = self.residual(x)
        x = self.encoder(x)
        x = x + residual  # Add residual connection
        x = self.final_encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    
class ResidualCNNDecoder(nn.Module):
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(ResidualCNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length

        self.fc = nn.Linear(latent_dim, latent_channel * seq_length)
        self.initial_decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channel, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Residual connection
        self.residual = nn.ConvTranspose1d(latent_channel, 128, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.Softplus(),  # Ensure strictly positive outputs
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.latent_channel, self.seq_length)
        residual = self.residual(x)
        x = self.initial_decoder(x)
        x = x + residual  # Add residual connection
        x = self.decoder(x)
        return x

class ResidualCNNVAE(BaseVAE):
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(ResidualCNNVAE, self).__init__(latent_dim, latent_channel, seq_length)
        self.encoder = ResidualCNNEncoder(latent_dim, latent_channel, seq_length)
        self.decoder = ResidualCNNDecoder(latent_dim, latent_channel, seq_length)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar