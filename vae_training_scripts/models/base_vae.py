import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseVAE(nn.Module):
    """Base Variational Autoencoder."""
    def __init__(self, latent_dim, latent_channel, seq_length):
        super(BaseVAE, self).__init__()
        self.latent_dim = latent_dim
        self.latent_channel = latent_channel
        self.seq_length = seq_length

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")