# models/beta_vae.py
import torch
import torch.nn as nn

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=10, beta=4.0, seq_length=435):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * (seq_length//4), latent_dim)
        self.fc_logvar = nn.Linear(256 * (seq_length//4), latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * (seq_length//4)),
            nn.Unflatten(1, (256, seq_length//4)),
            nn.ConvTranspose1d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, 3, padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld