# models/__init__.py
from .shallow_cnn_vae import VAE
from .deep_cnn_vae import DeepCNNVAE
from .residual_cnn_vae import ResidualCNNVAE


__all__ = ['VAE', 'DeepCNNVAE', 'ResidualCNNVAE']