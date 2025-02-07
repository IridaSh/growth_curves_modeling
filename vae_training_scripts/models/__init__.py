# models/__init__.py
from .shallow_cnn_vae import VAE
from .deep_cnn_vae import DeepCNNVAE
from .residual_cnn_vae import ResidualCNNVAE
from .vq_vae import VQVAE
from .transformer_vae import TransformerVAE



__all__ = ['VAE', 'DeepCNNVAE', 'ResidualCNNVAE', 'VQVAE', 'TransformerVAE']