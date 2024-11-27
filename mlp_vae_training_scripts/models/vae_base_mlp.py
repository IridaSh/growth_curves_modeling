import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = torch.clamp(out, min=0, max=1)  # Force the output to be within 0 & 1.
        return out

class CombinedModel(nn.Module):
    def __init__(self, vae_model, mlp_model):
        super(CombinedModel, self).__init__()
        self.mlp_model = mlp_model
        self.vae_model = vae_model

    def forward(self, x):
        x = x.unsqueeze(1) 
        with torch.no_grad():
            mean, logvar = self.vae_model.encoder(x)
            latent_variables = self.vae_model.reparameterize(mean, logvar)
        predicted_parameters = self.mlp_model(latent_variables)
        return predicted_parameters