# training.py

import torch
import torch.nn as nn
import numpy as np
import logging

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for inputs, targets in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        predicted_parameters = model(inputs)
        predicted_parameters = predicted_parameters.view_as(targets)
        
        # Compute loss
        loss = criterion(predicted_parameters, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted_parameters = model(inputs)
            predicted_parameters = predicted_parameters.view_as(targets)
            
            # Compute loss
            loss = criterion(predicted_parameters, targets)
            running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total trainable parameters: {total_params}')
    return total_params