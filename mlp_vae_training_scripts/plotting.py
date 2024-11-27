# plotting.py

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import r2_score
import logging

def plot_loss_curves(train_losses, test_losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(test_losses, label='Testing Loss')
    plt.title('Training & Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f'Loss curves saved to {save_path}')

def plot_parameter_predictions(subset_train_data, output_train, subset_test_data, output_test, parameter_names, save_dir):
    for col in range(output_train.shape[1]):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Adjusted figsize for better visibility

        # Training data plot
        axs[0].scatter(subset_train_data[:, col], output_train[:, col], s=5, color='blue', alpha=0.5)
        min_val = min(np.min(subset_train_data[:, col]), np.min(output_train[:, col]))
        max_val = max(np.max(subset_train_data[:, col]), np.max(output_train[:, col]))
        axs[0].plot([min_val, max_val], [min_val, max_val], 'r')  # y=x line
        axs[0].set_xlim(min_val, max_val)
        axs[0].set_ylim(min_val, max_val)
        axs[0].set_aspect('equal', adjustable='box')
        axs[0].set_xlabel('Original')
        axs[0].set_ylabel('Predicted')
        axs[0].set_title(f'Train: {parameter_names[col]}')
        r2_train = r2_score(subset_train_data[:, col], output_train[:, col])
        axs[0].text(0.05, 0.95, f'R² = {r2_train:.3f}', transform=axs[0].transAxes, verticalalignment='top')

        # Testing data plot
        axs[1].scatter(subset_test_data[:, col], output_test[:, col], s=5, color='green', alpha=0.5)
        min_val = min(np.min(subset_test_data[:, col]), np.min(output_test[:, col]))
        max_val = max(np.max(subset_test_data[:, col]), np.max(output_test[:, col]))
        axs[1].plot([min_val, max_val], [min_val, max_val], 'r')  # y=x line
        axs[1].set_xlim(min_val, max_val)
        axs[1].set_ylim(min_val, max_val)
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].set_xlabel('Original')
        axs[1].set_ylabel('Predicted')
        axs[1].set_title(f'Test: {parameter_names[col]}')
        r2_test = r2_score(subset_test_data[:, col], output_test[:, col])
        axs[1].text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=axs[1].transAxes, verticalalignment='top')

        plt.tight_layout()
        # Save the figure
        plot_filename = f'{parameter_names[col]}_prediction.png'
        plt.savefig(os.path.join(save_dir, plot_filename))
        plt.close()
        logging.info(f'Parameter prediction plot saved to {os.path.join(save_dir, plot_filename)}')