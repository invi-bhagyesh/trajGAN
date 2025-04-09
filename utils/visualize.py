import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wandb

def save_image_grid(images, save_path, nrow=8):
    """Save a grid of images"""
    vutils.save_image(images, save_path, nrow=nrow, normalize=True)

def visualize_trajectory(generator, trajectory, save_path=None, wandb_log=False):
    """
    Visualize a trajectory of latent vectors as a sequence of images
    trajectory shape: [batch_size, path_length + 1, latent_dim]
    """
    batch_size = trajectory.size(0)
    path_length = trajectory.size(1)
    
    # Generate images for each latent in trajectory
    all_images = []
    for t in range(path_length):
        images = generator(trajectory[:, t])
        all_images.append(images)
    
    # Stack images horizontally for each sample in batch
    for b in range(min(batch_size, 4)):  # Visualize up to 4 samples
        sample_images = torch.stack([img[b] for img in all_images], dim=0)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image_grid(sample_images, save_path / f'trajectory_{b}.png', nrow=path_length)
        
        if wandb_log:
            wandb.log({
                f"trajectory_{b}": wandb.Image(sample_images, caption=f"Sample {b}")
            })

def plot_losses(train_losses, val_losses, title, save_path=None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def log_metrics(metrics, step, prefix=''):
    """Log metrics to wandb"""
    wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    wandb.log(wandb_metrics, step=step) 