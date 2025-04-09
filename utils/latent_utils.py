import torch
import numpy as np

def sample_latent(batch_size, latent_dim, device):
    """Sample random latent vectors from normal distribution"""
    return torch.randn(batch_size, latent_dim, device=device)

def interpolate_latents(z1, z2, num_steps):
    """
    Linear interpolation between two latent vectors
    z1, z2: [batch_size, latent_dim]
    """
    alpha = torch.linspace(0, 1, num_steps, device=z1.device)
    alpha = alpha.view(-1, 1)  # [num_steps, 1]
    
    # Expand z1 and z2 for broadcasting
    z1 = z1.unsqueeze(0)  # [1, batch_size, latent_dim]
    z2 = z2.unsqueeze(0)  # [1, batch_size, latent_dim]
    
    # Interpolate
    interpolated = (1 - alpha) * z1 + alpha * z2  # [num_steps, batch_size, latent_dim]
    return interpolated

def spherical_interpolation(z1, z2, num_steps):
    """
    Spherical linear interpolation (slerp) between two latent vectors
    z1, z2: [batch_size, latent_dim]
    """
    # Normalize the latent vectors
    z1_norm = z1 / z1.norm(dim=1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=1, keepdim=True)
    
    # Compute the angle between vectors
    dot_product = (z1_norm * z2_norm).sum(dim=1)
    angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    
    # Generate interpolation steps
    t = torch.linspace(0, 1, num_steps, device=z1.device)
    t = t.view(-1, 1, 1)  # [num_steps, 1, 1]
    
    # Expand for broadcasting
    z1 = z1.unsqueeze(0)  # [1, batch_size, latent_dim]
    z2 = z2.unsqueeze(0)  # [1, batch_size, latent_dim]
    angle = angle.unsqueeze(0).unsqueeze(-1)  # [1, batch_size, 1]
    
    # Compute slerp
    sin_angle = torch.sin(angle)
    interpolated = (torch.sin((1 - t) * angle) / sin_angle) * z1 + \
                  (torch.sin(t * angle) / sin_angle) * z2
    
    return interpolated

def add_noise_to_latent(z, noise_scale=0.1):
    """Add Gaussian noise to latent vectors"""
    noise = torch.randn_like(z) * noise_scale
    return z + noise 