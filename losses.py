import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss:
    def __init__(self, device):
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_labels(self, batch_size, is_real):
        if is_real:
            labels = torch.ones(batch_size, device=self.device)
        else:
            labels = torch.zeros(batch_size, device=self.device)
        return labels
    
    def generator_loss(self, discriminator, generator, z):
        batch_size = z.size(0)
        fake_images = generator(z)
        fake_pred = discriminator(fake_images)
        
        # Generator wants discriminator to think fake images are real
        labels = self.get_labels(batch_size, is_real=True)
        loss = self.criterion(fake_pred, labels)
        
        return loss, fake_images
    
    def discriminator_loss(self, discriminator, generator, real_images, z):
        batch_size = real_images.size(0)
        
        # Real images should be classified as real
        real_pred = discriminator(real_images)
        real_labels = self.get_labels(batch_size, is_real=True)
        real_loss = self.criterion(real_pred, real_labels)
        
        # Generated images should be classified as fake
        fake_images = generator(z).detach()  # Detach to avoid generator update
        fake_pred = discriminator(fake_images)
        fake_labels = self.get_labels(batch_size, is_real=False)
        fake_loss = self.criterion(fake_pred, fake_labels)
        
        # Total discriminator loss
        loss = real_loss + fake_loss
        
        return loss

class TrajectoryLoss:
    def __init__(self, device, smoothness_weight=1.0, adversarial_weight=0.1):
        self.device = device
        self.smoothness_weight = smoothness_weight
        self.adversarial_weight = adversarial_weight
        self.mse = nn.MSELoss()
        
    def smoothness_loss(self, trajectory):
        """
        Compute smoothness loss between consecutive latent vectors
        trajectory shape: [batch_size, path_length + 1, latent_dim]
        """
        # Compute pairwise differences between consecutive latents
        diffs = trajectory[:, 1:] - trajectory[:, :-1]
        # Return mean squared difference
        return self.mse(diffs, torch.zeros_like(diffs))
    
    def adversarial_loss(self, discriminator, generator, trajectory):
        """
        Compute adversarial loss using frozen discriminator
        """
        batch_size = trajectory.size(0)
        total_loss = 0
        
        # Generate images for each latent in trajectory
        for t in range(trajectory.size(1)):
            images = generator(trajectory[:, t])
            pred = discriminator(images)
            # Encourage realistic images
            total_loss += F.binary_cross_entropy_with_logits(
                pred, 
                torch.ones(batch_size, device=self.device)
            )
        
        return total_loss / trajectory.size(1)
    
    def __call__(self, trajectory, discriminator=None, generator=None):
        loss = self.smoothness_weight * self.smoothness_loss(trajectory)
        
        if discriminator is not None and generator is not None:
            adv_loss = self.adversarial_loss(discriminator, generator, trajectory)
            loss += self.adversarial_weight * adv_loss
            
        return loss 