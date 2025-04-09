import torch
import torch.optim as optim
import yaml
import wandb
from pathlib import Path
from tqdm import tqdm

from models.gan import Generator, Discriminator
from losses import GANLoss
from dataset import get_dataloader
from utils.visualize import save_image_grid, log_metrics
from utils.latent_utils import sample_latent

def train_gan(config):
    # Initialize wandb
    wandb.init(
        project=config['logging']['project_name'],
        entity=config['logging']['entity'],
        config=config
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    generator = Generator(
        latent_dim=config['gan']['latent_dim'],
        image_size=config['gan']['image_size'],
        channels=config['gan']['channels']
    ).to(device)
    
    discriminator = Discriminator(
        image_size=config['gan']['image_size'],
        channels=config['gan']['channels']
    ).to(device)
    
    # Create optimizers
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config['gan']['lr_generator'],
        betas=(config['gan']['beta1'], config['gan']['beta2'])
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config['gan']['lr_discriminator'],
        betas=(config['gan']['beta1'], config['gan']['beta2'])
    )
    
    # Create loss functions
    gan_loss = GANLoss(device)
    
    # Create dataloader
    dataloader = get_dataloader(config)
    
    # Training loop
    for epoch in range(config['gan']['num_epochs']):
        generator.train()
        discriminator.train()
        
        g_losses = []
        d_losses = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["gan"]["num_epochs"]}')
        for batch_idx, real_images in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Sample random latent vectors
            z = sample_latent(batch_size, config['gan']['latent_dim'], device)
            
            # Compute discriminator loss
            d_loss = gan_loss.discriminator_loss(discriminator, generator, real_images, z)
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            # Sample new random latent vectors
            z = sample_latent(batch_size, config['gan']['latent_dim'], device)
            
            # Compute generator loss
            g_loss, fake_images = gan_loss.generator_loss(discriminator, generator, z)
            g_loss.backward()
            g_optimizer.step()
            
            # Store losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}'
            })
            
            # Log to wandb
            if batch_idx % config['logging']['log_interval'] == 0:
                log_metrics({
                    'g_loss': g_loss.item(),
                    'd_loss': d_loss.item()
                }, epoch * len(dataloader) + batch_idx)
                
                # Log sample images
                wandb.log({
                    'real_images': wandb.Image(real_images[:16]),
                    'fake_images': wandb.Image(fake_images[:16])
                })
        
        # Save models
        if (epoch + 1) % config['gan']['save_interval'] == 0:
            save_dir = Path(config['logging']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(generator.state_dict(), save_dir / f'generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), save_dir / f'discriminator_epoch_{epoch+1}.pth')
            
            # Save sample images
            save_image_grid(
                fake_images[:64],
                save_dir / f'samples_epoch_{epoch+1}.png',
                nrow=8
            )
    
    wandb.finish()

if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_gan(config) 