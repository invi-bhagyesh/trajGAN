import torch
import torch.optim as optim
import yaml
import wandb
from pathlib import Path
from tqdm import tqdm

from models.gan import Generator, Discriminator
from models.trajectory_net import TrajectoryNet
from losses import TrajectoryLoss
from dataset import get_dataloader
from utils.visualize import visualize_trajectory, log_metrics
from utils.latent_utils import sample_latent

def train_trajectory(config):
    # Initialize wandb
    wandb.init(
        project=config['logging']['project_name'],
        entity=config['logging']['entity'],
        config=config
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained GAN
    generator = Generator(
        latent_dim=config['gan']['latent_dim'],
        image_size=config['gan']['image_size'],
        channels=config['gan']['channels']
    ).to(device)
    
    discriminator = Discriminator(
        image_size=config['gan']['image_size'],
        channels=config['gan']['channels']
    ).to(device)
    
    # Load pretrained weights
    save_dir = Path(config['logging']['save_dir'])
    generator.load_state_dict(torch.load(save_dir / 'generator_final.pth'))
    discriminator.load_state_dict(torch.load(save_dir / 'discriminator_final.pth'))
    
    # Freeze GAN parameters
    for param in generator.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False
    
    # Create trajectory network
    trajectory_net = TrajectoryNet(
        latent_dim=config['gan']['latent_dim'],
        path_length=config['trajectory']['path_length'],
        hidden_dim=config['trajectory']['hidden_dim'],
        num_layers=config['trajectory']['num_layers']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        trajectory_net.parameters(),
        lr=config['trajectory']['lr']
    )
    
    # Create loss function
    trajectory_loss = TrajectoryLoss(
        device,
        smoothness_weight=config['trajectory']['smoothness_weight'],
        adversarial_weight=config['trajectory']['adversarial_weight']
    )
    
    # Create dataloader
    dataloader = get_dataloader(config, split='train')
    
    # Training loop
    for epoch in range(config['trajectory']['num_epochs']):
        trajectory_net.train()
        
        losses = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["trajectory"]["num_epochs"]}')
        for batch_idx, _ in enumerate(progress_bar):
            batch_size = config['trajectory']['batch_size']
            
            # Sample initial latent vectors
            z0 = sample_latent(batch_size, config['gan']['latent_dim'], device)
            
            # Generate trajectory
            trajectory = trajectory_net(z0)
            
            # Compute loss
            loss = trajectory_loss(trajectory, discriminator, generator)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store loss
            losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if batch_idx % config['logging']['log_interval'] == 0:
                log_metrics({'loss': loss.item()}, epoch * len(dataloader) + batch_idx)
                
                # Visualize trajectory
                visualize_trajectory(
                    generator,
                    trajectory[:4],  # Visualize first 4 samples
                    wandb_log=True
                )
        
        # Save model
        if (epoch + 1) % config['trajectory']['save_interval'] == 0:
            save_dir = Path(config['logging']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(
                trajectory_net.state_dict(),
                save_dir / f'trajectory_net_epoch_{epoch+1}.pth'
            )
            
            # Save sample trajectories
            with torch.no_grad():
                z0 = sample_latent(4, config['gan']['latent_dim'], device)
                trajectory = trajectory_net(z0)
                visualize_trajectory(
                    generator,
                    trajectory,
                    save_path=save_dir / f'trajectories_epoch_{epoch+1}.png'
                )
    
    wandb.finish()

if __name__ == '__main__':
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_trajectory(config) 