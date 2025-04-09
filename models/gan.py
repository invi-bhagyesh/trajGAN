import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, image_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Initial dense layer
        self.fc = nn.Linear(latent_dim, 512 * (image_size // 8) * (image_size // 8))
        
        # Main generator blocks
        self.main = nn.Sequential(
            # Input is latent_dim x 1 x 1
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: 64 x 16 x 16
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final size: channels x image_size x image_size
        )

    def forward(self, z):
        # Reshape the input
        x = self.fc(z)
        x = x.view(-1, 512, self.image_size // 8, self.image_size // 8)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, image_size, channels):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is channels x image_size x image_size
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 64 x (image_size/2) x (image_size/2)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128 x (image_size/4) x (image_size/4)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 256 x (image_size/8) x (image_size/8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 512 x (image_size/16) x (image_size/16)
        )
        
        # Calculate the size of the flattened features
        self.fc_size = 512 * (image_size // 16) * (image_size // 16)
        self.fc = nn.Linear(self.fc_size, 1)
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, self.fc_size)
        return self.fc(x).view(-1, 1).squeeze(1) 