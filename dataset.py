import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

class CelebADataset(Dataset):
    def __init__(self, root_dir, image_size=64, transform=None):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.glob("*.jpg"))
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_dataloader(config, split='train'):
    """
    Create DataLoader for CelebA dataset
    """
    transform = transforms.Compose([
        transforms.Resize(config['dataset']['image_size']),
        transforms.CenterCrop(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CelebADataset(
        root_dir=config['dataset']['root'],
        image_size=config['dataset']['image_size'],
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['gan']['batch_size'] if split == 'train' else config['trajectory']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    
    return dataloader 