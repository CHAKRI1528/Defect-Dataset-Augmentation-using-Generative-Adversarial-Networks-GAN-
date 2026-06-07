"""
Data loading utilities for defect images.
Handles image preprocessing and batch creation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DefectDataset(Dataset):
    """PyTorch Dataset for defect images."""
    
    def __init__(self, image_dir, img_size=128, transform=None):
        """
        Args:
            image_dir: Directory containing defect images
            img_size: Target image size
            transform: Optional image transforms
        """
        self.image_dir = image_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Load and transform image."""
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_dataloader(image_dir, batch_size=32, img_size=128, shuffle=True, num_workers=0):
    """
    Create DataLoader for defect images.
    
    Args:
        image_dir: Directory with training images
        batch_size: Batch size
        img_size: Image size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        
    Returns:
        PyTorch DataLoader
    """
    dataset = DefectDataset(image_dir, img_size=img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
