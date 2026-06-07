"""
DCGAN architecture for defect image generation.
Implements generator and discriminator networks.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network that transforms noise into synthetic defect images."""
    
    def __init__(self, latent_dim=100, img_channels=3):
        """
        Args:
            latent_dim: Dimension of the input noise vector
            img_channels: Number of output image channels (3 for RGB)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Initial fully connected layer
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Transposed convolution layers (deconvolution)
        self.layers = nn.Sequential(
            # Input: (512, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Output: (256, 16, 16)
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (128, 32, 32)
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (64, 64, 64)
            
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: (img_channels, 128, 128)
        )
    
    def forward(self, z):
        """
        Generate image from noise vector.
        
        Args:
            z: Noise vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated image of shape (batch_size, img_channels, 128, 128)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 512, 8, 8)
        x = self.layers(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network that classifies images as real or generated."""
    
    def __init__(self, img_channels=3):
        """
        Args:
            img_channels: Number of input image channels (3 for RGB)
        """
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            # Input: (img_channels, 128, 128)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (64, 64, 64)
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (128, 32, 32)
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (256, 16, 16)
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (512, 8, 8)
            
            nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # Output: (1, 1, 1)
        )
    
    def forward(self, img):
        """
        Classify image as real or generated.
        
        Args:
            img: Image of shape (batch_size, img_channels, 128, 128)
            
        Returns:
            Classification score of shape (batch_size, 1)
        """
        x = self.layers(img)
        x = x.view(x.size(0), -1)
        return x


def initialize_weights(model):
    """Initialize model weights with normal distribution."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
