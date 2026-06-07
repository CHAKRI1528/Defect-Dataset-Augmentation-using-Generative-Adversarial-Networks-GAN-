"""
Utility functions for the defect GAN project.
Includes visualization, metrics, and helper functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def visualize_batch(images, title="Batch of Images", num_cols=4):
    """
    Visualize a batch of images.
    
    Args:
        images: Tensor of shape (batch, channels, height, width)
        title: Title for the plot
        num_cols: Number of columns in grid
        
    Returns:
        matplotlib figure
    """
    batch_size = images.shape[0]
    num_rows = (batch_size + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(num_rows, num_cols)
    
    for idx in range(batch_size):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        
        if isinstance(images, torch.Tensor):
            img = images[idx].cpu().detach()
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
            else:
                img = img.squeeze().numpy()
                img = np.clip(img, 0, 1)
        else:
            img = images[idx]
        
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(batch_size, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def calculate_gradient_penalty(discriminator, real_images, fake_images, device, lambda_gp=10):
    """
    Calculate gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator network
        real_images: Real image batch
        fake_images: Fake image batch
        device: torch device
        lambda_gp: Gradient penalty coefficient
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_images.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    
    fake_labels = torch.ones(batch_size, 1, requires_grad=False).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=fake_labels,
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def sample_images_before_after(generator, discriminator, num_samples=8, device='cpu'):
    """
    Generate samples and evaluate them.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        num_samples: Number of samples to generate
        device: torch device
        
    Returns:
        Generated images, discriminator scores
    """
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, 100).to(device)
        images = generator(z)
        scores = discriminator(images)
    
    return images.cpu(), scores.cpu()


def save_training_state(generator, discriminator, optimizer_g, optimizer_d, 
                       epoch, output_dir='outputs'):
    """
    Save complete training state.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        epoch: Current epoch
        output_dir: Output directory
    """
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
    }
    
    filepath = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
    torch.save(checkpoint, filepath)
    return filepath


def load_training_state(checkpoint_path, generator, discriminator, 
                       optimizer_g, optimizer_d, device):
    """
    Load complete training state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        generator: Generator network
        discriminator: Discriminator network
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        device: torch device
        
    Returns:
        Epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state'])
    discriminator.load_state_dict(checkpoint['discriminator_state'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
    
    return checkpoint['epoch']


def denormalize(images, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Denormalize images from [-1, 1] to [0, 1].
    
    Args:
        images: Tensor of shape (batch, 3, H, W)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized images in [0, 1]
    """
    if isinstance(images, torch.Tensor):
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
        images = images * std + mean
    else:
        images = images * np.array(std).reshape(1, 3, 1, 1) + np.array(mean).reshape(1, 3, 1, 1)
    
    return torch.clamp(images, 0, 1) if isinstance(images, torch.Tensor) else np.clip(images, 0, 1)


def tensor_to_pil(tensor):
    """
    Convert torch tensor to PIL Image.
    
    Args:
        tensor: Image tensor (C, H, W) in [0, 1]
        
    Returns:
        PIL Image
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach()
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        else:
            tensor = tensor.squeeze()
        array = (tensor.numpy() * 255).astype(np.uint8)
    else:
        array = (tensor * 255).astype(np.uint8)
    
    if len(array.shape) == 3:
        return Image.fromarray(array, 'RGB')
    else:
        return Image.fromarray(array, 'L')
