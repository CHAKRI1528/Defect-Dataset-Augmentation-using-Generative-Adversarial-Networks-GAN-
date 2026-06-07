"""
Training script for DCGAN model on defect images.
Implements adversarial training loop with loss tracking.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.dcgan import Generator, Discriminator, initialize_weights
from data.dataloader import get_dataloader


class GANTrainer:
    """Trainer class for DCGAN model."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create models
        self.generator = Generator(
            latent_dim=config['latent_dim'],
            img_channels=config['img_channels']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            img_channels=config['img_channels']
        ).to(self.device)
        
        # Initialize weights
        initialize_weights(self.generator)
        initialize_weights(self.discriminator)
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config['lr_g'],
            betas=(config['beta1'], config['beta2'])
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr_d'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(config['log_dir'])
        self.global_step = 0
        
        # Create output directory
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        g_losses = []
        d_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config['epochs']}")
        
        for real_images in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Real and fake labels
            real_label = torch.ones(batch_size, 1).to(self.device)
            fake_label = torch.zeros(batch_size, 1).to(self.device)
            
            # ============== Train Discriminator ==============
            self.optimizer_d.zero_grad()
            
            # Real images
            real_output = self.discriminator(real_images)
            d_real_loss = self.criterion(real_output, real_label)
            
            # Fake images
            z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
            fake_images = self.generator(z)
            fake_output = self.discriminator(fake_images.detach())
            d_fake_loss = self.criterion(fake_output, fake_label)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.optimizer_d.step()
            
            # ============== Train Generator ==============
            self.optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
            fake_images = self.generator(z)
            fake_output = self.discriminator(fake_images)
            
            g_loss = self.criterion(fake_output, real_label)
            g_loss.backward()
            self.optimizer_g.step()
            
            # Track losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': np.mean(d_losses[-10:]),
                'G_loss': np.mean(g_losses[-10:])
            })
            
            # Log to TensorBoard
            if self.global_step % 50 == 0:
                self.writer.add_scalar('Loss/discriminator', d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/generator', g_loss.item(), self.global_step)
            
            self.global_step += 1
        
        return np.mean(g_losses), np.mean(d_losses)
    
    def generate_samples(self, num_samples=16, epoch=0):
        """Generate sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['latent_dim']).to(self.device)
            fake_images = self.generator(z)
            
            # Denormalize
            fake_images = fake_images * 0.5 + 0.5
            fake_images = torch.clamp(fake_images, 0, 1)
        
        return fake_images.cpu()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        return checkpoint['epoch']
    
    def train(self, dataloader):
        """Train for specified number of epochs."""
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            g_loss, d_loss = self.train_epoch(dataloader, epoch)
            
            print(f"Epoch {epoch + 1}/{self.config['epochs']} - "
                  f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
            
            # Save checkpoint every save_interval epochs
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
                print(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Save sample images
            if (epoch + 1) % self.config['sample_interval'] == 0:
                self.save_sample_images(epoch)
        
        self.writer.close()
        print("Training completed!")
    
    def save_sample_images(self, epoch):
        """Save generated sample images."""
        samples = self.generate_samples(num_samples=16, epoch=epoch)
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        
        for idx, ax in enumerate(axes.flat):
            sample = samples[idx].permute(1, 2, 0).numpy()
            ax.imshow(sample)
            ax.axis('off')
        
        sample_dir = os.path.join(self.config['output_dir'], 'samples')
        Path(sample_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(os.path.join(sample_dir, f'samples_epoch_{epoch + 1}.png'), 
                   bbox_inches='tight', dpi=100)
        plt.close()


def main():
    """Main training function."""
    config = {
        'epochs': 100,
        'batch_size': 32,
        'latent_dim': 100,
        'img_channels': 3,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'img_size': 128,
        'output_dir': 'outputs',
        'log_dir': 'outputs/logs',
        'save_interval': 10,
        'sample_interval': 5,
    }
    
    # Get data loader
    dataloader = get_dataloader(
        image_dir='data/raw_defects',
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    # Create trainer and train
    trainer = GANTrainer(config)
    trainer.train(dataloader)
    
    # Save final model
    torch.save(trainer.generator.state_dict(), 'outputs/generator_final.pth')
    print("Final model saved!")


if __name__ == "__main__":
    main()
