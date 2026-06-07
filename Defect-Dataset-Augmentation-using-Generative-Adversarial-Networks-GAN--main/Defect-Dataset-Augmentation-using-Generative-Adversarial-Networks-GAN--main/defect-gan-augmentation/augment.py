"""
Inference script for generating augmented defect images.
Use trained generator to create new synthetic defects.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from models.dcgan import Generator


class DefectAugmentor:
    """Generate augmented defect images using trained GAN."""
    
    def __init__(self, model_path, latent_dim=100, img_channels=3, device=None):
        """
        Args:
            model_path: Path to trained generator model
            latent_dim: Dimension of latent space (must match training config)
            img_channels: Number of image channels
            device: torch device (cuda or cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # Load generator
        self.generator = Generator(latent_dim=latent_dim, img_channels=img_channels)
        self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
        self.generator.to(self.device)
        self.generator.eval()
        
        print(f"Loaded generator from {model_path}")
    
    def generate_images(self, num_images=10, seed=None):
        """
        Generate synthetic defect images.
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of generated images (num_images, 3, 128, 128)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim).to(self.device)
            fake_images = self.generator(z)
            
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = fake_images * 0.5 + 0.5
            fake_images = torch.clamp(fake_images, 0, 1)
        
        return fake_images.cpu()
    
    def save_images(self, images, output_dir='outputs/augmented_defects', prefix='augmented'):
        """
        Save generated images to disk.
        
        Args:
            images: Tensor of images
            output_dir: Directory to save images
            prefix: Prefix for filenames
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for idx, image in enumerate(images):
            # Convert tensor to PIL Image
            image_np = image.permute(1, 2, 0).numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            
            filename = f"{prefix}_{idx:04d}.png"
            filepath = Path(output_dir) / filename
            image_pil.save(filepath)
        
        print(f"Saved {len(images)} augmented images to {output_dir}")
    
    def visualize_images(self, images, title="Generated Defect Images"):
        """
        Display generated images in grid.
        
        Args:
            images: Tensor of images
            title: Title for the plot
        """
        num_images = len(images)
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, image in enumerate(images):
            image_np = image.permute(1, 2, 0).numpy()
            axes[idx].imshow(image_np)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def augment_dataset(self, original_dir, output_dir='outputs/augmented_data', 
                       num_per_original=5):
        """
        Augment existing dataset by generating multiple synthetic images per original.
        
        Args:
            original_dir: Directory with original defect images
            output_dir: Directory to save augmented dataset
            num_per_original: Number of synthetic images to generate per original
        """
        import os
        from shutil import copy
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy original images
        original_files = [f for f in os.listdir(original_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, filename in enumerate(original_files):
            src = os.path.join(original_dir, filename)
            dest = os.path.join(output_dir, f"original_{idx:04d}.png")
            copy(src, dest)
        
        # Generate synthetic images
        num_to_generate = len(original_files) * num_per_original
        synthetic_images = self.generate_images(num_to_generate)
        
        for idx, image in enumerate(synthetic_images):
            image_np = image.permute(1, 2, 0).numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            
            filename = f"synthetic_{idx:04d}.png"
            filepath = os.path.join(output_dir, filename)
            image_pil.save(filepath)
        
        total = len(original_files) + num_to_generate
        print(f"Augmented dataset saved to {output_dir}")
        print(f"Original images: {len(original_files)}")
        print(f"Synthetic images: {num_to_generate}")
        print(f"Total images: {total}")
        print(f"Augmentation factor: {total / len(original_files):.1f}x")


def main():
    """Example usage of DefectAugmentor."""
    
    # Initialize augmentor with trained model
    augmentor = DefectAugmentor(
        model_path='outputs/generator_final.pth',
        latent_dim=100,
        img_channels=3
    )
    
    # Generate new synthetic defect images
    print("\n--- Generating synthetic defect images ---")
    synthetic_images = augmentor.generate_images(num_images=20)
    augmentor.save_images(synthetic_images, 
                         output_dir='outputs/generated_defects',
                         prefix='generated')
    
    # Visualize samples
    print("\n--- Visualizing samples ---")
    sample_images = augmentor.generate_images(num_images=16, seed=42)
    fig = augmentor.visualize_images(sample_images)
    plt.savefig('outputs/sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Augment existing dataset
    print("\n--- Augmenting dataset ---")
    augmentor.augment_dataset(
        original_dir='data/raw_defects',
        output_dir='outputs/augmented_dataset',
        num_per_original=3
    )


if __name__ == "__main__":
    main()
