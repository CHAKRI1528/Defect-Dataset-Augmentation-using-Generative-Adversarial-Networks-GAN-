"""
Defect Dataset Augmentation using Generative Adversarial Networks (GAN)

A complete PyTorch implementation of DCGAN for synthetic manufacturing defect
image generation and dataset augmentation.

Example usage:
    # Generate synthetic defects
    from augment import DefectAugmentor
    
    augmentor = DefectAugmentor('outputs/generator_final.pth')
    images = augmentor.generate_images(num_images=50)
    augmentor.save_images(images, output_dir='my_defects')
    
    # Train a new model
    from train import GANTrainer
    from data.dataloader import get_dataloader
    
    config = {
        'epochs': 100,
        'batch_size': 32,
        'latent_dim': 100,
        ...
    }
    
    dataloader = get_dataloader('data/raw_defects', batch_size=32)
    trainer = GANTrainer(config)
    trainer.train(dataloader)
"""

__version__ = '1.0.0'
__author__ = 'DCGAN Project'
__title__ = 'Defect Dataset Augmentation using DCGAN'

# Core modules
from models.dcgan import Generator, Discriminator, initialize_weights
from data.dataloader import DefectDataset, get_dataloader
from data.dataset_generator import DefectImageGenerator

__all__ = [
    'Generator',
    'Discriminator',
    'initialize_weights',
    'DefectDataset',
    'get_dataloader',
    'DefectImageGenerator',
]
