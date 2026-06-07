"""
Complete example workflow: Generate data, train model, and augment dataset.
Run this script to execute the entire pipeline end-to-end.
"""

import os
import torch
import argparse
from pathlib import Path

from data.dataset_generator import DefectImageGenerator
from data.dataloader import get_dataloader
from models.dcgan import Generator, Discriminator, initialize_weights
from train import GANTrainer
from augment import DefectAugmentor


def create_example_workflow(args):
    """Execute complete DCGAN workflow."""
    
    print("\n" + "="*60)
    print("DEFECT DATASET AUGMENTATION USING DCGAN")
    print("="*60)
    
    # ============ Step 1: Generate Training Data ============
    print("\n[STEP 1] Generating synthetic defect training data...")
    print("-" * 60)
    
    generator = DefectImageGenerator(
        img_size=args.img_size,
        num_images=args.num_training_images,
        output_dir=args.data_dir
    )
    generator.generate_dataset()
    print(f"✓ Generated {args.num_training_images} training images")
    
    # ============ Step 2: Configure Training ============
    print("\n[STEP 2] Configuring training parameters...")
    print("-" * 60)
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'latent_dim': args.latent_dim,
        'img_channels': 3,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'beta1': 0.5,
        'beta2': 0.999,
        'img_size': args.img_size,
        'output_dir': args.output_dir,
        'log_dir': os.path.join(args.output_dir, 'logs'),
        'save_interval': args.save_interval,
        'sample_interval': args.sample_interval,
    }
    
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rates: G={config['lr_g']}, D={config['lr_d']}")
    print(f"Image size: {config['img_size']}x{config['img_size']}")
    
    # ============ Step 3: Train DCGAN ============
    print("\n[STEP 3] Training DCGAN...")
    print("-" * 60)
    
    # Get dataloader
    dataloader = get_dataloader(
        image_dir=args.data_dir,
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    # Create trainer
    trainer = GANTrainer(config)
    
    # Train model
    if args.skip_training:
        print("⊘ Training skipped (--skip-training flag set)")
    else:
        trainer.train(dataloader)
        print("✓ Training completed!")
        
        # Save final model
        torch.save(trainer.generator.state_dict(), 
                  os.path.join(args.output_dir, 'generator_final.pth'))
        print(f"✓ Final model saved to {args.output_dir}/generator_final.pth")
    
    # ============ Step 4: Generate Augmented Images ============
    print("\n[STEP 4] Generating augmented defect images...")
    print("-" * 60)
    
    # Initialize augmentor
    model_path = os.path.join(args.output_dir, 'generator_final.pth')
    
    if os.path.exists(model_path):
        augmentor = DefectAugmentor(
            model_path=model_path,
            latent_dim=config['latent_dim'],
            img_channels=3
        )
        
        # Generate synthetic images
        num_synthetic = len(os.listdir(args.data_dir)) * args.augment_factor
        synthetic_images = augmentor.generate_images(num_images=int(num_synthetic))
        
        # Save augmented images
        augmented_dir = os.path.join(args.output_dir, 'augmented_dataset')
        augmentor.save_images(
            synthetic_images,
            output_dir=augmented_dir,
            prefix='synthetic'
        )
        
        # Print augmentation summary
        original_count = len(os.listdir(args.data_dir))
        synthetic_count = len(synthetic_images)
        total_count = original_count + synthetic_count
        
        print("\n" + "="*60)
        print("AUGMENTATION SUMMARY")
        print("="*60)
        print(f"Original images:     {original_count}")
        print(f"Synthetic images:    {synthetic_count}")
        print(f"Total images:        {total_count}")
        print(f"Augmentation factor: {total_count / original_count:.2f}x")
        print("="*60)
        
    else:
        print(f"⚠ Model not found at {model_path}")
        print("  Skipping augmentation step")
    
    print("\n✓ Workflow completed successfully!")
    print("\nOutput directory structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── checkpoints/")
    print(f"  ├── samples/")
    print(f"  ├── logs/")
    print(f"  ├── generator_final.pth")
    print(f"  └── augmented_dataset/")


def main():
    parser = argparse.ArgumentParser(
        description='Complete DCGAN workflow for defect dataset augmentation'
    )
    
    # Data parameters
    parser.add_argument('--img-size', type=int, default=128,
                       help='Image size (default: 128)')
    parser.add_argument('--num-training-images', type=int, default=500,
                       help='Number of training images to generate (default: 500)')
    parser.add_argument('--data-dir', type=str, default='data/raw_defects',
                       help='Directory for training data (default: data/raw_defects)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--latent-dim', type=int, default=100,
                       help='Latent dimension (default: 100)')
    parser.add_argument('--lr-g', type=float, default=0.0002,
                       help='Generator learning rate (default: 0.0002)')
    parser.add_argument('--lr-d', type=float, default=0.0002,
                       help='Discriminator learning rate (default: 0.0002)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--sample-interval', type=int, default=5,
                       help='Save samples every N epochs (default: 5)')
    
    # Augmentation parameters
    parser.add_argument('--augment-factor', type=float, default=3.0,
                       help='Augmentation factor (synthetic/original ratio) (default: 3.0)')
    
    # Workflow control
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step (use existing model)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run workflow
    create_example_workflow(args)


if __name__ == "__main__":
    main()
