"""
Validation and testing script for the DCGAN project.
Verifies installation, data generation, model architecture, and basic training.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path


def test_imports():
    """Test all required imports."""
    print("\n[TEST 1] Testing imports...")
    print("-" * 60)
    
    try:
        import torch
        import torchvision
        import numpy
        import PIL
        import matplotlib
        import tqdm
        import cv2
        
        print("✓ All required packages imported successfully")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  PyTorch Vision: {torchvision.__version__}")
        print(f"  NumPy: {numpy.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\n[TEST 2] Testing CUDA availability...")
    print("-" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("Warning: CUDA not available, will use CPU (slower training)")
        return True


def test_model_architecture():
    """Test model loading and forward pass."""
    print("\n[TEST 3] Testing model architecture...")
    print("-" * 60)
    
    try:
        from models.dcgan import Generator, Discriminator, initialize_weights
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test Generator
        print("Testing Generator...")
        generator = Generator(latent_dim=100, img_channels=3).to(device)
        initialize_weights(generator)
        
        z = torch.randn(2, 100).to(device)
        fake_images = generator(z)
        
        assert fake_images.shape == (2, 3, 128, 128), f"Unexpected output shape: {fake_images.shape}"
        print(f"  ✓ Generator output shape: {fake_images.shape}")
        
        # Test Discriminator
        print("Testing Discriminator...")
        discriminator = Discriminator(img_channels=3).to(device)
        initialize_weights(discriminator)
        
        scores = discriminator(fake_images)
        assert scores.shape == (2, 1), f"Unexpected output shape: {scores.shape}"
        print(f"  ✓ Discriminator output shape: {scores.shape}")
        
        # Count parameters
        gen_params = sum(p.numel() for p in generator.parameters())
        dis_params = sum(p.numel() for p in discriminator.parameters())
        print(f"  ✓ Generator parameters: {gen_params:,}")
        print(f"  ✓ Discriminator parameters: {dis_params:,}")
        
        return True
    
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_data_generation():
    """Test synthetic data generation."""
    print("\n[TEST 4] Testing synthetic data generation...")
    print("-" * 60)
    
    try:
        from data.dataset_generator import DefectImageGenerator
        import shutil
        
        # Create temporary directory
        test_dir = 'test_defects'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        # Generate small dataset
        print("Generating 10 test images...")
        generator = DefectImageGenerator(
            img_size=128,
            num_images=10,
            output_dir=test_dir
        )
        
        output_dir = generator.generate_dataset()
        
        # Verify
        num_images = len([f for f in os.listdir(test_dir) if f.endswith('.png')])
        assert num_images == 10, f"Expected 10 images, got {num_images}"
        
        print(f"  ✓ Generated {num_images} images in {test_dir}/")
        
        # Cleanup
        shutil.rmtree(test_dir)
        print("  ✓ Cleanup complete")
        
        return True
    
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False


def test_dataloader():
    """Test DataLoader creation."""
    print("\n[TEST 5] Testing DataLoader...")
    print("-" * 60)
    
    try:
        from data.dataset_generator import DefectImageGenerator
        from data.dataloader import get_dataloader
        import shutil
        
        # Generate test data
        test_dir = 'test_defects'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        print("Creating test dataset...")
        generator = DefectImageGenerator(
            img_size=128,
            num_images=50,
            output_dir=test_dir
        )
        generator.generate_dataset()
        
        # Create dataloader
        print("Creating DataLoader...")
        dataloader = get_dataloader(
            image_dir=test_dir,
            batch_size=16,
            img_size=128,
            shuffle=True
        )
        
        print(f"  ✓ DataLoader created")
        print(f"  ✓ Number of batches: {len(dataloader)}")
        
        # Test iteration
        print("Testing batch loading...")
        for batch_idx, images in enumerate(dataloader):
            print(f"  ✓ Batch {batch_idx + 1}: shape {images.shape}")
            
            assert images.shape[1] == 3, "Expected 3 channels"
            assert images.shape[2] == 128, "Expected 128x128 images"
            assert images.shape[3] == 128, "Expected 128x128 images"
            
            if batch_idx == 0:
                break
        
        # Cleanup
        shutil.rmtree(test_dir)
        print("  ✓ Cleanup complete")
        
        return True
    
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        return False


def test_training_step():
    """Test single training step."""
    print("\n[TEST 6] Testing training step...")
    print("-" * 60)
    
    try:
        from data.dataset_generator import DefectImageGenerator
        from data.dataloader import get_dataloader
        from models.dcgan import Generator, Discriminator, initialize_weights
        import torch.optim as optim
        import torch.nn as nn
        import shutil
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        test_dir = 'test_defects'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        print("Creating test dataset...")
        generator_data = DefectImageGenerator(
            img_size=128,
            num_images=32,
            output_dir=test_dir
        )
        generator_data.generate_dataset()
        
        # Create dataloader
        dataloader = get_dataloader(
            image_dir=test_dir,
            batch_size=16,
            img_size=128
        )
        
        # Create models
        print("Creating models...")
        generator = Generator(latent_dim=100, img_channels=3).to(device)
        discriminator = Discriminator(img_channels=3).to(device)
        
        initialize_weights(generator)
        initialize_weights(discriminator)
        
        # Create optimizers
        optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        # Training step
        print("Executing training step...")
        
        generator.train()
        discriminator.train()
        
        for batch_idx, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            
            # Train discriminator
            optimizer_d.zero_grad()
            
            real_output = discriminator(real_images)
            d_real_loss = criterion(real_output, real_label)
            
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_output, fake_label)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)
            fake_output = discriminator(fake_images)
            
            g_loss = criterion(fake_output, real_label)
            g_loss.backward()
            optimizer_g.step()
            
            print(f"  ✓ Batch {batch_idx + 1}")
            print(f"    D Loss: {d_loss.item():.4f}")
            print(f"    G Loss: {g_loss.item():.4f}")
            
            if batch_idx == 0:
                break
        
        # Cleanup
        shutil.rmtree(test_dir)
        print("  ✓ Training step successful")
        print("  ✓ Cleanup complete")
        
        return True
    
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("DCGAN PROJECT - VALIDATION TESTS")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Model Architecture", test_model_architecture),
        ("Data Generation", test_data_generation),
        ("DataLoader", test_dataloader),
        ("Training Step", test_training_step),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_workflow.py")
        print("2. Or check QUICK_START.md for options")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check errors above.")
    
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
