"""
Configuration file for the DCGAN training pipeline.
Centralized settings for easy experimentation.
"""

import os

# Training Configuration
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Model Configuration
LATENT_DIM = 100
IMG_CHANNELS = 3
IMG_SIZE = 128

# Data Configuration
DATA_DIR = 'data'
RAW_DEFECTS_DIR = os.path.join(DATA_DIR, 'raw_defects')
NUM_TRAINING_IMAGES = 500
NUM_WORKERS = 0  # Set to 2-4 for faster data loading on multi-core systems

# Output Configuration
OUTPUT_DIR = 'outputs'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'samples')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
AUGMENTED_DIR = os.path.join(OUTPUT_DIR, 'augmented_dataset')

# Training Parameters
SAVE_INTERVAL = 10  # Save checkpoint every N epochs
SAMPLE_INTERVAL = 5  # Generate samples every N epochs
PRINT_INTERVAL = 50  # Print loss every N batches

# Augmentation Configuration
AUGMENTATION_RATIO = 3  # Generate N synthetic images per original
NUM_AUGMENTED_SAMPLES = 50  # Number of samples to generate for visualization

# Device Configuration (auto-detect)
DEVICE = 'cuda'  # Set to 'cpu' for CPU training

# Synthetic Data Generation
DEFECT_TYPES = ['crack', 'dent', 'scratch', 'discoloration']
DEFECT_PROBABILITY = {
    'crack': 0.5,
    'dent': 0.4,
    'scratch': 0.4,
    'discoloration': 0.3,
}

# Quality Assurance
FID_EVAL_INTERVAL = 20  # Evaluate FID score every N epochs (requires inception_score)
EVALUATE_FID = False  # Set to True if you have inception_score installed
