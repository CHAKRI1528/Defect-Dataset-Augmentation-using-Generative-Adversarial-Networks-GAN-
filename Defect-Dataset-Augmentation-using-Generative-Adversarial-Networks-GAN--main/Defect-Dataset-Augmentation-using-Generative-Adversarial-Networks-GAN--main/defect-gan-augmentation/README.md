# Defect Dataset Augmentation using GANs

A complete implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for synthetic manufacturing defect image generation and dataset augmentation.

## Project Overview

This project addresses the challenge of limited defect datasets in manufacturing by using GANs to generate realistic synthetic defect images. The approach includes:

- **DCGAN Architecture**: Generator and Discriminator networks optimized for 128×128 defect images
- **Synthetic Data Generation**: Realistic manufacturing defects (cracks, dents, scratches, discoloration)
- **Training Framework**: Full adversarial training pipeline with monitoring
- **Augmentation Tools**: Convert trained model into production augmentation system
- **Dataset Expansion**: Scale existing defect datasets 3-5x without manual labeling

## Features

✅ **Full DCGAN Implementation**
- Generator: Transforms latent noise → 128×128 defect images
- Discriminator: Binary classification (real vs. generated)
- Proper normalization with batch normalization and activation functions

✅ **Realistic Defect Synthesis**
- Procedurally generated manufacturing defects
- Surface cracks, mechanical dents, scratches, discoloration
- Configurable defect types and intensities

✅ **Production-Ready Training**
- Configurable training parameters
- TensorBoard monitoring
- Checkpoint saving and resuming
- GPU acceleration support

✅ **Dataset Augmentation Pipeline**
- Load trained model and generate new images
- Batch augmentation of existing datasets
- Flexible output formatting

✅ **Visualization & Analysis**
- Sample visualization during training
- Loss tracking over epochs
- Generated defect comparison

## Project Structure

```
defect-gan-augmentation/
├── data/
│   ├── dataset_generator.py      # Synthetic training data creation
│   ├── dataloader.py              # PyTorch Dataset and DataLoader
│   └── raw_defects/               # Generated training images (auto-created)
├── models/
│   └── dcgan.py                   # Generator and Discriminator architectures
├── notebooks/
│   └── gan_exploration.ipynb      # Interactive analysis notebooks
├── outputs/
│   ├── checkpoints/               # Model checkpoints during training
│   ├── samples/                   # Sample images per epoch
│   ├── logs/                      # TensorBoard logs
│   ├── generator_final.pth        # Final trained model
│   └── augmented_dataset/         # Augmented dataset output
├── train.py                       # Main training script
├── augment.py                     # Inference and augmentation script
├── config.py                      # Configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. **Clone/Setup repository**
   ```bash
   cd defect-gan-augmentation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify setup**
   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__)"
   ```

## Quick Start

### Step 1: Generate Synthetic Training Data

```bash
python -c "from data.dataset_generator import DefectImageGenerator; \
           DefectImageGenerator(num_images=500).generate_dataset()"
```

This creates 500 realistic synthetic defect images in `data/raw_defects/`.

**Defect types included:**
- **Cracks**: Irregular linear damage patterns
- **Dents**: Mechanical depression marks
- **Scratches**: Surface abrasion damage
- **Discoloration**: Color and tone variations

### Step 2: Train the DCGAN

```bash
python train.py
```

**What happens during training:**
- Generator learns to create defects that fool the Discriminator
- Discriminator learns to identify real vs. generated defects
- Loss curves converge as quality improves
- Sample images saved every 5 epochs
- Checkpoints saved every 10 epochs
- TensorBoard logs for monitoring

**Expected behavior:**
- **Epoch 1-10**: D loss high, samples are noise
- **Epoch 20-50**: Images start showing defect patterns
- **Epoch 50-100**: Realistic defects, smooth training curves

**Training time:** 
- ~30 mins on GPU (RTX 3060+)
- ~2-3 hours on CPU

**Monitor progress:**
```bash
tensorboard --logdir=outputs/logs
# Open http://localhost:6006 in browser
```

### Step 3: Generate Augmented Images

```bash
python augment.py
```

This uses the trained generator to:
- Generate 20 high-quality synthetic defects
- Create augmented dataset (3x expansion)
- Save to `outputs/augmented_dataset/`

## Usage Examples

### Generate Custom Number of Images

```python
from augment import DefectAugmentor

augmentor = DefectAugmentor('outputs/generator_final.pth')

# Generate 50 images
images = augmentor.generate_images(num_images=50)

# Save to disk
augmentor.save_images(images, output_dir='my_defects')

# Visualize
fig = augmentor.visualize_images(images[:16])
```

### Augment Your Own Dataset

```python
augmentor = DefectAugmentor('outputs/generator_final.pth')

# 5x dataset expansion
augmentor.augment_dataset(
    original_dir='path/to/your/defects',
    output_dir='path/to/augmented',
    num_per_original=5
)
```

### Load Custom Dataset for Training

```python
from data.dataloader import get_dataloader

# Load your defect images
dataloader = get_dataloader(
    image_dir='your_defect_images/',
    batch_size=32,
    img_size=128
)
```

## Configuration

Edit training parameters in `train.py`:

```python
config = {
    'epochs': 100,              # Training epochs
    'batch_size': 32,           # Batch size
    'latent_dim': 100,          # Noise vector dimension
    'lr_g': 0.0002,             # Generator learning rate
    'lr_d': 0.0002,             # Discriminator learning rate
    'img_size': 128,            # Output image size
    'save_interval': 10,        # Save checkpoint every N epochs
    'sample_interval': 5,       # Save samples every N epochs
}
```

## Model Architecture

### Generator
- Input: Latent vector (100D noise)
- FC Layer: 100 → 512×8×8
- 4× Transposed Conv blocks (with batch norm + ReLU)
- Output: 3×128×128 RGB image (Tanh activation)
- **Total params**: ~3.5M

### Discriminator
- Input: 3×128×128 RGB image
- 4× Conv blocks (with batch norm + LeakyReLU)
- Final: Sigmoid for binary classification
- Output: Real/Fake probability (Sigmoid 0-1)
- **Total params**: ~2.8M

### Loss Functions
- **Generator Loss**: Binary Cross Entropy (fool the discriminator)
- **Discriminator Loss**: Binary Cross Entropy (classify real vs. fake)

## Performance Metrics

Expected results after 100 epochs:
- **Discriminator Accuracy**: 50-60% (confused between real/fake - good sign!)
- **Generator Loss**: 0.5-1.5 range
- **Image Quality**: FID ~30-50 (varies with convergence)

Visual quality assessment:
- ✅ Recognizable defect patterns
- ✅ Realistic textures similar to originals
- ✅ Diverse variations (not repetitive)
- ⚠️ Some minor artifacts normal for GAN training

## Troubleshooting

### Mode Collapse (Generator creates identical images)
- **Solution**: Adjust learning rates, use spectral normalization
- **Code change**: Reduce `lr_g` or increase `lr_d`

### Training Crashes with Memory Error
- **Solution**: Reduce batch size
- **Change**: `'batch_size': 16` (instead of 32)

### Generated Images are Blurry
- **Cause**: Underfitting, training too early
- **Solution**: Train more epochs (200+) or use better learning rate schedule

### GPU Not Used
- **Check**: `python -c "import torch; print(torch.cuda.is_available())"`
- **Solution**: Install CUDA, reinstall PyTorch with CUDA support

## Advanced Usage

### Custom Defect Generator

Create defect types in `data/dataset_generator.py`:

```python
def add_rust(self, img):
    """Add rust corrosion defect."""
    draw = ImageDraw.Draw(img, 'RGBA')
    # Your rust generation code here
    return img
```

### Transfer Learning

Start training from existing checkpoint:

```python
trainer = GANTrainer(config)
epoch = trainer.load_checkpoint('outputs/checkpoints/checkpoint_epoch_50.pth')
trainer.train(dataloader)
```

### Conditional Generation (CGAN)

Modify to generate specific defect types:
- Add class embedding to latent vector
- Condition discriminator on defect label
- See `models/dcgan.py` for extension points

## Results & Output

After successful training:

```
outputs/
├── samples/
│   ├── samples_epoch_5.png      # Grid of 16 samples at epoch 5
│   ├── samples_epoch_10.png
│   └── ...
├── checkpoints/
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_20.pth
│   └── ...
├── logs/
│   └── events.*                 # TensorBoard logs
├── generator_final.pth          # Trained generator model
└── augmented_dataset/
    ├── original_0000.png
    ├── original_0001.png
    ├── synthetic_0000.png
    ├── synthetic_0001.png
    └── ...
```

## Performance Optimization

### For Faster Training
1. Reduce image size: `img_size=64` (4x speedup)
2. Reduce epochs: Start with 50
3. Increase batch size: `batch_size=64` (GPU permitting)

### For Better Quality
1. Train longer: 200-300 epochs
2. Reduce batch size: `batch_size=16`
3. Lower learning rates: `lr_g=0.0001`
4. More training data: Generate 1000+ images

## References

**Original Papers:**
- [Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)](https://arxiv.org/abs/1511.06434)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

**Relevant for Manufacturing:**
- Industrial defect detection datasets
- Quality assurance automation
- Data augmentation for computer vision

## License

MIT License - Feel free to use and modify!

## Contributing

Areas for enhancement:
- [ ] WGAN-GP for improved stability
- [ ] Progressive GAN for higher resolution
- [ ] Conditional GAN for defect-specific generation
- [ ] Multi-GPU training support
- [ ] Automated hyperparameter tuning
- [ ] Web UI for augmentation

## Contact & Support

For issues or questions:
1. Check troubleshooting section
2. Review training loss patterns in TensorBoard
3. Verify dataset quality in `data/raw_defects/`

---

**Happy Augmenting!** 🎨✨
