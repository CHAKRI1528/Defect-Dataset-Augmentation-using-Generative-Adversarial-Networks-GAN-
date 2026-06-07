# Quick Start Guide - Defect Dataset Augmentation with GANs

## 🚀 5-Minute Quick Start

### 1. Setup Environment
```bash
# Navigate to project directory
cd defect-gan-augmentation

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Execute entire workflow in one command
python example_workflow.py --epochs 50 --batch-size 32
```

This will:
- ✅ Generate 500 synthetic defect images
- ✅ Train DCGAN for 50 epochs
- ✅ Generate augmented dataset (3x expansion)
- ✅ Save all outputs to `outputs/` directory

**Expected time:** 20-30 minutes on GPU, 1-2 hours on CPU

### 3. View Results
```bash
# Results saved in outputs/ directory
ls outputs/
# Outputs:
# - checkpoints/          : Model checkpoints
# - samples/              : Training samples per epoch
# - logs/                 : TensorBoard logs
# - generator_final.pth   : Final trained model
# - augmented_dataset/    : Generated synthetic defects
```

### 4. Monitor Training (Optional)
```bash
# In a new terminal, view training progress
tensorboard --logdir=outputs/logs
# Open http://localhost:6006 in browser
```

---

## 📚 Detailed Usage

### Generate Only
```bash
python -c "from data.dataset_generator import DefectImageGenerator; \
           DefectImageGenerator(num_images=500).generate_dataset()"
```

### Train Only
```bash
python train.py
```

Edit `train.py` config section to customize:
- Training epochs
- Batch size
- Learning rates
- Output directory

### Augment Existing Dataset
```python
from augment import DefectAugmentor

augmentor = DefectAugmentor('outputs/generator_final.pth')

# Option 1: Generate new images
images = augmentor.generate_images(num_images=100)
augmentor.save_images(images, output_dir='my_dataset')

# Option 2: Augment existing dataset (5x expansion)
augmentor.augment_dataset(
    original_dir='path/to/your/defects',
    output_dir='path/to/augmented',
    num_per_original=5
)
```

### Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/gan_exploration.ipynb
```

Includes:
- Step-by-step training walkthrough
- Visualization of generated images
- Loss curve analysis
- Real vs. generated comparison

---

## 🛠️ Customization

### Change Image Size
```python
# Edit models/dcgan.py or use config
generator = Generator(latent_dim=100)  # Creates 128x128 images
# To change: modify Generator class architecture
```

### Adjust Training Parameters
```python
config = {
    'epochs': 200,          # Train longer
    'batch_size': 16,       # Lower for better quality
    'lr_g': 0.0001,         # Lower learning rate
    'lr_d': 0.0001,
}
```

### Train on Custom Defect Data
```python
from data.dataloader import get_dataloader

dataloader = get_dataloader(
    image_dir='your_defects/',
    batch_size=32,
    img_size=128
)
```

---

## 📊 Expected Performance

| Phase | Expected Behavior |
|-------|-------------------|
| Epoch 1-10 | D loss high (0.8-1.0), generated images are noise |
| Epoch 20-50 | D loss decreases, images show defect patterns |
| Epoch 50+ | Both losses ~0.5-0.7, realistic defects generated |
| Final | G loss: 0.5-1.5, D loss: 0.5-1.0 |

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size in config
'batch_size': 16  # Instead of 32
```

### Training Too Slow
```python
# Use smaller dataset
generator = DefectImageGenerator(num_images=200)

# Or reduce image size (modify Generator in dcgan.py)
```

### Generated Images Blurry
```python
# Train longer
'epochs': 200  # Instead of 100

# Or lower learning rate
'lr_g': 0.0001
'lr_d': 0.0001
```

### GPU Not Detected
```python
python -c "import torch; print(torch.cuda.is_available())"
# If False, install CUDA-enabled PyTorch:
# pip install torch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## 📁 Project Structure

```
defect-gan-augmentation/
├── data/
│   ├── dataset_generator.py
│   ├── dataloader.py
│   └── raw_defects/              # Generated training images
├── models/
│   └── dcgan.py                  # Generator & Discriminator
├── notebooks/
│   └── gan_exploration.ipynb     # Interactive tutorial
├── outputs/
│   ├── checkpoints/
│   ├── samples/
│   ├── logs/
│   └── augmented_dataset/
├── train.py                      # Training script
├── augment.py                    # Inference and augmentation
├── config.py                     # Configuration file
├── example_workflow.py           # Complete end-to-end example
├── utils.py                      # Helper utilities
├── requirements.txt
├── README.md                     # Full documentation
└── QUICK_START.md               # This file
```

---

## 💡 Tips & Best Practices

1. **Start with small dataset**
   - Generate 200 images first to test pipeline
   - Scale up once working

2. **Monitor training**
   - Watch TensorBoard loss curves
   - Mode collapse = losses diverging
   - Healthy = smooth convergence

3. **Save checkpoints**
   - Models saved every 10 epochs (configurable)
   - Can resume from checkpoint anytime
   - Allows experimentation with different schedules

4. **Validate results**
   - Compare real vs. generated visually
   - Check for diversity in synthetic images
   - Test on downstream task (defect classifier)

5. **Scale for production**
   - Use WGAN-GP for stability
   - Implement progressive growing for higher resolution
   - Convert to ONNX for deployment

---

## 🎓 Learning Resources

**Inside this project:**
- `README.md` - Full technical documentation
- `notebooks/gan_exploration.ipynb` - Step-by-step walkthrough
- Well-documented code with comments

**External resources:**
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [GAN Intro](https://arxiv.org/abs/1406.2661)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## 🚀 Next Steps

1. ✅ Complete basic training
2. ✅ Visualize generated images
3. → **Train classifier on augmented data**
4. → **Measure FID score improvement**
5. → **Deploy to production**

---

## 📞 Support

For issues:
1. Check Troubleshooting section above
2. Review training loss in TensorBoard
3. Verify dataset quality in `data/raw_defects/`
4. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

---

**Ready to augment? Let's go!** 🎨✨
