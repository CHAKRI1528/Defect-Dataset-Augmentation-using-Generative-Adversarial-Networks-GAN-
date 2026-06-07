"""
Generate synthetic defect images for training the GAN.
Creates realistic manufacturing defects with various patterns.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random


class DefectImageGenerator:
    """Generate synthetic defect images for training data."""
    
    def __init__(self, img_size=128, num_images=500, output_dir="data/raw_defects"):
        """
        Args:
            img_size: Size of generated images (square)
            num_images: Number of synthetic defect images to generate
            output_dir: Directory to save generated images
        """
        self.img_size = img_size
        self.num_images = num_images
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_surface_texture(self):
        """Create base surface texture."""
        img = Image.new('RGB', (self.img_size, self.img_size), color=(200, 200, 200))
        pixels = img.load()
        
        # Add noise to simulate material surface
        for i in range(self.img_size):
            for j in range(self.img_size):
                noise = random.randint(-30, 30)
                r = min(255, max(0, 200 + noise))
                g = min(255, max(0, 200 + noise))
                b = min(255, max(0, 200 + noise))
                pixels[i, j] = (r, g, b)
        
        return img
    
    def add_crack(self, img):
        """Add crack defect to image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Generate crack path
        num_points = random.randint(8, 15)
        points = []
        x = random.randint(10, self.img_size - 10)
        y = random.randint(10, self.img_size - 10)
        
        for _ in range(num_points):
            points.append((x, y))
            x += random.randint(-15, 15)
            y += random.randint(-15, 15)
            x = max(10, min(self.img_size - 10, x))
            y = max(10, min(self.img_size - 10, y))
        
        # Draw crack
        width = random.randint(2, 5)
        color = (30, 30, 30, 200)
        draw.line(points, fill=color, width=width)
        
        return img
    
    def add_dent(self, img):
        """Add dent defect to image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Random dent center and size
        center_x = random.randint(30, self.img_size - 30)
        center_y = random.randint(30, self.img_size - 30)
        radius = random.randint(10, 25)
        
        # Draw dent as ellipse
        bbox = [center_x - radius, center_y - radius, 
                center_x + radius, center_y + radius]
        color = (100, 100, 100, 150)
        draw.ellipse(bbox, fill=color)
        
        return img
    
    def add_scratch(self, img):
        """Add scratch defect to image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Random scratch parameters
        start_x = random.randint(0, self.img_size)
        start_y = random.randint(0, self.img_size)
        length = random.randint(30, 80)
        angle = random.uniform(0, 2 * np.pi)
        
        end_x = start_x + int(length * np.cos(angle))
        end_y = start_y + int(length * np.sin(angle))
        
        width = random.randint(1, 3)
        color = (50, 50, 50, 180)
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
        
        return img
    
    def add_discoloration(self, img):
        """Add discoloration defect to image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Random discoloration patch
        center_x = random.randint(20, self.img_size - 20)
        center_y = random.randint(20, self.img_size - 20)
        radius = random.randint(15, 35)
        
        bbox = [center_x - radius, center_y - radius,
                center_x + radius, center_y + radius]
        color_val = random.randint(80, 150)
        color = (color_val, color_val, color_val, 100)
        draw.ellipse(bbox, fill=color)
        
        return img
    
    def generate_defect_image(self):
        """Generate a single defect image with random defects."""
        img = self.generate_surface_texture()
        
        # Randomly add defects (0-3 per image)
        defect_types = []
        if random.random() < 0.5:
            img = self.add_crack(img)
            defect_types.append('crack')
        if random.random() < 0.4:
            img = self.add_dent(img)
            defect_types.append('dent')
        if random.random() < 0.4:
            img = self.add_scratch(img)
            defect_types.append('scratch')
        if random.random() < 0.3:
            img = self.add_discoloration(img)
            defect_types.append('discoloration')
        
        # Apply slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def generate_dataset(self):
        """Generate complete dataset of synthetic defect images."""
        print(f"Generating {self.num_images} synthetic defect images...")
        
        for i in range(self.num_images):
            img = self.generate_defect_image()
            filename = os.path.join(self.output_dir, f"defect_{i:04d}.png")
            img.save(filename)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{self.num_images} images")
        
        print(f"Dataset saved to {self.output_dir}")
        return self.output_dir


if __name__ == "__main__":
    generator = DefectImageGenerator(img_size=128, num_images=500, 
                                      output_dir="data/raw_defects")
    generator.generate_dataset()
