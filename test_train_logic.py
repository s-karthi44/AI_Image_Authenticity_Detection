
import os
import sys
# Add current directory to path
sys.path.append(os.getcwd())

from ai_model.train import find_images, prepare_dataset

print("Testing find_images...")
data_dir = os.path.abspath("dataset")
real_dir = os.path.join(data_dir, 'real')

real_images = find_images(real_dir)
print(f"Found {len(real_images)} real images.")

if len(real_images) > 0:
    print(f"First image: {real_images[0]}")
else:
    print("Zero images found!")
    
# Check AI images too
ai_dir = os.path.join(data_dir, 'ai')
ai_images = find_images(ai_dir)
print(f"Found {len(ai_images)} AI images.")
