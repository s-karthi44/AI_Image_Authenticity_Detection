
import os
import sys
# Add current directory to path
sys.path.append(os.getcwd())

from ai_model.train import find_images

print("Testing find_images v2...", flush=True)
data_dir = os.path.abspath("dataset")
real_dir = os.path.join(data_dir, 'real')
print(f"Real dir: {real_dir}", flush=True)

real_images = find_images(real_dir)
print(f"Returned from find_images(real). Count: {len(real_images)}", flush=True)
