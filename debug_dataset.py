
import os
import glob

def find_images(directory):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    files = []
    print(f"Scanning {directory}", flush=True)
    for ext in extensions:
        found = glob.glob(os.path.join(directory, '**', ext), recursive=True)
        files.extend(found)
        print(f"  Found {len(found)} with extension {ext}", flush=True)
    return files

data_dir = os.path.abspath("dataset")
real_dir = os.path.join(data_dir, 'real')
ai_dir = os.path.join(data_dir, 'ai')

real_images = find_images(real_dir)
print(f"Found {len(real_images)} real images.")

ai_images = find_images(ai_dir)
print(f"Found {len(ai_images)} AI images.")
