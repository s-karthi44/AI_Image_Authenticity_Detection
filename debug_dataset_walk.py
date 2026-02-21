
import os

def find_images_walk(directory):
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    files = []
    print(f"Scanning {directory} using os.walk", flush=True)
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files.append(os.path.join(root, filename))
    return files

data_dir = os.path.abspath("dataset")
real_dir = os.path.join(data_dir, 'real')
ai_dir = os.path.join(data_dir, 'ai')

real_images = find_images_walk(real_dir)
print(f"Found {len(real_images)} real images.", flush=True)

ai_images = find_images_walk(ai_dir)
print(f"Found {len(ai_images)} AI images.", flush=True)
