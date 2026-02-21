
import os

print(f"Listing dataset/real using os.listdir", flush=True)
real_dir = os.path.abspath("dataset/real")
try:
    print(os.listdir(real_dir))
except Exception as e:
    print(f"Error: {e}")

print(f"Listing dataset/real/Human Faces Dataset using os.listdir", flush=True)
try:
    print(os.listdir(os.path.join(real_dir, "Human Faces Dataset")))
except Exception as e:
    print(f"Error: {e}")
    
print(f"Listing dataset/real/Human Faces Dataset/Real Images using os.listdir", flush=True)
try:
    path = os.path.join(real_dir, "Human Faces Dataset", "Real Images")
    items = os.listdir(path)
    print(f"Found {len(items)} items. First 10: {items[:10]}")
except Exception as e:
    print(f"Error: {e}")
