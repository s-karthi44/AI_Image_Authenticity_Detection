import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
from ai_model.model import AIDetectorModel
import ai_model.config as config

class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Handle empty or corrupt images
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle appropriately (here we might fail or return None, 
            # but returning None crashes DataLoader default collate. 
            # Better to return the next valid image or blacklist before dataset creation)
            # For simplicity in this script, we'll try to find another index or just return zeros?
            # Safe bet: fail loudly or filter beforehand. I will filter beforehand.
            dummy = torch.zeros((3, 224, 224)) 
            return dummy, label

def find_images(directory):
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    files = []
    print(f"Scanning {directory}...", flush=True)
    count = 0
    for root, dirs, filenames in os.walk(directory):
        # print(f"  Visiting {root}, dirs: {len(dirs)}, files: {len(filenames)}", flush=True)
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files.append(os.path.join(root, filename))
                count += 1
    print(f"  Found {count} files in {directory}.", flush=True)
    return files

def prepare_dataset(data_dir, limit=None):
    real_dir = os.path.join(data_dir, 'real')
    ai_dir = os.path.join(data_dir, 'ai')

    print(f"Scanning for real images in {real_dir}...")
    real_images = find_images(real_dir)
    print(f"Found {len(real_images)} real images.")

    print(f"Scanning for AI images in {ai_dir}...")
    ai_images = find_images(ai_dir)
    print(f"Found {len(ai_images)} AI images.")

    if not real_images:
        print("Error: No real images found!")
    if not ai_images:
        print("Error: No AI images found!")

    if len(real_images) == 0 or len(ai_images) == 0:
        return [], []

    if limit:
        print(f"Limiting dataset to {limit} images (balanced).")
        import random
        per_class = limit // 2
        
        # Sample with replacement if limit > available (unlikely here but safe)
        if len(real_images) < per_class:
             sampled_real = real_images
        else:
             sampled_real = random.sample(real_images, per_class)
             
        if len(ai_images) < per_class:
             sampled_ai = ai_images
        else:
             sampled_ai = random.sample(ai_images, per_class)
             
        all_files = sampled_real + sampled_ai
        all_labels = [0] * len(sampled_real) + [1] * len(sampled_ai)
        
        # Shuffle together
        combined = list(zip(all_files, all_labels))
        random.shuffle(combined)
        all_files, all_labels = zip(*combined)
        all_files = list(all_files)
        all_labels = list(all_labels)
    else:
        all_files = real_images + ai_images
        all_labels = [0] * len(real_images) + [1] * len(ai_images)

    return all_files, all_labels

def create_dataloaders(files, labels, batch_size=32):
    # Splits
    train_files, val_test_files, train_labels, val_test_labels = train_test_split(
        files, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        val_test_files, val_test_labels, test_size=0.5, random_state=42, stratify=val_test_labels
    )

    print(f"Train size: {len(train_files)}, Val size: {len(val_files)}, Test size: {len(test_files)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(train_files, train_labels, transform=train_transform)
    val_dataset = CustomImageDataset(val_files, val_labels, transform=val_test_transform)
    test_dataset = CustomImageDataset(test_files, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers 0 for Windows compatibility safely
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def train_model(train_loader, val_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AIDetectorModel(base='efficientnet_b7', num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                print(f"\rBatch {i}/{len(train_loader)} Loss: {loss.item():.4f}", end="")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"\nTrain Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, 'best_model.pth')
            print("Saved best model.")

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    return model

def evaluate_model(test_loader, model=None, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None and model_path:
        model = AIDetectorModel(base='efficientnet_b7', num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    elif model:
        model.to(device)
    else:
        print("No model provided for evaluation.")
        return

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit number of images for quick training')
    args = parser.parse_args()

    data_dir = os.path.abspath("dataset")
    files, labels = prepare_dataset(data_dir, limit=args.limit)
    
    if len(files) == 0:
        print("No images found! Check your dataset path.")
    else:
        # Debug labels
        from collections import Counter
        print(f"Labels distribution: {Counter(labels)}")
        
        # Use config params
        train_loader, val_loader, test_loader = create_dataloaders(files, labels, batch_size=config.BATCH_SIZE)
        # Reduced epochs for demonstration if needed, but using config defaults
        model = train_model(train_loader, val_loader, epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE)
        evaluate_model(test_loader, model)
