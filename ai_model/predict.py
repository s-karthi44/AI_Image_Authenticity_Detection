import torch
from torchvision import transforms
from PIL import Image
from ai_model.model import AIDetectorModel

def load_model(checkpoint_path, device='cpu'):
    model = AIDetectorModel(base='resnet50', num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_single(model, image_path, device='cpu'):
    img_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # Class 1 is AI/Fake
        score = probabilities[0][1].item()
    return score

if __name__ == "__main__":
    # Example usage
    import sys
    model_path = 'best_model.pth'
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(model_path, device=device)
            score = predict_single(model, img_path, device=device)
            print(f"Fake Probability: {score:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python -m ai_model.predict <image_path>")
