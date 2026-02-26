import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import numpy as np

class DeepFeatureAnalyzer:
    """
    Uses pretrained deep networks to find semantic inconsistencies.
    AI images may have physically impossible combinations.
    """
    
    def __init__(self):
        # Load pretrained ResNet50 for feature extraction
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        
        # Remove final classification layer
        self.feature_extractor = torch.nn.Sequential(
            *list(self.model.children())[:-1]
        )
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.feature_extractor.to(self.device)
    
    def extract_regional_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from different regions of image.
        """
        h, w = image.shape[:2]
        
        # Divide into 9 regions (3x3 grid)
        regions = []
        step_h, step_w = max(1, h // 3), max(1, w // 3)
        
        for i in range(3):
            for j in range(3):
                region = image[
                    i*step_h:(i+1)*step_h,
                    j*step_w:(j+1)*step_w
                ]
                if region.shape[0] > 0 and region.shape[1] > 0:
                    regions.append(region)
        
        # Extract features from each region
        features = []
        with torch.no_grad():
            for region in regions:
                try:
                    tensor = self.transform(region).unsqueeze(0).to(self.device)
                    feat = self.feature_extractor(tensor)
                    feat = feat.squeeze().cpu().numpy()
                    features.append(feat)
                except Exception:
                    pass
        
        if not features:
            return np.array([])
        return np.array(features)
    
    def calculate_feature_consistency(self, features: np.ndarray) -> float:
        """
        Real photos: features are consistent (same scene, lighting)
        AI images: features may be inconsistent (pieced together)
        """
        if len(features) < 2:
            return 1.0

        # Calculate pairwise cosine similarity
        similarities = []
        n = len(features)
        
        for i in range(n):
            for j in range(i+1, n):
                norm_i = np.linalg.norm(features[i])
                norm_j = np.linalg.norm(features[j])
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(features[i], features[j]) / (norm_i * norm_j)
                    similarities.append(sim)
        
        if not similarities:
            return 1.0

        # High average similarity = consistent (real)
        # Low similarity = inconsistent (AI)
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Consistent images have high avg, low std
        consistency = avg_similarity * (1 - std_similarity)
        
        return float(consistency)
    
    def detect_impossible_combinations(self, image: np.ndarray) -> float:
        """
        Detect physically impossible feature combinations.
        Example: Indoor lighting + outdoor background
        """
        try:
            # Extract global features
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Use pretrained model predictions
            # Get top-k predictions
            self.model.to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top5_probs, top5_indices = torch.topk(probs, 5)
            
            # Check for conflicting categories
            top_probs = top5_probs[0].cpu().numpy()
            
            # If confidence is spread across many categories = inconsistent
            entropy = -np.sum(top_probs * np.log(top_probs + 1e-10))
            
            # Normalize entropy (max entropy for 5 categories)
            max_entropy = -np.log(1/5)
            normalized_entropy = entropy / max_entropy
            
            # High entropy = inconsistent/impossible combinations
            return float(normalized_entropy)
        except Exception:
            return 0.0
    
    def analyze(self, image: np.ndarray) -> dict:
        """
        Complete deep feature analysis.
        """
        regional_features = self.extract_regional_features(image)
        consistency = self.calculate_feature_consistency(regional_features)
        impossibility = self.detect_impossible_combinations(image)
        
        # Combine scores
        # High consistency + low impossibility = real
        # Low consistency or high impossibility = AI
        
        inconsistency_score = (1 - consistency) * 0.6 + impossibility * 0.4
        
        return {
            'inconsistency_score': float(max(0, min(1, inconsistency_score))),
            'regional_consistency': float(consistency),
            'impossibility_score': float(impossibility),
            'has_inconsistencies': inconsistency_score > 0.5
        }
