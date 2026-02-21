class DecisionEngine:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                'prnu': 0.20,
                'frequency': 0.15,
                'pixel_noise': 0.10,
                'facial': 0.10,
                'shadow': 0.10,
                'reflection': 0.10,
                'metadata': 0.05,
                'ai_model': 0.20
            }
        else:
            self.weights = weights

    def fuse_scores(self, scores_dict):
        final_score = 0.0
        total_weight = 0.0
        for key, weight in self.weights.items():
            if key in scores_dict and scores_dict[key] is not None:
                final_score += scores_dict[key] * weight
                total_weight += weight
        
        # Normalize if some scores are missing
        if total_weight > 0:
            final_score = final_score / total_weight
        else:
            final_score = 0.5  # Uncertain if no data
            
        return final_score

    def classify(self, final_score):
        if final_score < 0.48:
            return "AI_GENERATED"
        elif final_score <= 0.52:
            return "UNCERTAIN"
        else:
            return "REAL"

    def compute_confidence(self, final_score, verdict):
        # Scale the score to a confidence percentage
        if verdict == "AI_GENERATED":
            # Map 0.48 -> 50%, 0.0 -> 100%
            conf = (0.48 - final_score) / 0.48 * 50 + 50
        elif verdict == "REAL":
            # Map 0.52 -> 50%, 1.0 -> 100%
            conf = (final_score - 0.52) / 0.48 * 50 + 50
        else:
            # Uncertain: map 0.48-0.52 to 0-49%
            dist_from_center = abs(final_score - 0.5)
            if dist_from_center == 0:
                conf = 0
            else:
                conf = (dist_from_center / 0.02) * 49
            
        return min(max(int(conf), 0), 100)

    def generate_reasoning(self, scores, verdict):
        reasoning = []
        
        prnu = scores.get('prnu')
        if prnu is not None:
            if prnu < 0.3:
                reasoning.append("No camera sensor noise detected")
            elif prnu > 0.7:
                reasoning.append("Natural camera sensor noise (PRNU) detected")
                
        freq = scores.get('frequency')
        if freq is not None:
            if freq > 0.7:
                reasoning.append("Frequency artifacts characteristic of AI generation observed")
            elif freq < 0.3:
                reasoning.append("Smooth natural frequency decay observed")
                
        facial = scores.get('facial')
        if facial is not None:
            if facial < 0.3:
                reasoning.append("Severe facial inconsistencies or unnatural symmetry")
            elif facial > 0.8:
                reasoning.append("Natural facial geometry observed")
                
        shadow = scores.get('shadow')
        if shadow is not None:
            if shadow < 0.3:
                reasoning.append("Anomalous flat lighting or inconsistent shadow geometry")
            elif shadow > 0.7:
                reasoning.append("Consistent natural shadow geometry")
                
        refl = scores.get('reflection')
        if refl is not None:
            if refl < 0.3:
                reasoning.append("Mismatched eye catchlights / reflections")
            elif refl > 0.7:
                reasoning.append("Consistent bilateral eye catchlights")
                
        meta = scores.get('metadata')
        if meta is not None:
            if meta < 0.3:
                reasoning.append("EXIF data stripped or signs of tampered software")
            elif meta > 0.8:
                reasoning.append("Authentic camera EXIF metadata found")
                
        pixel = scores.get('pixel_noise')
        if pixel is not None:
            if pixel < 0.4:
                reasoning.append("Unnatural pixel noise patterns detected")
                
        ai_model = scores.get('ai_model')
        if ai_model is not None:
            if ai_model > 0.8:
                reasoning.append("Strong AI fingerprint detected by neural network")
            elif ai_model < 0.2:
                reasoning.append("CNN model confirms natural photographic features")
                
        # Default reasoning if empty
        if not reasoning:
            reasoning.append("Based on aggregate score from forensic analysis.")
            
        return reasoning

    def analyze(self, scores_dict):
        final_score = self.fuse_scores(scores_dict)
        verdict = self.classify(final_score)
        confidence = self.compute_confidence(final_score, verdict)
        reasoning = self.generate_reasoning(scores_dict, verdict)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "final_score": final_score,
            "scores": scores_dict,
            "reasoning": reasoning
        }
