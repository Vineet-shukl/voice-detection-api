
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from app.config import settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoiceDetector, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.feature_extractor = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} on {self.device}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(settings.MODEL_NAME)
            self.model = AutoModelForAudioClassification.from_pretrained(settings.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, audio_array):
        """
        Predicts whether the audio is REAL or FAKE (AI Generated).
        """
        if self.model is None:
            self.load_model()
            
        try:
            # Prepare input
            inputs = self.feature_extractor(
                audio_array, 
                sampling_rate=settings.SAMPLE_RATE, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # The model specific labels need to be checked.
            # Usually for deepfake models: 0 -> Real, 1 -> Fake or vice versa.
            # We will use id2label to be sure.
            id2label = self.model.config.id2label
            
            # Get the predicted class index
            pred_idx = torch.argmax(probs, dim=-1).item()
            label = id2label[pred_idx]
            confidence = probs[0][pred_idx].item()
            
            # Standardizing output as per requirement
            # If label contains "fake" or "generated", map to "AI_GENERATED"
            # If label contains "real" or "bonafide", map to "HUMAN"
            
            result_label = "UNKNOWN"
            if "fake" in label.lower() or "spoof" in label.lower():
                result_label = "AI_GENERATED"
            elif "real" in label.lower() or "bonafide" in label.lower():
                result_label = "HUMAN"
            else:
                # Fallback based on index if labels are ambiguous
                # For `mo-thecreator/Deepfake-audio-detection`:
                # label 0: real, label 1: fake (Need to verify this mapping or assume standard)
                # Let's trust the label text first.
                result_label = label
                
            return result_label, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

voice_detector = VoiceDetector()
