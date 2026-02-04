"""
Evaluate metrics of the current model on test files
"""
import torch
import librosa
import numpy as np
from app.core.model import voice_detector
from app.config import settings

def evaluate_file(filepath, expected_label):
    print(f"Testing {filepath}...")
    # Load audio
    audio, sr = librosa.load(filepath, sr=settings.SAMPLE_RATE)
    
    # Predict
    label, confidence = voice_detector.predict(audio)
    
    print(f"  Expected: {expected_label}")
    print(f"  Predicted: {label}")
    print(f"  Confidence: {confidence:.4f}")
    
    is_correct = (label == expected_label)
    print(f"  Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
    return is_correct

print("--- Baseline Model Evaluation ---")
try:
    # Test Human Voice
    evaluate_file("tests/audio_samples/human_voice_test.wav", "HUMAN")
    print("-" * 20)
    
    # Test AI Voice
    evaluate_file("tests/audio_samples/ai_voice_test.wav", "AI_GENERATED")
    
except Exception as e:
    print(f"Error evaluating: {e}")
