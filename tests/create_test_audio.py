"""
Generate test audio samples for the Voice Detection API.
Creates both human-like and AI-like audio patterns for testing.
"""
import numpy as np
import soundfile as sf
from scipy import signal
import os

def create_human_voice_sample(duration=3.0, sample_rate=16000):
    """
    Create a synthetic human-like voice sample.
    Uses multiple harmonics and natural variations.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Fundamental frequency (typical human voice range: 85-255 Hz)
    f0 = 150  # Hz
    
    # Create harmonics (what makes it sound voice-like)
    audio = np.zeros_like(t)
    for harmonic in range(1, 8):
        frequency = f0 * harmonic
        amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
        
        # Add slight frequency modulation (vibrato)
        vibrato = 5 * np.sin(2 * np.pi * 5 * t)
        audio += amplitude * np.sin(2 * np.pi * (frequency + vibrato) * t)
    
    # Add amplitude envelope (attack, sustain, release)
    envelope = np.ones_like(t)
    attack_samples = int(0.1 * sample_rate)
    release_samples = int(0.2 * sample_rate)
    
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    audio *= envelope
    
    # Add some noise (breath sounds)
    noise = np.random.normal(0, 0.02, len(t))
    audio += noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sample_rate

def create_ai_voice_sample(duration=3.0, sample_rate=16000):
    """
    Create a synthetic AI-like voice sample.
    More regular patterns, less natural variation.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # More regular fundamental frequency
    f0 = 180  # Hz
    
    # Create very regular harmonics (AI voices tend to be more precise)
    audio = np.zeros_like(t)
    for harmonic in range(1, 10):
        frequency = f0 * harmonic
        amplitude = 1.0 / (harmonic * 1.2)
        audio += amplitude * np.sin(2 * np.pi * frequency * t)
    
    # More regular envelope
    envelope = np.ones_like(t)
    attack_samples = int(0.05 * sample_rate)
    release_samples = int(0.1 * sample_rate)
    
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    audio *= envelope
    
    # Less noise (AI voices are cleaner)
    noise = np.random.normal(0, 0.005, len(t))
    audio += noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sample_rate

def main():
    output_dir = "tests/audio_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating test audio samples...")
    
    # Create human voice samples
    print("1. Creating human voice sample...")
    human_audio, sr = create_human_voice_sample(duration=3.0)
    human_path = os.path.join(output_dir, "human_voice_test.wav")
    sf.write(human_path, human_audio, sr)
    print(f"   ✅ Saved: {human_path}")
    
    # Create AI voice samples
    print("2. Creating AI voice sample...")
    ai_audio, sr = create_ai_voice_sample(duration=3.0)
    ai_path = os.path.join(output_dir, "ai_voice_test.wav")
    sf.write(ai_path, ai_audio, sr)
    print(f"   ✅ Saved: {ai_path}")
    
    # Create a short sample (for edge case testing)
    print("3. Creating short sample...")
    short_audio, sr = create_human_voice_sample(duration=0.5)
    short_path = os.path.join(output_dir, "short_sample.wav")
    sf.write(short_path, short_audio, sr)
    print(f"   ✅ Saved: {short_path}")
    
    print(f"\n✅ All test audio samples created in: {output_dir}")
    print("\nYou can now test these with:")
    print("  python verify.py tests/audio_samples/human_voice_test.wav")
    print("  python verify.py tests/audio_samples/ai_voice_test.wav")

if __name__ == "__main__":
    main()
