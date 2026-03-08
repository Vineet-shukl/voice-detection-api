
import io
import base64
import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException
from app.config import settings
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


def decode_base64_audio(base64_string: str) -> io.BytesIO:
    """Decodes a Base64 string into a BytesIO object."""
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        audio_data = base64.b64decode(base64_string)
        return io.BytesIO(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 audio: {str(e)}")


def compute_audio_profile(audio_samples: np.ndarray, sample_rate: int) -> dict:
    """
    Compute a technical profile of the audio sample.
    Returns metadata useful for quality assessment and forensic analysis.
    """
    duration = len(audio_samples) / sample_rate

    # RMS energy (simple vector op)
    rms_energy = float(np.sqrt(np.mean(audio_samples ** 2)))

    # Optimization: Skip expensive spectral SNR calculation here
    # (It requires STFT/RMS framing which takes ~100ms)
    # The forensic module can compute detailed SNR if needed.
    snr_db = 0.0 

    # Clipping detection — samples at or near ±1.0
    clip_threshold = 0.999
    # Vectorized fast check
    clipping_ratio = float(np.mean(np.abs(audio_samples) > clip_threshold))
    clipping_detected = clipping_ratio > 0.001

    # Silence ratio (vectorized)
    silence_threshold = rms_energy * 0.1
    silence_ratio = float(np.mean(np.abs(audio_samples) < silence_threshold))

    return {
        "duration_sec": round(duration, 2),
        "snr_db": round(snr_db, 1), # Placeholder, computed later if needed
        "clipping_detected": clipping_detected,
        "silence_ratio": round(silence_ratio, 3),
        "rms_energy": round(rms_energy, 4),
        "sample_rate": sample_rate,
    }


def segment_audio(audio_samples: np.ndarray, sample_rate: int, segment_sec: float = 5.0,
                  overlap_sec: float = 1.0) -> list:
    """
    Split audio into overlapping segments for per-segment analysis.
    Short audio (< segment_sec) is returned as a single segment.
    """
    segment_length_samples = int(segment_sec * sample_rate)
    hop_length_samples = int((segment_sec - overlap_sec) * sample_rate)

    if len(audio_samples) <= segment_length_samples:
        return [audio_samples]

    segments = []
    start = 0
    while start < len(audio_samples):
        end = min(start + segment_length_samples, len(audio_samples))
        audio_segment = audio_samples[start:end]
        # Only include if at least 1 second long
        if len(audio_segment) >= sample_rate:
            segments.append(audio_segment)
        start += hop_length_samples

    return segments if segments else [audio_samples]


def preprocess_audio(audio_file: io.BytesIO):
    """
    Clean and standardized preprocessing for AI detection.
    Focuses on natural signal preservation to avoid false AI classifications.
    Returns: (audio_array, audio_profile_dict)
    """
    try:
        # Save to temporary file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(audio_file.read())
            temp_audio_file_path = tmp_file.name

        try:
            # Load audio at 16kHz (Standard for Wav2Vec2)
            audio_samples, sample_rate = librosa.load(temp_audio_file_path, sr=settings.SAMPLE_RATE)

            # Ensure mono
            if len(audio_samples.shape) > 1:
                audio_samples = librosa.to_mono(audio_samples)

            # Reject extremely short audio
            if len(audio_samples) < sample_rate * 0.3:
                raise HTTPException(
                    status_code=400,
                    detail="Audio too short. Minimum 0.3 seconds required."
                )

            # 1. Basic Silence Trimming (Safer threshold)
            silence_trimmed_audio, _ = librosa.effects.trim(audio_samples, top_db=40)
            if len(silence_trimmed_audio) > sample_rate * 0.1:
                audio_samples = silence_trimmed_audio

            # 2. Gentle Peak Normalization
            # Preserves natural dynamics which models use for detection
            peak_amplitude = np.max(np.abs(audio_samples))
            if peak_amplitude > 0:
                audio_samples = audio_samples / peak_amplitude

            # 3. Time Clamping — max 30 seconds
            max_duration = 30
            if len(audio_samples) > sample_rate * max_duration:
                audio_samples = audio_samples[:sample_rate * max_duration]

            # 4. Compute audio profile
            profile = compute_audio_profile(audio_samples, sample_rate)

            logger.info(
                f"Preprocessing complete: {profile['duration_sec']}s, "
                f"SNR={profile['snr_db']}dB, "
                f"clipping={'YES' if profile['clipping_detected'] else 'NO'}"
            )

            return audio_samples, profile

        finally:
            if os.path.exists(temp_audio_file_path):
                os.unlink(temp_audio_file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")
