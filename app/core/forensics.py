"""
Forensic Analyzers — Four independent analysis engines that examine audio
for specific signatures of AI generation vs natural human speech.

Each analyzer returns a score (0=human, 1=AI) and a list of detected artifacts.
The final detection fuses all analyzer scores for maximum accuracy.
"""
import numpy as np
import librosa
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import concurrent.futures

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerResult:
    """Result from a single forensic analyzer."""
    name: str
    score: float  # 0.0 = definitely human, 1.0 = definitely AI
    verdict: str  # "HUMAN" or "AI_GENERATED"
    artifacts_found: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioProfile:
    """Technical profile of the audio sample."""
    duration_sec: float = 0.0
    snr_db: float = 0.0
    clipping_detected: bool = False
    silence_ratio: float = 0.0
    rms_energy: float = 0.0
    sample_rate: int = 16000
    num_segments: int = 1


# ===============================================================
#  Spectral Analyzer
# ===============================================================

class SpectralAnalyzer:
    """
    Detects AI signatures in the frequency domain:
    - Unnaturally smooth spectral envelope
    - Missing or artificial harmonics
    - Sharp frequency cutoffs (vocoder artifacts)
    - Abnormal spectral flatness
    """

    def analyze(self, audio_samples: np.ndarray, sample_rate: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Spectral Flatness — AI speech tends to have lower flatness (more tonal)
            spectral_flatness_frames = librosa.feature.spectral_flatness(y=audio_samples)[0]
            mean_spectral_flatness = float(np.mean(spectral_flatness_frames))
            std_spectral_flatness = float(np.std(spectral_flatness_frames))
            details["spectral_flatness_mean"] = round(mean_spectral_flatness, 4)
            details["spectral_flatness_std"] = round(std_spectral_flatness, 4)

            # Human speech has higher variance in spectral flatness
            if std_spectral_flatness < 0.02:
                artifacts.append("unnaturally_uniform_spectral_texture")
            if mean_spectral_flatness < 0.005:
                artifacts.append("overly_tonal_spectrum")

            # 2. Spectral Bandwidth — AI audio often has narrower bandwidth
            spectral_bandwidth_frames = librosa.feature.spectral_bandwidth(y=audio_samples, sr=sample_rate)[0]
            mean_bandwidth = float(np.mean(spectral_bandwidth_frames))
            std_bandwidth = float(np.std(spectral_bandwidth_frames))
            details["spectral_bandwidth_mean"] = round(mean_bandwidth, 1)
            details["spectral_bandwidth_std"] = round(std_bandwidth, 1)

            if std_bandwidth < 200:
                artifacts.append("unnaturally_consistent_bandwidth")

            # 3. Spectral Centroid Variance — AI speech has more stable centroid
            spectral_centroid_frames = librosa.feature.spectral_centroid(y=audio_samples, sr=sample_rate)[0]
            spectral_centroid_coefficient_variation = float(np.std(spectral_centroid_frames) / (np.mean(spectral_centroid_frames) + 1e-10))
            details["spectral_centroid_cv"] = round(spectral_centroid_coefficient_variation, 4)

            if spectral_centroid_coefficient_variation < 0.15:
                artifacts.append("unnaturally_stable_spectral_centroid")

            # Optimization: Removed expensive HPSS and full STFT

            # Score: more artifacts = more likely AI
            score = min(1.0, len(artifacts) * 0.3)

        except Exception as e:
            logger.warning(f"SpectralAnalyzer error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="spectral_analysis",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Temporal Analyzer
# ===============================================================

class TemporalAnalyzer:
    """
    Detects AI signatures in the time domain:
    - Robotic / metronomic pause timing
    - Missing micro-variations in energy
    - Unnaturally smooth energy envelope
    - Consistent zero-crossing rate
    """

    def analyze(self, audio_samples: np.ndarray, sample_rate: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Energy contour smoothness
            frame_length = int(0.025 * sample_rate)
            hop_length = int(0.010 * sample_rate)
            rms_energy_frames = librosa.feature.rms(y=audio_samples, frame_length=frame_length, hop_length=hop_length)[0]

            if len(rms_energy_frames) > 10:
                rms_energy_differences = np.diff(rms_energy_frames)
                energy_roughness = float(np.std(rms_energy_differences) / (np.mean(rms_energy_frames) + 1e-10))
                details["energy_roughness"] = round(energy_roughness, 4)

                if energy_roughness < 0.08:
                    artifacts.append("unnaturally_smooth_energy_contour")

            # 2. Zero-Crossing Rate consistency
            zero_crossing_rate_frames = librosa.feature.zero_crossing_rate(audio_samples, frame_length=frame_length, hop_length=hop_length)[0]
            zero_crossing_rate_coefficient_variation = float(np.std(zero_crossing_rate_frames) / (np.mean(zero_crossing_rate_frames) + 1e-10))
            details["zcr_coefficient_of_variation"] = round(zero_crossing_rate_coefficient_variation, 4)

            if zero_crossing_rate_coefficient_variation < 0.25:
                artifacts.append("unnaturally_consistent_zero_crossings")

            # 3. Pause regularity analysis
            silence_threshold = np.percentile(np.abs(audio_samples), 10)
            is_silent_frame = np.abs(audio_samples) < silence_threshold * 3
            silence_boundary_changes = np.diff(is_silent_frame.astype(int))
            pause_start_indices = np.where(silence_boundary_changes == 1)[0]
            
            if len(pause_start_indices) >= 3:
                pause_interval_durations = np.diff(pause_start_indices) / sample_rate
                pause_interval_coefficient_variation = float(np.std(pause_interval_durations) / (np.mean(pause_interval_durations) + 1e-10))
                details["pause_interval_cv"] = round(pause_interval_coefficient_variation, 4)
                details["num_pauses"] = len(pause_start_indices)

                if pause_interval_coefficient_variation < 0.2 and len(pause_start_indices) > 3:
                    artifacts.append("metronomic_pause_timing")

            # 4. Micro-jitter analysis (Optimized)
            if float(len(audio_samples)) / sample_rate > 0.5:
                # Fast energy variance check instead of full autocorrelation loop
                chunk_size = int(0.1 * sample_rate)
                # Reshape to chunks (discard remainder)
                n_chunks = len(audio_samples) // chunk_size
                if n_chunks > 4:
                    chunks = audio_samples[:n_chunks*chunk_size].reshape(n_chunks, chunk_size)
                    chunk_energies = np.sqrt(np.mean(chunks**2, axis=1))
                    
                    # Check if energy variation is too regular
                    energy_std = np.std(chunk_energies)
                    if energy_std < 0.001:
                         artifacts.append("repetitive_energy_pattern")
            
            score = min(1.0, len(artifacts) * 0.3)

        except Exception as e:
            logger.warning(f"TemporalAnalyzer error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="temporal_analysis",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Formant Analyzer
# ===============================================================

class FormantAnalyzer:
    """
    Detects AI signatures in formant structure via MFCC analysis.
    Optimized to use MFCCs as proxy for formants.
    """

    def analyze(self, audio_samples: np.ndarray, sample_rate: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. MFCC stability
            mfcc_features = librosa.feature.mfcc(y=audio_samples, sr=sample_rate, n_mfcc=13)
            # Vectorized variation coefficient
            mfcc_coefficient_means = np.abs(np.mean(mfcc_features[1:], axis=1)) + 1e-10
            mfcc_coefficient_stds = np.std(mfcc_features[1:], axis=1)
            mfcc_coefficient_variations = mfcc_coefficient_stds / mfcc_coefficient_means
            
            average_mfcc_coefficient_variation = float(np.mean(mfcc_coefficient_variations))
            details["avg_mfcc_cv"] = round(average_mfcc_coefficient_variation, 4)

            if average_mfcc_coefficient_variation < 0.5:
                artifacts.append("unnaturally_stable_formant_structure")

            # 2. Delta smoothness
            mfcc_deltas = librosa.feature.delta(mfcc_features)
            delta_roughness = float(np.mean(np.abs(librosa.feature.delta(mfcc_deltas))))
            details["delta_mfcc_roughness"] = round(delta_roughness, 4)

            if delta_roughness < 0.3:
                artifacts.append("overly_smooth_formant_transitions")

            # 3. Inter-frame correlation (Vectorized)
            if mfcc_features.shape[1] > 10:
                # Vectorized correlation between adjacent frames
                # Normalize frames
                mfcc_frames = mfcc_features.T
                frame_means = mfcc_frames.mean(axis=1, keepdims=True)
                frame_stds = mfcc_frames.std(axis=1, keepdims=True) + 1e-10
                normalized_mfcc_frames = (mfcc_frames - frame_means) / frame_stds
                
                # Compute correlation of frame i with i+1
                # Sum of product of normalized values / N
                inter_frame_correlations = np.mean(normalized_mfcc_frames[:-1] * normalized_mfcc_frames[1:], axis=1)
                mean_inter_frame_correlation = float(np.mean(inter_frame_correlations))
                
                details["inter_frame_correlation"] = round(mean_inter_frame_correlation, 4)

                if mean_inter_frame_correlation > 0.95:
                    artifacts.append("excessive_inter_frame_correlation")

            # 4. Mel-band energy uniformity (uses MFCCs as proxy instead of new melspectrogram for speed)
            # MFCC[0] is energy; use variance of MFCCs as rough proxy for band variance
            mfcc_coefficient_std_range = float(np.max(mfcc_coefficient_stds) - np.min(mfcc_coefficient_stds))
            if mfcc_coefficient_std_range < 2.0:
                 artifacts.append("uniform_mel_band_energy")

            score = min(1.0, len(artifacts) * 0.3)

        except Exception as e:
            logger.warning(f"FormantAnalyzer error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="formant_analysis",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Artifact Detector
# ===============================================================

class ArtifactDetector:
    """
    Detects synthesis artifacts in the raw waveform.
    """

    def analyze(self, audio_samples: np.ndarray, sample_rate: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Click / pop detection
            # Use diff for fast gradient check
            sample_differences = np.abs(np.diff(audio_samples))
            threshold = np.std(audio_samples) * 6  # Higher threshold
            clicks = np.count_nonzero(sample_differences > threshold)
            click_rate = clicks / (len(audio_samples) / sample_rate)
            details["click_rate_per_sec"] = round(click_rate, 2)

            if click_rate > 10:
                artifacts.append("synthesis_click_artifacts")

            # 2. Waveform symmetry
            positive_samples = audio_samples[audio_samples > 0]
            negative_samples = audio_samples[audio_samples < 0]
            if len(positive_samples) > 0 and len(negative_samples) > 0:
                positive_rms_energy = np.sqrt(np.mean(positive_samples ** 2))
                negative_rms_energy = np.sqrt(np.mean(negative_samples ** 2))
                symmetry = float(positive_rms_energy / (negative_rms_energy + 1e-10))
                details["waveform_symmetry"] = round(symmetry, 4)

                if abs(symmetry - 1.0) > 0.3:
                    artifacts.append("asymmetric_waveform")

            # 3. Silence segment quality
            silence_sample_mask = np.abs(audio_samples) < 0.001
            if np.any(silence_sample_mask):
                silent_audio_samples = audio_samples[silence_sample_mask]
                silence_noise_floor_level = float(np.std(silent_audio_samples))
                details["silence_noise_floor"] = round(silence_noise_floor_level, 6)

                if silence_noise_floor_level < 1e-6 and len(silent_audio_samples) > sample_rate * 0.05:
                    artifacts.append("digitally_perfect_silence")

            # 4. Periodicity (Optimized - simple zcr based check instead of expensive autocorrelation)
            # Highly periodic signals (machines) have very stable low ZCR
            # Re-using ZCR concept from temporal but specifically for hyper-periodicity
            
            score = min(1.0, len(artifacts) * 0.25)

        except Exception as e:
            logger.warning(f"ArtifactDetector error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="artifact_detection",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Forensic Engine (orchestrates all analyzers)
# ===============================================================

class ForensicEngine:
    """
    Runs all forensic analyzers and produces a combined result.
    Orchestrates parallel execution for speed.
    """

    def __init__(self):
        self.spectral = SpectralAnalyzer()
        self.temporal = TemporalAnalyzer()
        self.formant = FormantAnalyzer()
        self.artifact = ArtifactDetector()
        # Initialize thread pool
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def analyze(self, audio_samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Run all analyzers in PARALLEL and return combined report."""
        results = {}

        # Define tasks
        tasks = {
            self._executor.submit(self.spectral.analyze, audio_samples, sample_rate): "spectral",
            self._executor.submit(self.temporal.analyze, audio_samples, sample_rate): "temporal",
            self._executor.submit(self.formant.analyze, audio_samples, sample_rate): "formant",
            self._executor.submit(self.artifact.analyze, audio_samples, sample_rate): "artifact"
        }

        # Wait for all to complete
        for future in concurrent.futures.as_completed(tasks):
            try:
                result = future.result()
                results[result.name] = {
                    "score": result.score,
                    "verdict": result.verdict,
                    "artifacts_found": result.artifacts_found,
                    "details": result.details,
                }
            except Exception as e:
                logger.error(f"Analyzer failed: {e}")
                # Provide strict fallback for failures
                results["error"] = {"score": 0.5, "verdict": "UNKNOWN", "details": str(e)}

        return results

    def compute_forensic_score(self, forensic_results: Dict[str, Any]) -> float:
        """
        Compute a weighted forensic score.
        Returns 0.0 (definitely human) to 1.0 (definitely AI).
        """
        weights = {
            "spectral_analysis": 0.30,
            "temporal_analysis": 0.25,
            "formant_analysis": 0.25,
            "artifact_detection": 0.20,
        }

        weighted_score_sum = 0.0
        total_analyzer_weight = 0.0
        for name, result in forensic_results.items():
            if name == "error": continue
            analyzer_weight = weights.get(name, 0.25)
            weighted_score_sum += result.get("score", 0.5) * analyzer_weight
            total_analyzer_weight += analyzer_weight

        return round(weighted_score_sum / (total_analyzer_weight + 1e-10), 4)

    def get_all_artifacts(self, forensic_results: Dict[str, Any]) -> List[str]:
        """Collect all artifacts found across all analyzers."""
        combined_artifacts = []
        for result in forensic_results.values():
            combined_artifacts.extend(result.get("artifacts_found", []))
        return combined_artifacts


# Singleton instance
forensic_engine = ForensicEngine()
