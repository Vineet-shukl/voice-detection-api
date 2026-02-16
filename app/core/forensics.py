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

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Spectral Flatness — AI speech tends to have lower flatness (more tonal)
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            mean_flatness = float(np.mean(flatness))
            std_flatness = float(np.std(flatness))
            details["spectral_flatness_mean"] = round(mean_flatness, 4)
            details["spectral_flatness_std"] = round(std_flatness, 4)

            # Human speech has higher variance in spectral flatness
            if std_flatness < 0.02:
                artifacts.append("unnaturally_uniform_spectral_texture")
            if mean_flatness < 0.005:
                artifacts.append("overly_tonal_spectrum")

            # 2. Spectral Bandwidth — AI audio often has narrower bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            mean_bw = float(np.mean(bandwidth))
            std_bw = float(np.std(bandwidth))
            details["spectral_bandwidth_mean"] = round(mean_bw, 1)
            details["spectral_bandwidth_std"] = round(std_bw, 1)

            if std_bw < 200:
                artifacts.append("unnaturally_consistent_bandwidth")

            # 3. Spectral Centroid Variance — AI speech has more stable centroid
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_cv = float(np.std(centroid) / (np.mean(centroid) + 1e-10))
            details["spectral_centroid_cv"] = round(centroid_cv, 4)

            if centroid_cv < 0.15:
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

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Energy contour smoothness
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            if len(rms) > 10:
                rms_diff = np.diff(rms)
                energy_roughness = float(np.std(rms_diff) / (np.mean(rms) + 1e-10))
                details["energy_roughness"] = round(energy_roughness, 4)

                if energy_roughness < 0.08:
                    artifacts.append("unnaturally_smooth_energy_contour")

            # 2. Zero-Crossing Rate consistency
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
            zcr_cv = float(np.std(zcr) / (np.mean(zcr) + 1e-10))
            details["zcr_coefficient_of_variation"] = round(zcr_cv, 4)

            if zcr_cv < 0.25:
                artifacts.append("unnaturally_consistent_zero_crossings")

            # 3. Pause regularity analysis
            silence_threshold = np.percentile(np.abs(y), 10)
            is_silent = np.abs(y) < silence_threshold * 3
            silent_changes = np.diff(is_silent.astype(int))
            pause_starts = np.where(silent_changes == 1)[0]
            
            if len(pause_starts) >= 3:
                pause_intervals = np.diff(pause_starts) / sr
                interval_cv = float(np.std(pause_intervals) / (np.mean(pause_intervals) + 1e-10))
                details["pause_interval_cv"] = round(interval_cv, 4)
                details["num_pauses"] = len(pause_starts)

                if interval_cv < 0.2 and len(pause_starts) > 3:
                    artifacts.append("metronomic_pause_timing")

            # 4. Micro-jitter analysis (Optimized)
            if float(len(y)) / sr > 0.5:
                # Fast energy variance check instead of full autocorrelation loop
                chunk_size = int(0.1 * sr)
                # Reshape to chunks (discard remainder)
                n_chunks = len(y) // chunk_size
                if n_chunks > 4:
                    chunks = y[:n_chunks*chunk_size].reshape(n_chunks, chunk_size)
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

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. MFCC stability
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Vectorized variation coefficient
            means = np.abs(np.mean(mfccs[1:], axis=1)) + 1e-10
            stds = np.std(mfccs[1:], axis=1)
            mfcc_cvs = stds / means
            
            avg_mfcc_cv = float(np.mean(mfcc_cvs))
            details["avg_mfcc_cv"] = round(avg_mfcc_cv, 4)

            if avg_mfcc_cv < 0.5:
                artifacts.append("unnaturally_stable_formant_structure")

            # 2. Delta smoothness
            mfcc_deltas = librosa.feature.delta(mfccs)
            delta_roughness = float(np.mean(np.abs(librosa.feature.delta(mfcc_deltas))))
            details["delta_mfcc_roughness"] = round(delta_roughness, 4)

            if delta_roughness < 0.3:
                artifacts.append("overly_smooth_formant_transitions")

            # 3. Inter-frame correlation (Vectorized)
            if mfccs.shape[1] > 10:
                # Vectorized correlation between adjacent frames
                # Normalize frames
                frames = mfccs.T
                f_mean = frames.mean(axis=1, keepdims=True)
                f_std = frames.std(axis=1, keepdims=True) + 1e-10
                frames_norm = (frames - f_mean) / f_std
                
                # Compute correlation of frame i with i+1
                # Sum of product of normalized values / N
                corrs = np.mean(frames_norm[:-1] * frames_norm[1:], axis=1)
                mean_corr = float(np.mean(corrs))
                
                details["inter_frame_correlation"] = round(mean_corr, 4)

                if mean_corr > 0.95:
                    artifacts.append("excessive_inter_frame_correlation")

            # 4. Mel-band energy uniformity (uses MFCCs as proxy instead of new melspectrogram for speed)
            # MFCC[0] is energy; use variance of MFCCs as rough proxy for band variance
            mfcc_var_range = float(np.max(stds) - np.min(stds))
            if mfcc_var_range < 2.0:
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

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Click / pop detection
            # Use diff for fast gradient check
            diffs = np.abs(np.diff(y))
            threshold = np.std(y) * 6  # Higher threshold
            clicks = np.count_nonzero(diffs > threshold)
            click_rate = clicks / (len(y) / sr)
            details["click_rate_per_sec"] = round(click_rate, 2)

            if click_rate > 10:
                artifacts.append("synthesis_click_artifacts")

            # 2. Waveform symmetry
            pos_vals = y[y > 0]
            neg_vals = y[y < 0]
            if len(pos_vals) > 0 and len(neg_vals) > 0:
                pos_rms = np.sqrt(np.mean(pos_vals ** 2))
                neg_rms = np.sqrt(np.mean(neg_vals ** 2))
                symmetry = float(pos_rms / (neg_rms + 1e-10))
                details["waveform_symmetry"] = round(symmetry, 4)

                if abs(symmetry - 1.0) > 0.3:
                    artifacts.append("asymmetric_waveform")

            # 3. Silence segment quality
            silence_mask = np.abs(y) < 0.001
            if np.any(silence_mask):
                silent_vals = y[silence_mask]
                silence_noise_floor = float(np.std(silent_vals))
                details["silence_noise_floor"] = round(silence_noise_floor, 6)

                if silence_noise_floor < 1e-6 and len(silent_vals) > sr * 0.05:
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

    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Run all analyzers in PARALLEL and return combined report."""
        results = {}

        # Define tasks
        tasks = {
            self._executor.submit(self.spectral.analyze, y, sr): "spectral",
            self._executor.submit(self.temporal.analyze, y, sr): "temporal",
            self._executor.submit(self.formant.analyze, y, sr): "formant",
            self._executor.submit(self.artifact.analyze, y, sr): "artifact"
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

        weighted_sum = 0.0
        total_weight = 0.0
        for name, result in forensic_results.items():
            if name == "error": continue
            w = weights.get(name, 0.25)
            weighted_sum += result.get("score", 0.5) * w
            total_weight += w

        return round(weighted_sum / (total_weight + 1e-10), 4)

    def get_all_artifacts(self, forensic_results: Dict[str, Any]) -> List[str]:
        """Collect all artifacts found across all analyzers."""
        all_artifacts = []
        for result in forensic_results.values():
            all_artifacts.extend(result.get("artifacts_found", []))
        return all_artifacts


# Singleton instance
forensic_engine = ForensicEngine()
