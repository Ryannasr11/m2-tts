"""
Evaluation metrics for TTS model quality assessment
"""
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_mel_distance(pred_mel: torch.Tensor, target_mel: torch.Tensor) -> float:
    """Compute mel-spectrogram distance (L1 and L2)."""
    l1_distance = F.l1_loss(pred_mel, target_mel).item()
    l2_distance = F.mse_loss(pred_mel, target_mel).item()
    
    return {
        'mel_l1_distance': l1_distance,
        'mel_l2_distance': l2_distance,
        'mel_combined_distance': l1_distance + np.sqrt(l2_distance)
    }


def compute_spectral_convergence(pred_audio: np.ndarray, target_audio: np.ndarray) -> float:
    """Compute spectral convergence between audio signals."""
    # STFT
    pred_stft = librosa.stft(pred_audio, n_fft=1024, hop_length=256)
    target_stft = librosa.stft(target_audio, n_fft=1024, hop_length=256)
    
    # Magnitude
    pred_mag = np.abs(pred_stft)
    target_mag = np.abs(target_stft)
    
    # Spectral convergence
    numerator = np.linalg.norm(target_mag - pred_mag, ord='fro')
    denominator = np.linalg.norm(target_mag, ord='fro')
    
    return numerator / (denominator + 1e-8)


def compute_log_spectral_distance(pred_audio: np.ndarray, target_audio: np.ndarray) -> float:
    """Compute log spectral distance."""
    # STFT
    pred_stft = librosa.stft(pred_audio, n_fft=1024, hop_length=256)
    target_stft = librosa.stft(target_audio, n_fft=1024, hop_length=256)
    
    # Log magnitude
    pred_log_mag = np.log(np.abs(pred_stft) + 1e-8)
    target_log_mag = np.log(np.abs(target_stft) + 1e-8)
    
    # LSD
    diff = pred_log_mag - target_log_mag
    lsd = np.sqrt(np.mean(diff ** 2))
    
    return lsd


def compute_mcd(pred_mel: np.ndarray, target_mel: np.ndarray) -> float:
    """Compute Mel-Cepstral Distortion (MCD)."""
    # Convert mel-spectrogram to MFCC-like coefficients
    pred_mfcc = librosa.feature.mfcc(S=pred_mel, n_mfcc=13)
    target_mfcc = librosa.feature.mfcc(S=target_mel, n_mfcc=13)
    
    # Align sequences (simple truncation)
    min_frames = min(pred_mfcc.shape[1], target_mfcc.shape[1])
    pred_mfcc = pred_mfcc[:, :min_frames]
    target_mfcc = target_mfcc[:, :min_frames]
    
    # MCD calculation
    diff = pred_mfcc - target_mfcc
    mcd = np.sqrt(np.sum(diff ** 2, axis=0))
    
    return float(np.mean(mcd))


def estimate_mos_score(
    pred_audio: np.ndarray, 
    target_audio: Optional[np.ndarray] = None,
    sample_rate: int = 22050
) -> Dict[str, float]:
    """
    Estimate MOS score using audio quality metrics.
    This is a simplified approximation - real MOS requires human evaluation.
    """
    scores = {}
    
    # Basic audio quality checks
    
    # 1. SNR estimation (if target available)
    if target_audio is not None:
        # Align lengths
        min_len = min(len(pred_audio), len(target_audio))
        pred_audio_aligned = pred_audio[:min_len]
        target_audio_aligned = target_audio[:min_len]
        
        # SNR
        noise = pred_audio_aligned - target_audio_aligned
        signal_power = np.mean(target_audio_aligned ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        scores['snr_db'] = float(snr)
        
        # Spectral convergence
        spec_conv = compute_spectral_convergence(pred_audio_aligned, target_audio_aligned)
        scores['spectral_convergence'] = float(spec_conv)
        
        # LSD
        lsd = compute_log_spectral_distance(pred_audio_aligned, target_audio_aligned)
        scores['log_spectral_distance'] = float(lsd)
        
    # 2. Audio statistics
    scores['rms_energy'] = float(np.sqrt(np.mean(pred_audio ** 2)))
    scores['zero_crossing_rate'] = float(np.mean(np.abs(np.diff(np.sign(pred_audio)))))
    
    # 3. Spectral features
    stft = librosa.stft(pred_audio, n_fft=1024, hop_length=256)
    magnitude = np.abs(stft)
    
    # Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)
    scores['spectral_centroid'] = float(np.mean(spectral_centroid))
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)
    scores['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
    
    # 4. Estimate MOS based on metrics
    # This is a rough approximation based on typical values
    if target_audio is not None:
        # Use reference-based metrics
        snr_score = np.clip((snr + 20) / 40, 0, 1)  # Normalize SNR
        spec_score = np.clip(1 - spec_conv, 0, 1)   # Invert spectral convergence
        lsd_score = np.clip(1 - lsd / 5, 0, 1)      # Normalize LSD
        
        estimated_mos = 1 + 4 * (0.4 * snr_score + 0.3 * spec_score + 0.3 * lsd_score)
    else:
        # Use non-reference metrics (less reliable)
        energy_score = np.clip(scores['rms_energy'] * 10, 0, 1)
        brightness_score = np.clip(scores['spectral_centroid'] / 3000, 0, 1)
        
        estimated_mos = 1 + 4 * (0.5 * energy_score + 0.5 * brightness_score)
    
    scores['estimated_mos'] = float(np.clip(estimated_mos, 1.0, 5.0))
    
    return scores


def compute_duration_accuracy(
    pred_durations: torch.Tensor, 
    target_durations: torch.Tensor
) -> Dict[str, float]:
    """Compute duration prediction accuracy metrics."""
    # L1 and L2 losses
    l1_loss = F.l1_loss(pred_durations, target_durations).item()
    l2_loss = F.mse_loss(pred_durations, target_durations).item()
    
    # Correlation
    pred_flat = pred_durations.flatten()
    target_flat = target_durations.flatten()
    
    if len(pred_flat) > 1:
        correlation = np.corrcoef(
            pred_flat.detach().cpu().numpy(),
            target_flat.detach().cpu().numpy()
        )[0, 1]
        correlation = float(correlation) if not np.isnan(correlation) else 0.0
    else:
        correlation = 0.0
    
    return {
        'duration_l1_loss': l1_loss,
        'duration_l2_loss': l2_loss,
        'duration_correlation': correlation
    }


class TTSEvaluator:
    """Comprehensive TTS model evaluator."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def evaluate_sample(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        pred_audio: Optional[torch.Tensor] = None,
        target_audio: Optional[torch.Tensor] = None,
        pred_durations: Optional[torch.Tensor] = None,
        target_durations: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate a single sample."""
        metrics = {}
        
        # Mel-spectrogram metrics
        mel_metrics = compute_mel_distance(pred_mel, target_mel)
        metrics.update(mel_metrics)
        
        # Audio metrics (if available)
        if pred_audio is not None and target_audio is not None:
            pred_audio_np = pred_audio.squeeze().detach().cpu().numpy()
            target_audio_np = target_audio.squeeze().detach().cpu().numpy()
            
            audio_metrics = estimate_mos_score(pred_audio_np, target_audio_np, self.sample_rate)
            metrics.update(audio_metrics)
            
        elif pred_audio is not None:
            pred_audio_np = pred_audio.squeeze().detach().cpu().numpy()
            audio_metrics = estimate_mos_score(pred_audio_np, sample_rate=self.sample_rate)
            metrics.update(audio_metrics)
        
        # Duration metrics (if available)
        if pred_durations is not None and target_durations is not None:
            duration_metrics = compute_duration_accuracy(pred_durations, target_durations)
            metrics.update(duration_metrics)
        
        return metrics
    
    def evaluate_batch(
        self,
        pred_mels: torch.Tensor,
        target_mels: torch.Tensor,
        pred_audios: Optional[torch.Tensor] = None,
        target_audios: Optional[torch.Tensor] = None,
        pred_durations: Optional[torch.Tensor] = None,
        target_durations: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of samples."""
        batch_size = pred_mels.size(0)
        batch_metrics = {}
        
        # Collect metrics for each sample
        all_sample_metrics = []
        
        for i in range(batch_size):
            # Get sample data
            pred_mel = pred_mels[i]
            target_mel = target_mels[i]
            
            # Truncate to actual length if available
            if mel_lengths is not None:
                mel_len = mel_lengths[i].item()
                pred_mel = pred_mel[:mel_len]
                target_mel = target_mel[:, :mel_len]  # Target is [mel_dim, time]
            
            pred_audio = pred_audios[i] if pred_audios is not None else None
            target_audio = target_audios[i] if target_audios is not None else None
            pred_duration = pred_durations[i] if pred_durations is not None else None
            target_duration = target_durations[i] if target_durations is not None else None
            
            # Evaluate sample
            sample_metrics = self.evaluate_sample(
                pred_mel.transpose(0, 1) if len(pred_mel.shape) == 2 else pred_mel,
                target_mel,
                pred_audio,
                target_audio,
                pred_duration,
                target_duration
            )
            
            all_sample_metrics.append(sample_metrics)
        
        # Average across batch
        if all_sample_metrics:
            for key in all_sample_metrics[0].keys():
                values = [m[key] for m in all_sample_metrics if key in m]
                if values:
                    batch_metrics[key] = np.mean(values)
        
        return batch_metrics
    
    def generate_evaluation_report(self, metrics: Dict[str, float]) -> str:
        """Generate a human-readable evaluation report."""
        report = "TTS Model Evaluation Report\n"
        report += "=" * 40 + "\n\n"
        
        # Quality Assessment
        if 'estimated_mos' in metrics:
            mos = metrics['estimated_mos']
            report += f"Overall Quality (Est. MOS): {mos:.2f}/5.0\n"
            
            if mos >= 4.0:
                quality = "Excellent"
            elif mos >= 3.5:
                quality = "Good"
            elif mos >= 3.0:
                quality = "Fair"
            else:
                quality = "Poor"
            
            report += f"Quality Rating: {quality}\n\n"
        
        # Detailed Metrics
        report += "Detailed Metrics:\n"
        report += "-" * 20 + "\n"
        
        for metric, value in sorted(metrics.items()):
            if isinstance(value, float):
                report += f"{metric}: {value:.4f}\n"
            else:
                report += f"{metric}: {value}\n"
        
        return report


def benchmark_model_performance(
    model,
    dataloader,
    device: torch.device,
    num_samples: int = 100
) -> Dict[str, float]:
    """Benchmark model performance on a dataset."""
    model.eval()
    evaluator = TTSEvaluator()
    
    all_metrics = []
    samples_processed = 0
    
    logger.info(f"Benchmarking model on {num_samples} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            # Model inference
            outputs = model(
                phoneme_ids=batch['phoneme_ids'],
                phoneme_lengths=batch['text_lengths'],
                target_durations=batch['durations'],
                max_target_length=batch['mel_specs'].size(2)
            )
            
            # Evaluate batch
            batch_metrics = evaluator.evaluate_batch(
                pred_mels=outputs['mel_output'],
                target_mels=batch['mel_specs'],
                pred_durations=outputs['duration_pred'],
                target_durations=batch['durations'],
                mel_lengths=batch['mel_lengths']
            )
            
            all_metrics.append(batch_metrics)
            samples_processed += batch['phoneme_ids'].size(0)
    
    # Average all metrics
    if all_metrics:
        final_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                final_metrics[key] = np.mean(values)
        
        return final_metrics
    
    return {}