"""
Audio processing utilities optimized for M2 MacBook Pro
"""
import numpy as np
import librosa
import soundfile as sf
import torch
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)


def load_audio(
    audio_path: Union[str, Path], 
    sample_rate: int = 22050,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with proper error handling.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        normalize: Whether to normalize audio
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        
        if normalize:
            # Normalize to [-1, 1] range
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
        return audio, sr
        
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        raise


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 64,
    fmin: int = 0,
    fmax: Optional[int] = None
) -> np.ndarray:
    """
    Compute mel spectrogram with M2-optimized parameters.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        n_fft: FFT size (reduced for M2)
        hop_length: Hop length
        win_length: Window length  
        n_mels: Number of mel bins (reduced for M2)
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Log mel spectrogram
    """
    if fmax is None:
        fmax = sample_rate // 2
    
    try:
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1] range for training stability
        mel_spec = 2 * (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) - 1
        
        return mel_spec.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Failed to compute mel spectrogram: {e}")
        raise


def mel_to_audio(
    mel_spec: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_iter: int = 32
) -> np.ndarray:
    """
    Convert mel spectrogram back to audio using Griffin-Lim algorithm.
    This is used for validation before the vocoder is trained.
    
    Args:
        mel_spec: Mel spectrogram
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_iter: Number of Griffin-Lim iterations
        
    Returns:
        Audio signal
    """
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.cpu().numpy()
    
    # Denormalize mel spectrogram
    mel_spec = (mel_spec + 1) / 2  # [0, 1]
    
    try:
        # Convert log mel back to linear scale
        mel_spec_linear = librosa.db_to_power(mel_spec)
        
        # Use Griffin-Lim to reconstruct audio
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec_linear,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_iter=n_iter
        )
        
        # Normalize output
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        return audio.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Failed to convert mel to audio: {e}")
        raise


def save_audio(
    audio: Union[np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    sample_rate: int = 22050
) -> None:
    """
    Save audio to file.
    
    Args:
        audio: Audio signal
        output_path: Output file path
        sample_rate: Sample rate
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Ensure audio is 1D
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    try:
        sf.write(str(output_path), audio, sample_rate)
        logger.debug(f"Saved audio to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save audio to {output_path}: {e}")
        raise


class AudioProcessor:
    """Audio processing pipeline for M2 TTS training."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 64,
        fmin: int = 0,
        fmax: Optional[int] = None
    ):
        """
        Initialize audio processor with M2-optimized settings.
        
        Args:
            sample_rate: Target sample rate (reduced from 24kHz)
            n_fft: FFT size (reduced for memory)
            hop_length: Hop length
            win_length: Window length
            n_mels: Number of mel bins (reduced for memory)
            fmin: Minimum frequency
            fmax: Maximum frequency
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate // 2
        
        logger.info(f"AudioProcessor initialized with:")
        logger.info(f"  Sample rate: {self.sample_rate}")
        logger.info(f"  N_FFT: {self.n_fft}")
        logger.info(f"  Hop length: {self.hop_length}")
        logger.info(f"  N_mels: {self.n_mels}")
    
    def process_file(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single audio file to mel spectrogram.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio, mel_spectrogram)
        """
        # Load audio
        audio, _ = load_audio(audio_path, self.sample_rate)
        
        # Compute mel spectrogram
        mel_spec = compute_mel_spectrogram(
            audio=audio,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        return audio, mel_spec
    
    def mel_to_audio(self, mel_spec: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert mel spectrogram to audio using Griffin-Lim."""
        return mel_to_audio(
            mel_spec=mel_spec,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )


def validate_audio_config(config: dict) -> dict:
    """
    Validate and optimize audio configuration for M2 MacBook.
    
    Args:
        config: Audio configuration dictionary
        
    Returns:
        Validated configuration
    """
    # M2 optimization recommendations
    optimized_config = config.copy()
    
    # Memory-optimized defaults
    if optimized_config.get('n_fft', 2048) > 1024:
        logger.warning("Large n_fft detected, reducing to 1024 for M2 optimization")
        optimized_config['n_fft'] = 1024
        
    if optimized_config.get('n_mels', 80) > 64:
        logger.warning("Large n_mels detected, reducing to 64 for M2 optimization")
        optimized_config['n_mels'] = 64
        
    if optimized_config.get('sample_rate', 24000) > 22050:
        logger.warning("High sample rate detected, reducing to 22050 for M2 optimization")
        optimized_config['sample_rate'] = 22050
    
    return optimized_config