"""
Advanced loss functions for M2 TTS training - Stage 2 improvements
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SpectralLoss(nn.Module):
    """Multi-scale spectral loss for better audio quality."""
    
    def __init__(self, n_fft_list=[512, 1024, 2048], hop_length_factor=0.25):
        super().__init__()
        self.n_fft_list = n_fft_list
        self.hop_length_factor = hop_length_factor
        
    def stft_loss(self, pred_audio, target_audio, n_fft):
        """Compute STFT-based loss."""
        hop_length = int(n_fft * self.hop_length_factor)
        
        # Compute STFT
        pred_stft = torch.stft(
            pred_audio.squeeze(1), n_fft=n_fft, hop_length=hop_length,
            window=torch.hann_window(n_fft, device=pred_audio.device),
            return_complex=True
        )
        target_stft = torch.stft(
            target_audio.squeeze(1), n_fft=n_fft, hop_length=hop_length, 
            window=torch.hann_window(n_fft, device=target_audio.device),
            return_complex=True
        )
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        magnitude_loss = F.l1_loss(pred_mag, target_mag)
        
        # Phase loss (optional, can be computationally expensive)
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        phase_loss = F.l1_loss(pred_phase, target_phase) * 0.1  # Weighted down
        
        return magnitude_loss + phase_loss
    
    def forward(self, pred_audio, target_audio):
        """Compute multi-scale spectral loss."""
        total_loss = 0
        
        for n_fft in self.n_fft_list:
            total_loss += self.stft_loss(pred_audio, target_audio, n_fft)
            
        return total_loss / len(self.n_fft_list)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for adversarial training."""
    
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.discriminators = nn.ModuleList([
            self._make_discriminator() for _ in scales
        ])
        
    def _make_discriminator(self):
        """Create a single discriminator network."""
        return nn.Sequential(
            nn.Conv1d(1, 64, 15, 1, padding=7),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, 41, 4, padding=20, groups=4),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, 41, 4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, 41, 4, padding=20, groups=64),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(512, 1024, 41, 4, padding=20, groups=256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(1024, 1, 3, 1, padding=1),
        )
    
    def forward(self, x):
        """Forward pass through all discriminators."""
        outputs = []
        feature_maps = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                # Downsample for different scales
                x_scaled = F.avg_pool1d(x, kernel_size=self.scales[i], stride=self.scales[i])
            else:
                x_scaled = x
                
            features = []
            h = x_scaled
            
            # Get intermediate feature maps
            for layer in disc:
                h = layer(h)
                if isinstance(layer, nn.Conv1d) and h.size(1) > 1:
                    features.append(h)
            
            outputs.append(h)
            feature_maps.append(features)
            
        return outputs, feature_maps


class AdversarialLoss(nn.Module):
    """GAN-based adversarial loss."""
    
    def __init__(self):
        super().__init__()
        self.discriminator = MultiScaleDiscriminator()
        
    def discriminator_loss(self, real_audio, fake_audio):
        """Compute discriminator loss."""
        # Real audio
        real_outputs, _ = self.discriminator(real_audio)
        real_loss = 0
        for output in real_outputs:
            real_loss += F.mse_loss(output, torch.ones_like(output))
        
        # Fake audio
        fake_outputs, _ = self.discriminator(fake_audio.detach())
        fake_loss = 0
        for output in fake_outputs:
            fake_loss += F.mse_loss(output, torch.zeros_like(output))
            
        return (real_loss + fake_loss) / len(real_outputs)
    
    def generator_loss(self, fake_audio):
        """Compute generator loss."""
        fake_outputs, _ = self.discriminator(fake_audio)
        gen_loss = 0
        
        for output in fake_outputs:
            gen_loss += F.mse_loss(output, torch.ones_like(output))
            
        return gen_loss / len(fake_outputs)
    
    def feature_matching_loss(self, real_audio, fake_audio):
        """Compute feature matching loss."""
        _, real_features = self.discriminator(real_audio)
        _, fake_features = self.discriminator(fake_audio)
        
        fm_loss = 0
        for real_feats, fake_feats in zip(real_features, fake_features):
            for real_feat, fake_feat in zip(real_feats, fake_feats):
                fm_loss += F.l1_loss(fake_feat, real_feat)
                
        return fm_loss / (len(real_features) * len(real_features[0]))


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained features."""
    
    def __init__(self, model_type="mel"):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "mel":
            # Use mel-spectrogram features (lightweight)
            self.feature_extractor = self._mel_features
        else:
            # Could add other perceptual models here
            self.feature_extractor = self._mel_features
    
    def _mel_features(self, audio, n_mels=80):
        """Extract mel-spectrogram features."""
        # Simple mel-spec computation for perceptual comparison
        stft = torch.stft(
            audio.squeeze(1), n_fft=1024, hop_length=256,
            window=torch.hann_window(1024, device=audio.device),
            return_complex=True
        )
        magnitude = torch.abs(stft)  # [batch, freq_bins, time]
        
        # Create mel filter bank matrix [n_mels, freq_bins]
        freq_bins = magnitude.size(1)
        mel_filters = torch.linspace(0, 1, n_mels, device=audio.device).unsqueeze(1).expand(n_mels, freq_bins)
        mel_filters = mel_filters / (mel_filters.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
        
        # Apply mel filters: [batch, n_mels, time]
        mel_spec = torch.matmul(mel_filters.unsqueeze(0), magnitude)
        
        return torch.log(mel_spec + 1e-8)
    
    def forward(self, pred_audio, target_audio):
        """Compute perceptual loss."""
        pred_features = self.feature_extractor(pred_audio)
        target_features = self.feature_extractor(target_audio)
        
        return F.l1_loss(pred_features, target_features)


class CombinedTTSLoss(nn.Module):
    """Combined loss function for Stage 2 TTS training."""
    
    def __init__(
        self,
        mel_loss_weight: float = 1.0,
        duration_loss_weight: float = 0.1,
        adversarial_loss_weight: float = 0.25,
        feature_matching_weight: float = 2.0,
        spectral_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 0.5,
        use_discriminator: bool = True
    ):
        super().__init__()
        
        # Loss weights
        self.mel_loss_weight = mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_matching_weight = feature_matching_weight
        self.spectral_loss_weight = spectral_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        
        # Loss components
        self.spectral_loss = SpectralLoss()
        self.perceptual_loss = PerceptualLoss()
        
        if use_discriminator:
            self.adversarial_loss = AdversarialLoss()
        else:
            self.adversarial_loss = None
        
        self.training_step = 0
        
    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        audio_pred: Optional[torch.Tensor] = None,
        audio_target: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None,
        optimize_discriminator: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined TTS loss.
        
        Args:
            mel_pred: Predicted mel spectrogram [batch, time, mel_dim]
            mel_target: Target mel spectrogram [batch, mel_dim, time]
            duration_pred: Predicted durations [batch, text_len]
            duration_target: Target durations [batch, text_len]
            audio_pred: Predicted audio waveform [batch, 1, audio_len]
            audio_target: Target audio waveform [batch, 1, audio_len]
            mel_lengths: Actual mel lengths [batch]
            optimize_discriminator: Whether to compute discriminator loss
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Transpose mel target to match prediction: [batch, time, mel_dim]
        mel_target_transposed = mel_target.transpose(1, 2)
        
        # 1. Mel spectrogram loss (L1)
        if mel_lengths is not None:
            mel_loss = 0
            batch_size = mel_pred.size(0)
            for i in range(batch_size):
                mel_len = min(mel_lengths[i].item(), mel_pred.size(1), mel_target_transposed.size(1))
                pred_slice = mel_pred[i, :mel_len, :]
                target_slice = mel_target_transposed[i, :mel_len, :]
                mel_loss += F.l1_loss(pred_slice, target_slice)
            mel_loss = mel_loss / batch_size
        else:
            mel_loss = F.l1_loss(mel_pred, mel_target_transposed)
        
        losses['mel_loss'] = mel_loss
        
        # 2. Duration loss (MSE)
        duration_loss = F.mse_loss(duration_pred, duration_target)
        losses['duration_loss'] = duration_loss
        
        # 3. Audio-based losses (if audio is available)
        if audio_pred is not None and audio_target is not None:
            # Spectral loss
            spectral_loss = self.spectral_loss(audio_pred, audio_target)
            losses['spectral_loss'] = spectral_loss
            
            # Perceptual loss
            perceptual_loss = self.perceptual_loss(audio_pred, audio_target)
            losses['perceptual_loss'] = perceptual_loss
            
            # Adversarial losses
            if self.adversarial_loss is not None:
                if optimize_discriminator:
                    # Discriminator loss
                    disc_loss = self.adversarial_loss.discriminator_loss(audio_target, audio_pred)
                    losses['discriminator_loss'] = disc_loss
                else:
                    # Generator losses
                    gen_loss = self.adversarial_loss.generator_loss(audio_pred)
                    fm_loss = self.adversarial_loss.feature_matching_loss(audio_target, audio_pred)
                    
                    losses['generator_loss'] = gen_loss
                    losses['feature_matching_loss'] = fm_loss
        
        # Combine losses
        if optimize_discriminator and 'discriminator_loss' in losses:
            # Only discriminator loss when optimizing discriminator
            total_loss = losses['discriminator_loss']
        else:
            # Generator total loss
            total_loss = (
                self.mel_loss_weight * losses['mel_loss'] +
                self.duration_loss_weight * losses['duration_loss']
            )
            
            if 'spectral_loss' in losses:
                total_loss += self.spectral_loss_weight * losses['spectral_loss']
            
            if 'perceptual_loss' in losses:
                total_loss += self.perceptual_loss_weight * losses['perceptual_loss']
                
            if 'generator_loss' in losses:
                total_loss += self.adversarial_loss_weight * losses['generator_loss']
                
            if 'feature_matching_loss' in losses:
                total_loss += self.feature_matching_weight * losses['feature_matching_loss']
        
        losses['total_loss'] = total_loss
        self.training_step += 1
        
        return losses
    
    def get_discriminator_parameters(self):
        """Get discriminator parameters for separate optimization."""
        if self.adversarial_loss is not None:
            return self.adversarial_loss.discriminator.parameters()
        return []


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 10000, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        return self.wait >= self.patience