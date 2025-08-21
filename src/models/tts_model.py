"""
Main TTS model architecture - M2 MacBook optimized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, Any

from .components import (
    PositionalEncoding, TransformerEncoderLayer, VariancePredictor,
    LightweightResBlock, create_padding_mask, count_parameters,
    initialize_weights
)

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Lightweight text encoder for M2 MacBook."""
    
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Text embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads, 
                ffn_dim=hidden_dim * 2,  # Reduced from 4x for memory
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(initialize_weights)
        
        logger.info(f"TextEncoder: {hidden_dim}d, {num_layers} layers, {num_heads} heads")
        
    def forward(
        self, 
        phoneme_ids: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            phoneme_ids: [batch_size, seq_len]
            lengths: [batch_size] - actual sequence lengths
            
        Returns:
            Tuple of (encoded_text [batch, seq_len, hidden_dim], padding_mask)
        """
        batch_size, seq_len = phoneme_ids.shape
        
        # Create padding mask
        padding_mask = None
        if lengths is not None:
            padding_mask = create_padding_mask(lengths, seq_len)
        
        # Embed and add positional encoding
        x = self.embedding(phoneme_ids)  # [batch, seq_len, hidden_dim]
        x = x * (self.hidden_dim ** 0.5)  # Scale embeddings
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, padding_mask)
        
        x = self.norm(x)
        
        return x, padding_mask


class DurationPredictor(nn.Module):
    """Predict phoneme durations."""
    
    def __init__(self, hidden_dim: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.predictor = VariancePredictor(hidden_dim, kernel_size, dropout)
        
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch_size, seq_len, hidden_dim]
            
        Returns:
            Duration predictions [batch_size, seq_len]
        """
        # Transpose for conv1d: [batch, hidden_dim, seq_len]
        x = encoder_output.transpose(1, 2)
        
        # Predict durations
        duration_pred = self.predictor(x)  # [batch, 1, seq_len]
        
        # Remove channel dimension and apply activation
        duration_pred = duration_pred.squeeze(1)  # [batch, seq_len]
        duration_pred = F.softplus(duration_pred)  # Ensure positive
        
        return duration_pred


class LengthRegulator(nn.Module):
    """Regulate sequence length based on duration predictions."""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        encoder_output: torch.Tensor, 
        durations: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: [batch_size, seq_len, hidden_dim]
            durations: [batch_size, seq_len] - duration for each phoneme
            max_length: Maximum output length
            
        Returns:
            Length-regulated output [batch_size, regulated_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = encoder_output.shape
        device = encoder_output.device
        
        regulated_outputs = []
        
        for batch_idx in range(batch_size):
            output_seq = []
            
            for seq_idx in range(seq_len):
                duration = int(durations[batch_idx, seq_idx].item())
                if duration > 0:
                    # Repeat the encoder output 'duration' times
                    repeated = encoder_output[batch_idx, seq_idx].unsqueeze(0).repeat(duration, 1)
                    output_seq.append(repeated)
            
            if output_seq:
                output_seq = torch.cat(output_seq, dim=0)
            else:
                # Handle edge case where all durations are 0
                output_seq = torch.zeros(1, hidden_dim, device=device)
            
            regulated_outputs.append(output_seq)
        
        # Pad sequences to same length
        if max_length is None:
            max_length = max(seq.size(0) for seq in regulated_outputs)
        
        padded_outputs = []
        for seq in regulated_outputs:
            if seq.size(0) < max_length:
                padding = torch.zeros(max_length - seq.size(0), hidden_dim, device=device)
                seq = torch.cat([seq, padding], dim=0)
            elif seq.size(0) > max_length:
                seq = seq[:max_length]
            
            padded_outputs.append(seq)
        
        return torch.stack(padded_outputs, dim=0)


class MelDecoder(nn.Module):
    """Decode encoder output to mel spectrogram."""
    
    def __init__(
        self, 
        hidden_dim: int = 64,
        mel_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Decoder transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * 2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.mel_projection = nn.Linear(hidden_dim, mel_channels)
        
        # Initialize weights
        self.apply(initialize_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            
        Returns:
            Mel spectrogram [batch_size, seq_len, mel_channels]
        """
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Project to mel dimensions
        mel_output = self.mel_projection(x)
        
        return mel_output


class SimpleVocoder(nn.Module):
    """Lightweight vocoder for M2 MacBook."""
    
    def __init__(
        self,
        mel_channels: int = 64,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        n_layers: int = 4
    ):
        super().__init__()
        
        # Upsampling layers (simplified HiFi-GAN style)
        upsample_rates = [4, 4, 2, 2]  # Total upsample = 64x
        
        self.input_conv = nn.Conv1d(mel_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        
        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        current_channels = hidden_channels
        
        for rate in upsample_rates:
            # Upsample layer
            self.upsamples.append(
                nn.ConvTranspose1d(
                    current_channels, 
                    current_channels // 2,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2
                )
            )
            
            # Residual blocks
            current_channels = current_channels // 2
            self.resblocks.append(
                LightweightResBlock(current_channels, kernel_size)
            )
        
        # Output layer
        self.output_conv = nn.Conv1d(current_channels, 1, kernel_size, padding=kernel_size//2)
        
        # Initialize weights
        self.apply(initialize_weights)
        
        logger.info(f"SimpleVocoder: {mel_channels} -> 1, {len(upsample_rates)} upsample layers")
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [batch_size, mel_channels, time_steps]
            
        Returns:
            Audio waveform [batch_size, 1, audio_samples]
        """
        x = self.input_conv(mel)
        
        # Upsample and apply residual blocks
        for upsample, resblock in zip(self.upsamples, self.resblocks):
            x = F.leaky_relu(upsample(x), 0.1)
            x = resblock(x)
        
        # Generate final waveform
        x = torch.tanh(self.output_conv(x))
        
        return x


class M2TTSModel(nn.Module):
    """Complete M2 MacBook optimized TTS model."""
    
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 64,
        mel_channels: int = 64,
        text_encoder_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        vocoder_channels: int = 128
    ):
        super().__init__()
        
        # Model components
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=text_encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.duration_predictor = DurationPredictor(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.length_regulator = LengthRegulator()
        
        self.decoder = MelDecoder(
            hidden_dim=hidden_dim,
            mel_channels=mel_channels,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.vocoder = SimpleVocoder(
            mel_channels=mel_channels,
            hidden_channels=vocoder_channels
        )
        
        # Count parameters
        total_params, trainable_params = count_parameters(self)
        logger.info(f"M2TTSModel: {total_params:,} total parameters, {trainable_params:,} trainable")
        logger.info(f"Estimated model size: {total_params * 4 / (1024*1024):.1f} MB (FP32)")
        
    def forward(
        self, 
        phoneme_ids: torch.Tensor,
        phoneme_lengths: Optional[torch.Tensor] = None,
        target_durations: Optional[torch.Tensor] = None,
        max_target_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            phoneme_ids: [batch_size, seq_len]
            phoneme_lengths: [batch_size] - actual phoneme sequence lengths
            target_durations: [batch_size, seq_len] - target durations for training
            max_target_length: Maximum target sequence length
            
        Returns:
            Dictionary with model outputs
        """
        # Text encoding
        encoder_output, padding_mask = self.text_encoder(phoneme_ids, phoneme_lengths)
        
        # Duration prediction
        duration_pred = self.duration_predictor(encoder_output)
        
        # Use target durations during training, predictions during inference
        durations = target_durations if target_durations is not None else duration_pred
        
        # Length regulation
        regulated_output = self.length_regulator(
            encoder_output, durations, max_target_length
        )
        
        # Mel spectrogram generation
        mel_output = self.decoder(regulated_output)
        
        # Audio generation (optional, can be done separately for efficiency)
        audio_output = None
        if not self.training:  # Only generate audio during inference
            # Transpose mel for vocoder: [batch, mel_channels, time]
            mel_transposed = mel_output.transpose(1, 2)
            audio_output = self.vocoder(mel_transposed)
        
        return {
            'encoder_output': encoder_output,
            'duration_pred': duration_pred,
            'regulated_output': regulated_output,
            'mel_output': mel_output,
            'audio_output': audio_output,
            'padding_mask': padding_mask
        }
    
    def inference(
        self, 
        phoneme_ids: torch.Tensor,
        phoneme_lengths: Optional[torch.Tensor] = None,
        duration_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-only forward pass.
        
        Args:
            phoneme_ids: [batch_size, seq_len]
            phoneme_lengths: [batch_size] - actual sequence lengths
            duration_scale: Scale factor for duration predictions
            
        Returns:
            Tuple of (mel_spectrogram, audio_waveform)
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(phoneme_ids, phoneme_lengths)
            
            # Scale durations if needed
            if duration_scale != 1.0:
                scaled_durations = outputs['duration_pred'] * duration_scale
                regulated_output = self.length_regulator(
                    outputs['encoder_output'], scaled_durations
                )
                mel_output = self.decoder(regulated_output)
                outputs['mel_output'] = mel_output
            
            # Generate audio
            mel_transposed = outputs['mel_output'].transpose(1, 2)
            audio_output = self.vocoder(mel_transposed)
            
            return outputs['mel_output'], audio_output
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get detailed model size information."""
        component_params = {}
        
        for name, module in self.named_children():
            total_params, trainable_params = count_parameters(module)
            component_params[name] = {
                'total': total_params,
                'trainable': trainable_params,
                'size_mb': total_params * 4 / (1024 * 1024)  # FP32
            }
        
        total_params, trainable_params = count_parameters(self)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_size_mb': total_params * 4 / (1024 * 1024),
            'components': component_params
        }