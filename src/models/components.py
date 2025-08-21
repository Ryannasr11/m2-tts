"""
Core neural network components for M2-optimized TTS model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer layers."""
    
    def __init__(self, hidden_dim: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """Memory-efficient multi-head attention for M2."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Single linear layer for Q, K, V
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
        Returns:
            Attention output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Expand mask to match attention scores shape: [batch, heads, seq_len, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            scores.masked_fill_(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Memory-efficient feed-forward network."""
    
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with gradient checkpointing support."""
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        ffn_dim: int, 
        dropout: float = 0.1,
        use_checkpointing: bool = True
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward, x, mask, use_reentrant=False)
        else:
            return self._forward(x, mask)
    
    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class ConvBlock(nn.Module):
    """1D Convolutional block with normalization and activation."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
        Returns:
            Processed tensor [batch_size, out_channels, seq_len]
        """
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class LightweightResBlock(nn.Module):
    """Lightweight residual block for vocoder."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=self._get_padding(kernel_size, dilation),
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=self._get_padding(kernel_size, 1),
            dilation=1
        )
        
    def _get_padding(self, kernel_size: int, dilation: int) -> int:
        return (kernel_size - 1) * dilation // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = self.conv2(x)
        return x + residual


class VariancePredictor(nn.Module):
    """Predict variance information (duration, pitch, energy)."""
    
    def __init__(self, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout),
            ConvBlock(hidden_dim, hidden_dim, kernel_size, dropout)
        ])
        self.projection = nn.Conv1d(hidden_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, hidden_dim, seq_len]
        Returns:
            Predictions [batch_size, 1, seq_len]
        """
        for conv in self.conv_layers:
            x = conv(x)
        return self.projection(x)


def create_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Create padding mask for variable length sequences.
    
    Args:
        lengths: Sequence lengths [batch_size]
        max_length: Maximum sequence length
        
    Returns:
        Padding mask [batch_size, max_length]
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_length, device=lengths.device).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return mask


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to convolutional and linear layers."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(module)
    return module


class GradientClipping:
    """Gradient clipping utility for stable training."""
    
    def __init__(self, clip_value: float = 5.0):
        self.clip_value = clip_value
        
    def __call__(self, model: nn.Module) -> float:
        """Clip gradients and return norm."""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def initialize_weights(module: nn.Module) -> None:
    """Initialize weights for stable training."""
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv1d):
        torch.nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)