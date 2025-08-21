#!/usr/bin/env python3
"""
Simple synthesis script for generating audio from text
"""
import sys
import argparse
import logging
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.tts_model import M2TTSModel
from utils.device import setup_device
from utils.text import TextProcessor
from utils.audio import save_audio
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> M2TTSModel:
    """Load trained model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')
    
    if config is None:
        # Use default config if not available in checkpoint
        logger.warning("No config found in checkpoint, using default")
        model = M2TTSModel()
    else:
        model = M2TTSModel(
            vocab_size=config.model.text_encoder.vocab_size,
            hidden_dim=config.model.text_encoder.hidden_dim,
            mel_channels=config.model.decoder.mel_channels,
            text_encoder_layers=config.model.text_encoder.num_layers,
            decoder_layers=config.model.decoder.get('num_layers', 2),
            num_heads=config.model.text_encoder.num_heads,
            dropout=config.model.text_encoder.dropout,
            vocoder_channels=config.model.vocoder.hidden_channels
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Training step: {checkpoint.get('step', 'unknown')}")
    
    return model


def synthesize_text(
    text: str, 
    model: M2TTSModel, 
    text_processor: TextProcessor,
    device: torch.device,
    duration_scale: float = 1.0
) -> tuple:
    """Synthesize audio from text."""
    
    # Process text
    text_data = text_processor.process_text(text, max_length=256)
    
    phoneme_ids = torch.LongTensor(text_data['phoneme_ids']).unsqueeze(0).to(device)
    phoneme_lengths = torch.LongTensor([text_data['length']]).to(device)
    
    logger.info(f"Synthesizing: '{text}'")
    logger.info(f"Phonemes: {' '.join(text_data['phonemes'][:20])}...")  # First 20
    logger.info(f"Sequence length: {text_data['length']}")
    
    # Generate audio
    with torch.no_grad():
        mel_output, audio_output = model.inference(
            phoneme_ids, 
            phoneme_lengths,
            duration_scale=duration_scale
        )
    
    logger.info(f"Generated mel shape: {mel_output.shape}")
    logger.info(f"Generated audio shape: {audio_output.shape}")
    
    return mel_output, audio_output


def main():
    """Main synthesis function."""
    parser = argparse.ArgumentParser(description="M2 TTS Text Synthesis")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Duration scaling factor (1.0 = normal speed)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, device)
    
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Synthesize
    mel_output, audio_output = synthesize_text(
        args.text,
        model,
        text_processor, 
        device,
        args.duration_scale
    )
    
    # Save audio
    if audio_output is not None and audio_output.size(0) > 0:
        audio_np = audio_output[0, 0].cpu().numpy()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_audio(audio_np, output_path, args.sample_rate)
        logger.info(f"Audio saved to: {output_path}")
        logger.info(f"Duration: {len(audio_np) / args.sample_rate:.2f} seconds")
    else:
        logger.error("No audio generated")


if __name__ == "__main__":
    main()