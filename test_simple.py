#!/usr/bin/env python3
"""
Simple test script for M2 TTS pipeline
"""
import sys
import os
import logging
from pathlib import Path
import torch

# Set up paths
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to src directory for relative imports
original_cwd = os.getcwd()
os.chdir(src_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality without complex imports."""
    logger.info("="*60)
    logger.info("M2 TTS Basic Test Suite")
    logger.info("="*60)
    
    try:
        # Test 1: PyTorch and MPS
        logger.info("Testing PyTorch and MPS support...")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("✅ MPS is available and working")
        else:
            device = torch.device("cpu")
            logger.info("⚠️ MPS not available, using CPU")
        
        # Test tensor operations
        x = torch.randn(2, 3, 4).to(device)
        y = torch.randn(2, 3, 4).to(device)
        z = torch.matmul(x, y.transpose(-2, -1))
        logger.info(f"✅ Tensor operations work on {device}")
        
        # Test 2: Basic model components
        logger.info("\nTesting basic neural network components...")
        from models.components import MultiHeadAttention, PositionalEncoding
        
        hidden_dim = 64
        seq_len = 10
        batch_size = 2
        
        # Test positional encoding
        pos_enc = PositionalEncoding(hidden_dim).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        x_pos = pos_enc(x)
        
        # Test attention
        attention = MultiHeadAttention(hidden_dim, num_heads=2).to(device)
        attn_out = attention(x_pos)
        
        logger.info(f"✅ Basic components work: input {x.shape} -> output {attn_out.shape}")
        
        # Test 3: Simple model creation
        logger.info("\nTesting minimal TTS model...")
        from models.tts_model import M2TTSModel
        
        model = M2TTSModel(
            vocab_size=100,
            hidden_dim=32,
            mel_channels=32,
            text_encoder_layers=1,
            decoder_layers=1,
            num_heads=2,
            vocoder_channels=64
        ).to(device)
        
        model_info = model.get_model_size()
        logger.info(f"✅ Model created with {model_info['total_params']:,} parameters")
        logger.info(f"   Estimated size: {model_info['total_size_mb']:.1f} MB")
        
        # Test 4: Forward pass
        logger.info("\nTesting model forward pass...")
        
        phoneme_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        phoneme_lengths = torch.tensor([8, 6]).to(device)
        durations = torch.rand(batch_size, seq_len).to(device) * 2 + 1
        
        model.eval()
        with torch.no_grad():
            outputs = model(
                phoneme_ids=phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                target_durations=durations,
                max_target_length=30
            )
        
        logger.info(f"✅ Forward pass successful:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"   {key}: {value.shape}")
        
        # Test 5: Text processing
        logger.info("\nTesting text processing...")
        from utils.text import TextProcessor
        
        text_processor = TextProcessor()
        test_text = "Hello world"
        result = text_processor.process_text(test_text, max_length=50)
        
        logger.info(f"✅ Text processing: '{test_text}' -> {len(result['phonemes'])} phonemes")
        
        # Test 6: Inference
        logger.info("\nTesting inference...")
        phoneme_ids = torch.LongTensor(result['phoneme_ids']).unsqueeze(0).to(device)
        phoneme_lengths = torch.LongTensor([result['length']]).to(device)
        
        with torch.no_grad():
            mel_output, audio_output = model.inference(phoneme_ids, phoneme_lengths)
        
        logger.info(f"✅ Inference successful:")
        logger.info(f"   Mel output: {mel_output.shape}")
        logger.info(f"   Audio output: {audio_output.shape if audio_output is not None else 'None'}")
        
        # Test 7: Audio saving (basic)
        if audio_output is not None:
            logger.info("\nTesting audio output...")
            from utils.audio import save_audio
            
            output_dir = project_root / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            audio_np = audio_output[0, 0].cpu().numpy()
            output_path = output_dir / "test_output.wav"
            
            save_audio(audio_np, output_path, sample_rate=22050)
            logger.info(f"✅ Audio saved to {output_path}")
            logger.info(f"   Duration: {len(audio_np) / 22050:.2f} seconds")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("="*60)
        
        logger.info(f"\nSummary:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model parameters: {model_info['total_params']:,}")
        logger.info(f"  Model size: {model_info['total_size_mb']:.1f} MB")
        logger.info(f"  Test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)