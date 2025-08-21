#!/usr/bin/env python3
"""
Simplified Stage 2 test script
"""
import sys
import os
import logging
from pathlib import Path
import torch
import numpy as np

# Set up paths
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to src directory for relative imports
original_cwd = os.getcwd()
os.chdir(src_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_stage2_core():
    """Test core Stage 2 functionality."""
    logger.info("="*60)
    logger.info("M2 TTS Stage 2 Core Test")
    logger.info("="*60)
    
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Test 1: Stage 2 Model Architecture
        logger.info("\n1. Testing Stage 2 model architecture...")
        from models.tts_model import M2TTSModel
        
        stage2_model = M2TTSModel(
            vocab_size=256,
            hidden_dim=96,         # Stage 2 size
            mel_channels=80,       # Higher resolution
            text_encoder_layers=3, # More layers
            decoder_layers=3,
            num_heads=2,
            vocoder_channels=256   # Larger vocoder
        ).to(device)
        
        model_info = stage2_model.get_model_size()
        logger.info(f"✅ Model: {model_info['total_params']:,} parameters ({model_info['total_size_mb']:.1f} MB)")
        
        # Test forward pass
        batch_size = 2
        seq_len = 15
        phoneme_ids = torch.randint(0, 256, (batch_size, seq_len)).to(device)
        phoneme_lengths = torch.tensor([12, 10]).to(device)
        durations = torch.rand(batch_size, seq_len).to(device) * 2 + 1
        
        with torch.no_grad():
            outputs = stage2_model(
                phoneme_ids=phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                target_durations=durations,
                max_target_length=50
            )
        
        logger.info(f"✅ Forward pass: mel {outputs['mel_output'].shape}, audio generated")
        
        # Test 2: Enhanced Loss Functions
        logger.info("\n2. Testing enhanced loss functions...")
        from training.losses import CombinedTTSLoss
        
        loss_fn = CombinedTTSLoss(
            mel_loss_weight=1.0,
            adversarial_loss_weight=0.25,
            feature_matching_weight=2.0,
            spectral_loss_weight=1.0,
            use_discriminator=True
        )
        
        # Create test data
        mel_pred = torch.randn(batch_size, 40, 80)
        mel_target = torch.randn(batch_size, 80, 40)
        duration_pred = torch.rand(batch_size, seq_len) + 0.5
        duration_target = torch.rand(batch_size, seq_len) + 0.5
        audio_pred = torch.randn(batch_size, 1, 2560)
        audio_target = torch.randn(batch_size, 1, 2560)
        
        # Test generator loss
        loss_dict = loss_fn(
            mel_pred=mel_pred,
            mel_target=mel_target,
            duration_pred=duration_pred,
            duration_target=duration_target,
            audio_pred=audio_pred,
            audio_target=audio_target,
            optimize_discriminator=False
        )
        
        logger.info(f"✅ Generator loss: {loss_dict['total_loss']:.4f}")
        logger.info(f"   Components: mel={loss_dict.get('mel_loss', 0):.3f}, "
                   f"spectral={loss_dict.get('spectral_loss', 0):.3f}, "
                   f"perceptual={loss_dict.get('perceptual_loss', 0):.3f}")
        
        # Test discriminator loss
        loss_dict_d = loss_fn(
            mel_pred=mel_pred,
            mel_target=mel_target,
            duration_pred=duration_pred,
            duration_target=duration_target,
            audio_pred=audio_pred,
            audio_target=audio_target,
            optimize_discriminator=True
        )
        
        logger.info(f"✅ Discriminator loss: {loss_dict_d['total_loss']:.4f}")
        
        # Test 3: Evaluation Metrics
        logger.info("\n3. Testing evaluation metrics...")
        from evaluation.metrics import TTSEvaluator
        
        evaluator = TTSEvaluator(sample_rate=22050)
        
        # Test evaluation
        metrics = evaluator.evaluate_sample(
            pred_mel=mel_pred[0],
            target_mel=mel_target[0].transpose(0, 1),
            pred_durations=duration_pred[0],
            target_durations=duration_target[0]
        )
        
        logger.info(f"✅ Evaluation metrics computed")
        logger.info(f"   Mel L1 distance: {metrics.get('mel_l1_distance', 0):.4f}")
        logger.info(f"   Duration correlation: {metrics.get('duration_correlation', 0):.4f}")
        
        # Test 4: Text Processing
        logger.info("\n4. Testing text processing...")
        from utils.text import TextProcessor
        
        text_processor = TextProcessor()
        
        test_texts = [
            "Hello world, this is an improved TTS model.",
            "The Stage 2 implementation includes better quality and evaluation.",
            "M2 optimized training enables high quality synthesis."
        ]
        
        for text in test_texts:
            result = text_processor.process_text(text, max_length=100)
            logger.info(f"✅ '{text[:30]}...' -> {len(result['phonemes'])} phonemes")
        
        # Test 5: Model Comparison
        logger.info("\n5. Comparing Stage 1 vs Stage 2...")
        
        stage1_model = M2TTSModel(
            vocab_size=256,
            hidden_dim=64,
            mel_channels=64,
            text_encoder_layers=2,
            decoder_layers=2,
            num_heads=2,
            vocoder_channels=128
        )
        
        stage1_info = stage1_model.get_model_size()
        
        logger.info(f"✅ Model comparison:")
        logger.info(f"   Stage 1: {stage1_info['total_params']:,} params ({stage1_info['total_size_mb']:.1f} MB)")
        logger.info(f"   Stage 2: {model_info['total_params']:,} params ({model_info['total_size_mb']:.1f} MB)")
        
        param_ratio = model_info['total_params'] / stage1_info['total_params']
        size_ratio = model_info['total_size_mb'] / stage1_info['total_size_mb']
        
        logger.info(f"   Growth: {param_ratio:.1f}x parameters, {size_ratio:.1f}x size")
        
        # Test 6: Inference Test
        logger.info("\n6. Testing inference...")
        stage2_model.eval()
        
        text_data = text_processor.process_text("Hello world, Stage 2 testing.", max_length=50)
        phoneme_ids = torch.LongTensor(text_data['phoneme_ids']).unsqueeze(0).to(device)
        phoneme_lengths = torch.LongTensor([text_data['length']]).to(device)
        
        with torch.no_grad():
            mel_output, audio_output = stage2_model.inference(phoneme_ids, phoneme_lengths)
        
        logger.info(f"✅ Inference: mel {mel_output.shape}, audio {audio_output.shape}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 STAGE 2 CORE TESTS PASSED!")
        logger.info("="*60)
        
        logger.info(f"\nStage 2 Ready for Training:")
        logger.info(f"  Model: {model_info['total_params']:,} parameters ({model_info['total_size_mb']:.1f} MB)")
        logger.info(f"  Quality: Enhanced losses (spectral, adversarial, perceptual)")
        logger.info(f"  Evaluation: Comprehensive metrics including MOS estimation")
        logger.info(f"  Architecture: {param_ratio:.1f}x larger than Stage 1")
        
        logger.info(f"\nNext Steps:")
        logger.info(f"  make download-data    # Download LJSpeech subset")
        logger.info(f"  make train-stage2     # Start Stage 2 training")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Stage 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_stage2_core()
    sys.exit(0 if success else 1)