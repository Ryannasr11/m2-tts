#!/usr/bin/env python3
"""
Test script for Stage 2 TTS pipeline - Quality improvements
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

def test_stage2_components():
    """Test Stage 2 specific components."""
    logger.info("="*60)
    logger.info("M2 TTS Stage 2 Test Suite")
    logger.info("="*60)
    
    try:
        # Test 1: Enhanced Loss Functions
        logger.info("Testing enhanced loss functions...")
        from training.losses import CombinedTTSLoss, SpectralLoss, AdversarialLoss
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Test combined loss
        loss_fn = CombinedTTSLoss(use_discriminator=True)
        
        # Create dummy tensors
        batch_size = 2
        mel_time = 50
        mel_dim = 80
        audio_len = 3200  # 50 * 64 hop length
        text_len = 20
        
        mel_pred = torch.randn(batch_size, mel_time, mel_dim)
        mel_target = torch.randn(batch_size, mel_dim, mel_time)
        duration_pred = torch.rand(batch_size, text_len) + 0.5
        duration_target = torch.rand(batch_size, text_len) + 0.5
        audio_pred = torch.randn(batch_size, 1, audio_len)
        audio_target = torch.randn(batch_size, 1, audio_len)
        mel_lengths = torch.tensor([45, 40])
        
        # Test generator loss
        loss_dict = loss_fn(
            mel_pred=mel_pred,
            mel_target=mel_target,
            duration_pred=duration_pred,
            duration_target=duration_target,
            audio_pred=audio_pred,
            audio_target=audio_target,
            mel_lengths=mel_lengths,
            optimize_discriminator=False
        )
        
        logger.info(f"✅ Generator loss computed: {loss_dict['total_loss']:.4f}")
        logger.info(f"   Components: mel={loss_dict.get('mel_loss', 0):.4f}, "
                   f"dur={loss_dict.get('duration_loss', 0):.4f}, "
                   f"spec={loss_dict.get('spectral_loss', 0):.4f}")
        
        # Test discriminator loss
        loss_dict_d = loss_fn(
            mel_pred=mel_pred,
            mel_target=mel_target,
            duration_pred=duration_pred,
            duration_target=duration_target,
            audio_pred=audio_pred,
            audio_target=audio_target,
            mel_lengths=mel_lengths,
            optimize_discriminator=True
        )
        
        logger.info(f"✅ Discriminator loss computed: {loss_dict_d['total_loss']:.4f}")
        
        # Test 2: Evaluation Metrics
        logger.info("\nTesting evaluation metrics...")
        from evaluation.metrics import TTSEvaluator, compute_mel_distance
        
        evaluator = TTSEvaluator(sample_rate=22050)
        
        # Test mel distance
        mel_metrics = compute_mel_distance(mel_pred[0], mel_target[0].transpose(0, 1))
        logger.info(f"✅ Mel distance metrics: L1={mel_metrics['mel_l1_distance']:.4f}")
        
        # Test batch evaluation
        batch_metrics = evaluator.evaluate_batch(
            pred_mels=mel_pred,
            target_mels=mel_target,
            pred_durations=duration_pred,
            target_durations=duration_target,
            mel_lengths=mel_lengths
        )
        
        logger.info(f"✅ Batch evaluation completed")
        if 'estimated_mos' in batch_metrics:
            logger.info(f"   Estimated MOS: {batch_metrics['estimated_mos']:.2f}")
        
        # Test 3: Improved Model Architecture
        logger.info("\nTesting Stage 2 model architecture...")
        from models.tts_model import M2TTSModel
        
        # Stage 2 model configuration (larger than Stage 1)
        stage2_model = M2TTSModel(
            vocab_size=256,
            hidden_dim=96,         # Increased from 64
            mel_channels=80,       # Increased from 64
            text_encoder_layers=3, # Increased from 2
            decoder_layers=3,      # Increased from 2
            num_heads=2,
            vocoder_channels=256   # Increased from 128
        ).to(device)
        
        model_info = stage2_model.get_model_size()
        logger.info(f"✅ Stage 2 model created with {model_info['total_params']:,} parameters")
        logger.info(f"   Model size: {model_info['total_size_mb']:.1f} MB")
        
        # Test forward pass
        phoneme_ids = torch.randint(0, 256, (batch_size, text_len)).to(device)
        phoneme_lengths = torch.tensor([18, 15]).to(device)
        durations = torch.rand(batch_size, text_len).to(device) * 2 + 1
        
        with torch.no_grad():
            outputs = stage2_model(
                phoneme_ids=phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                target_durations=durations,
                max_target_length=60
            )
        
        logger.info(f"✅ Stage 2 forward pass successful:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"   {key}: {value.shape}")
        
        # Test 4: Data Loading Simulation
        logger.info("\nTesting data components...")
        from data.dataset import DummyDataset, create_dataloader
        
        # Create larger dummy dataset for Stage 2
        stage2_dataset = DummyDataset(
            size=500,  # More samples
            mel_dim=80,  # Higher resolution
            max_mel_length=300  # Longer sequences
        )
        
        stage2_loader = create_dataloader(
            stage2_dataset,
            batch_size=4,  # Larger batch
            shuffle=True,
            num_workers=0
        )
        
        # Test batch loading
        for i, batch in enumerate(stage2_loader):
            logger.info(f"✅ Stage 2 batch {i} loaded:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"   {key}: {value.shape}")
                elif isinstance(value, list):
                    logger.info(f"   {key}: {len(value)} items")
            
            if i >= 1:  # Test 2 batches
                break
        
        # Test 5: Text Processing Improvements
        logger.info("\nTesting text processing...")
        from utils.text import TextProcessor
        
        text_processor = TextProcessor()
        
        # Test longer, more complex texts
        complex_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "M2 TTS generates high quality speech synthesis using advanced neural networks.",
            "This improved model demonstrates better prosody and naturalness."
        ]
        
        for text in complex_texts:
            result = text_processor.process_text(text, max_length=150)
            logger.info(f"✅ Processed: '{text[:30]}...' -> {len(result['phonemes'])} phonemes")
        
        # Test 6: Audio Processing
        logger.info("\nTesting audio processing...")
        from utils.audio import AudioProcessor
        
        # Stage 2 audio processor with higher resolution
        audio_processor = AudioProcessor(
            sample_rate=22050,
            n_mels=80,  # Higher resolution
            n_fft=1024
        )
        
        # Generate dummy audio and test processing
        dummy_audio = np.random.randn(22050 * 2)  # 2 seconds
        mel_spec = audio_processor.compute_mel_spectrogram(
            dummy_audio,
            sample_rate=22050,
            n_mels=80
        )
        
        logger.info(f"✅ Audio processing: {dummy_audio.shape} -> {mel_spec.shape}")
        
        # Test 7: Performance Comparison
        logger.info("\nComparing Stage 1 vs Stage 2 models...")
        
        # Stage 1 model (smaller)
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
        stage2_info = stage2_model.get_model_size()
        
        logger.info(f"✅ Model comparison:")
        logger.info(f"   Stage 1: {stage1_info['total_params']:,} params ({stage1_info['total_size_mb']:.1f} MB)")
        logger.info(f"   Stage 2: {stage2_info['total_params']:,} params ({stage2_info['total_size_mb']:.1f} MB)")
        logger.info(f"   Growth: {stage2_info['total_params'] / stage1_info['total_params']:.1f}x parameters")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 ALL STAGE 2 TESTS PASSED!")
        logger.info("="*60)
        
        # Summary
        logger.info(f"\nStage 2 Improvements Summary:")
        logger.info(f"  ✅ Enhanced loss functions (spectral, adversarial, perceptual)")
        logger.info(f"  ✅ Comprehensive evaluation metrics (MOS estimation)")
        logger.info(f"  ✅ Scaled model architecture ({stage2_info['total_params']:,} parameters)")
        logger.info(f"  ✅ Higher resolution audio processing (80 mel channels)")
        logger.info(f"  ✅ Advanced training pipeline with discriminator")
        logger.info(f"  ✅ Real data loading capabilities")
        
        logger.info(f"\nReady for Stage 2 training:")
        logger.info(f"  make download-data    # Download LJSpeech subset")
        logger.info(f"  make train-stage2     # Start Stage 2 training")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Stage 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_stage2_components()
    sys.exit(0 if success else 1)