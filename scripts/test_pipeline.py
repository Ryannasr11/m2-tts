#!/usr/bin/env python3
"""
Test script to validate the complete TTS pipeline
"""
import sys
import logging
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from models.tts_model import M2TTSModel
from data.dataset import DummyDataset, create_dataloader
from utils.device import setup_device, get_device_info
from utils.text import TextProcessor, create_phoneme_dict_file
from utils.audio import save_audio
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_device_setup():
    """Test M2 device setup."""
    logger.info("Testing device setup...")
    
    device = setup_device()
    device_info = get_device_info()
    
    logger.info(f"Device: {device}")
    logger.info(f"Device info: {device_info}")
    
    # Test tensor creation
    x = torch.randn(2, 3, 4).to(device)
    y = torch.randn(2, 3, 4).to(device)
    z = x + y
    
    logger.info(f"Tensor operation successful. Result shape: {z.shape}")
    return device


def test_text_processing():
    """Test text processing pipeline."""
    logger.info("Testing text processing...")
    
    text_processor = TextProcessor()
    
    test_texts = [
        "Hello world",
        "This is a test of the text to speech system",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    for text in test_texts:
        result = text_processor.process_text(text, max_length=100)
        logger.info(f"Text: '{text}'")
        logger.info(f"  Phonemes: {result['phonemes'][:10]}...")  # First 10
        logger.info(f"  IDs: {result['phoneme_ids'][:10]}...")    # First 10  
        logger.info(f"  Length: {result['length']}")
    
    return text_processor


def test_model_creation(device):
    """Test model creation and forward pass."""
    logger.info("Testing model creation...")
    
    # Create minimal model for testing
    model = M2TTSModel(
        vocab_size=256,
        hidden_dim=32,  # Very small for testing
        mel_channels=32,
        text_encoder_layers=1,
        decoder_layers=1, 
        num_heads=2,
        vocoder_channels=64
    ).to(device)
    
    # Print model info
    model_info = model.get_model_size()
    logger.info(f"Model info: {model_info}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    
    phoneme_ids = torch.randint(0, 256, (batch_size, seq_len)).to(device)
    phoneme_lengths = torch.tensor([8, 6]).to(device)  # Actual lengths
    durations = torch.rand(batch_size, seq_len).to(device) * 2 + 1  # 1-3 duration
    
    logger.info(f"Input shapes:")
    logger.info(f"  phoneme_ids: {phoneme_ids.shape}")
    logger.info(f"  phoneme_lengths: {phoneme_lengths.shape}")
    logger.info(f"  durations: {durations.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            phoneme_ids=phoneme_ids,
            phoneme_lengths=phoneme_lengths,
            target_durations=durations,
            max_target_length=50
        )
    
    logger.info(f"Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape}")
    
    return model


def test_inference(model, device, text_processor):
    """Test inference pipeline."""
    logger.info("Testing inference...")
    
    test_text = "Hello world, this is a test."
    
    # Process text
    text_data = text_processor.process_text(test_text, max_length=50)
    
    phoneme_ids = torch.LongTensor(text_data['phoneme_ids']).unsqueeze(0).to(device)
    phoneme_lengths = torch.LongTensor([text_data['length']]).to(device)
    
    logger.info(f"Inference input:")
    logger.info(f"  Text: '{test_text}'")
    logger.info(f"  Phoneme IDs shape: {phoneme_ids.shape}")
    logger.info(f"  Length: {phoneme_lengths.item()}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        mel_output, audio_output = model.inference(phoneme_ids, phoneme_lengths)
    
    logger.info(f"Inference output:")
    logger.info(f"  Mel shape: {mel_output.shape}")
    logger.info(f"  Audio shape: {audio_output.shape}")
    
    # Save test audio
    if audio_output is not None and audio_output.size(0) > 0:
        audio_np = audio_output[0, 0].cpu().numpy()
        output_path = Path("outputs") / "test_audio.wav"
        output_path.parent.mkdir(exist_ok=True)
        
        save_audio(audio_np, output_path, sample_rate=22050)
        logger.info(f"Saved test audio: {output_path}")
    
    return mel_output, audio_output


def test_dataset():
    """Test dataset creation and loading.""" 
    logger.info("Testing dataset...")
    
    # Create dummy dataset
    dataset = DummyDataset(size=10, mel_dim=32)
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset, 
        batch_size=2, 
        shuffle=False,
        num_workers=0  # Avoid multiprocessing for testing
    )
    
    # Test batch loading
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i} shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                logger.info(f"  {key}: {len(value)} items")
        
        if i >= 2:  # Only test a few batches
            break
    
    return dataset


def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    config_path = Path("configs/stage1_poc.yaml")
    
    if config_path.exists():
        config = OmegaConf.load(config_path)
        logger.info(f"Loaded config: {config.model.text_encoder.hidden_dim}d hidden")
        return config
    else:
        logger.warning(f"Config file not found: {config_path}")
        return None


def create_phoneme_dict():
    """Create phoneme dictionary file."""
    logger.info("Creating phoneme dictionary...")
    
    dict_path = Path("configs/phoneme_dict.txt")
    dict_path.parent.mkdir(exist_ok=True)
    
    create_phoneme_dict_file(dict_path)
    logger.info(f"Created phoneme dictionary: {dict_path}")


def run_all_tests():
    """Run all pipeline tests."""
    logger.info("="*60)
    logger.info("M2 TTS Pipeline Test Suite")
    logger.info("="*60)
    
    try:
        # Test 1: Device setup
        device = test_device_setup()
        logger.info("✅ Device setup test passed")
        
        # Test 2: Text processing
        text_processor = test_text_processing()
        logger.info("✅ Text processing test passed")
        
        # Test 3: Create phoneme dictionary
        create_phoneme_dict()
        logger.info("✅ Phoneme dictionary creation passed")
        
        # Test 4: Model creation
        model = test_model_creation(device)
        logger.info("✅ Model creation test passed")
        
        # Test 5: Inference
        mel_output, audio_output = test_inference(model, device, text_processor)
        logger.info("✅ Inference test passed")
        
        # Test 6: Dataset
        dataset = test_dataset()
        logger.info("✅ Dataset test passed")
        
        # Test 7: Config loading
        config = test_config_loading()
        if config:
            logger.info("✅ Config loading test passed")
        else:
            logger.info("⚠️ Config loading test skipped")
        
        logger.info("="*60)
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("="*60)
        
        # Print summary
        logger.info(f"Summary:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Model size: ~{sum(p.numel() for p in model.parameters()) * 4 / (1024*1024):.1f} MB")
        logger.info(f"  Test audio generated: outputs/test_audio.wav")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)