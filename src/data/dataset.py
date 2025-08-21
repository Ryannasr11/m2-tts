"""
Dataset implementation for M2 TTS training
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pickle
from tqdm import tqdm

from ..utils.audio import AudioProcessor
from ..utils.text import TextProcessor

logger = logging.getLogger(__name__)


class TTSDataset(Dataset):
    """Dataset for TTS training with M2 optimizations."""
    
    def __init__(
        self,
        data_dir: Path,
        subset_size: Optional[int] = None,
        max_text_length: int = 256,
        max_mel_length: int = 1000,
        sample_rate: int = 22050,
        n_mels: int = 64,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize TTS dataset.
        
        Args:
            data_dir: Directory containing audio and text files
            subset_size: Use only first N samples (for POC)
            max_text_length: Maximum text sequence length
            max_mel_length: Maximum mel spectrogram length
            sample_rate: Audio sample rate
            n_mels: Number of mel bins
            cache_dir: Directory to cache processed data
        """
        self.data_dir = Path(data_dir)
        self.subset_size = subset_size
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        self.text_processor = TextProcessor()
        
        # Setup cache
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Load data
        self.samples = self._load_samples()
        
        logger.info(f"TTSDataset initialized with {len(self.samples)} samples")
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load and process dataset samples."""
        
        # Check for cached data
        cache_file = self.cache_dir / "processed_samples.pkl" if self.cache_dir else None
        
        if cache_file and cache_file.exists():
            logger.info(f"Loading cached samples from {cache_file}")
            with open(cache_file, 'rb') as f:
                samples = pickle.load(f)
                
            if self.subset_size:
                samples = samples[:self.subset_size]
                
            return samples
        
        # Process samples from scratch
        samples = []
        
        # Look for metadata file (LJSpeech format)
        metadata_file = self.data_dir / "metadata.csv"
        
        if metadata_file.exists():
            samples = self._load_ljspeech_format()
        else:
            # Try to find audio/text pairs
            samples = self._load_paired_files()
            
        # Apply subset limit
        if self.subset_size:
            samples = samples[:self.subset_size]
            logger.info(f"Limited to {len(samples)} samples for subset training")
        
        # Cache processed samples
        if cache_file:
            logger.info(f"Caching processed samples to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
                
        return samples
    
    def _load_ljspeech_format(self) -> List[Dict[str, Any]]:
        """Load LJSpeech format dataset."""
        metadata_file = self.data_dir / "metadata.csv"
        wavs_dir = self.data_dir / "wavs"
        
        samples = []
        
        logger.info(f"Loading LJSpeech format from {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Processing LJSpeech"):
            parts = line.strip().split('|')
            if len(parts) >= 2:
                file_id = parts[0]
                text = parts[1] if len(parts) == 2 else parts[2]  # Handle normalized text
                
                audio_path = wavs_dir / f"{file_id}.wav"
                
                if audio_path.exists():
                    try:
                        sample = self._process_sample(audio_path, text)
                        if sample:
                            samples.append(sample)
                    except Exception as e:
                        logger.warning(f"Failed to process {audio_path}: {e}")
                        
        return samples
    
    def _load_paired_files(self) -> List[Dict[str, Any]]:
        """Load paired audio/text files."""
        samples = []
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.data_dir.glob(f"**/*{ext}"))
            
        logger.info(f"Found {len(audio_files)} audio files")
        
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            # Look for corresponding text file
            text_path = audio_path.with_suffix('.txt')
            
            if text_path.exists():
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    sample = self._process_sample(audio_path, text)
                    if sample:
                        samples.append(sample)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {audio_path}: {e}")
                    
        return samples
    
    def _process_sample(self, audio_path: Path, text: str) -> Optional[Dict[str, Any]]:
        """Process individual audio/text sample."""
        try:
            # Process audio
            audio, mel_spec = self.audio_processor.process_file(audio_path)
            
            # Check mel length constraints
            if mel_spec.shape[1] > self.max_mel_length:
                # Truncate mel spectrogram
                mel_spec = mel_spec[:, :self.max_mel_length]
                
            # Process text
            text_info = self.text_processor.process_text(text, self.max_text_length)
            
            # Basic duration alignment (simplified)
            mel_length = mel_spec.shape[1]
            text_length = text_info['length']
            
            if text_length > 0:
                avg_duration = mel_length / text_length
                durations = [avg_duration] * text_length
                
                # Pad or truncate to match text length
                if len(durations) < len(text_info['phoneme_ids']):
                    durations.extend([0.0] * (len(text_info['phoneme_ids']) - len(durations)))
                else:
                    durations = durations[:len(text_info['phoneme_ids'])]
            else:
                durations = [0.0] * len(text_info['phoneme_ids'])
            
            return {
                'audio_path': str(audio_path),
                'text': text,
                'phoneme_ids': text_info['phoneme_ids'],
                'phonemes': text_info['phonemes'],
                'text_length': len(text_info['phoneme_ids']),
                'mel_spec': mel_spec,
                'mel_length': mel_length,
                'durations': durations
            }
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        sample = self.samples[idx]
        
        # Convert to tensors
        return {
            'phoneme_ids': torch.LongTensor(sample['phoneme_ids']),
            'text_length': torch.LongTensor([sample['text_length']]),
            'mel_spec': torch.FloatTensor(sample['mel_spec']),
            'mel_length': torch.LongTensor([sample['mel_length']]), 
            'durations': torch.FloatTensor(sample['durations']),
            'text': sample['text'],  # Keep for debugging
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.
    
    Args:
        batch: List of dataset samples
        
    Returns:
        Batched data dictionary
    """
    # Get batch statistics
    batch_size = len(batch)
    max_text_len = max(item['phoneme_ids'].size(0) for item in batch)
    max_mel_len = max(item['mel_spec'].size(1) for item in batch)
    mel_dim = batch[0]['mel_spec'].size(0)
    
    # Initialize padded tensors
    phoneme_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    mel_specs = torch.zeros(batch_size, mel_dim, max_mel_len, dtype=torch.float)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)
    durations = torch.zeros(batch_size, max_text_len, dtype=torch.float)
    texts = []
    
    # Fill batched tensors
    for i, item in enumerate(batch):
        text_len = item['phoneme_ids'].size(0)
        mel_len = item['mel_spec'].size(1)
        
        phoneme_ids[i, :text_len] = item['phoneme_ids']
        text_lengths[i] = item['text_length']
        mel_specs[i, :, :mel_len] = item['mel_spec'] 
        mel_lengths[i] = item['mel_length']
        durations[i, :text_len] = item['durations']
        texts.append(item['text'])
    
    return {
        'phoneme_ids': phoneme_ids,
        'text_lengths': text_lengths,
        'mel_specs': mel_specs,
        'mel_lengths': mel_lengths,
        'durations': durations,
        'texts': texts
    }


def create_dataloader(
    dataset: TTSDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last: bool = True
) -> DataLoader:
    """Create DataLoader with M2 optimizations."""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else 2
    )


class DummyDataset(Dataset):
    """Dummy dataset for testing without real data."""
    
    def __init__(
        self,
        size: int = 100,
        max_text_length: int = 50,
        max_mel_length: int = 200,
        mel_dim: int = 64,
        vocab_size: int = 256
    ):
        """
        Create dummy dataset for testing.
        
        Args:
            size: Number of samples
            max_text_length: Maximum text sequence length
            max_mel_length: Maximum mel sequence length  
            mel_dim: Mel spectrogram dimensions
            vocab_size: Phoneme vocabulary size
        """
        self.size = size
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        self.mel_dim = mel_dim
        self.vocab_size = vocab_size
        
        logger.info(f"Created DummyDataset with {size} samples")
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate random sample."""
        # Random sequence lengths
        text_len = torch.randint(10, self.max_text_length, (1,)).item()
        mel_len = torch.randint(50, self.max_mel_length, (1,)).item()
        
        # Generate random data
        phoneme_ids = torch.randint(0, self.vocab_size, (text_len,))
        mel_spec = torch.randn(self.mel_dim, mel_len)
        
        # Generate reasonable durations
        total_duration = mel_len
        durations = torch.rand(text_len)
        durations = durations / durations.sum() * total_duration
        
        return {
            'phoneme_ids': phoneme_ids,
            'text_length': torch.LongTensor([text_len]),
            'mel_spec': mel_spec,
            'mel_length': torch.LongTensor([mel_len]),
            'durations': durations,
            'text': f"dummy_text_{idx}"
        }