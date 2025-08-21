#!/usr/bin/env python3
"""
M2 TTS Model Training Script - Optimized for MacBook Pro
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.tts_model import M2TTSModel
from data.dataset import TTSDataset, DummyDataset, create_dataloader
from utils.device import setup_device, thermal_monitor, memory_tracker, clear_cache
from utils.audio import save_audio

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


class TTSLoss(nn.Module):
    """Combined loss function for TTS training."""
    
    def __init__(
        self,
        mel_loss_weight: float = 1.0,
        duration_loss_weight: float = 0.1
    ):
        super().__init__()
        self.mel_loss_weight = mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
        
    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        mel_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined TTS loss.
        
        Args:
            mel_pred: Predicted mel spectrogram [batch, time, mel_dim]
            mel_target: Target mel spectrogram [batch, mel_dim, time]
            duration_pred: Predicted durations [batch, text_len]
            duration_target: Target durations [batch, text_len]
            mel_lengths: Actual mel lengths [batch]
            
        Returns:
            Dictionary with loss components
        """
        # Transpose mel target to match prediction: [batch, time, mel_dim]
        mel_target = mel_target.transpose(1, 2)
        
        # Mel spectrogram loss (L1)
        mel_loss = 0
        batch_size = mel_pred.size(0)
        
        for i in range(batch_size):
            mel_len = mel_lengths[i].item()
            pred_slice = mel_pred[i, :mel_len, :]
            target_slice = mel_target[i, :mel_len, :]
            mel_loss += nn.functional.l1_loss(pred_slice, target_slice)
        
        mel_loss = mel_loss / batch_size
        
        # Duration loss (MSE)
        duration_loss = nn.functional.mse_loss(duration_pred, duration_target)
        
        # Combined loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                     self.duration_loss_weight * duration_loss)
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'duration_loss': duration_loss
        }


class M2TTSTrainer:
    """TTS Trainer optimized for M2 MacBook Pro."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = setup_device()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.setup_model()
        self.criterion = TTSLoss(
            mel_loss_weight=config.training.get('mel_loss_weight', 1.0),
            duration_loss_weight=config.training.get('duration_loss_weight', 0.1)
        )
        
        # Setup optimizer and scheduler
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        
        # Setup data loaders
        self.train_loader = self.setup_dataloader()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Setup monitoring
        if config.system.get('wandb_project'):
            self.setup_wandb()
        
        logger.info(f"M2TTSTrainer initialized on {self.device}")
        
    def setup_directories(self):
        """Create necessary directories."""
        self.output_dir = Path(self.config.paths.output_dir)
        self.checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        self.log_dir = Path(self.config.paths.log_dir)
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_model(self) -> M2TTSModel:
        """Initialize and setup model."""
        model = M2TTSModel(
            vocab_size=self.config.model.text_encoder.vocab_size,
            hidden_dim=self.config.model.text_encoder.hidden_dim,
            mel_channels=self.config.model.decoder.mel_channels,
            text_encoder_layers=self.config.model.text_encoder.num_layers,
            decoder_layers=self.config.model.decoder.get('num_layers', 2),
            num_heads=self.config.model.text_encoder.num_heads,
            dropout=self.config.model.text_encoder.dropout,
            vocoder_channels=self.config.model.vocoder.hidden_channels
        )
        
        model = model.to(self.device)
        
        # Log model info
        model_info = model.get_model_size()
        logger.info(f"Model loaded with {model_info['total_params']:,} parameters")
        logger.info(f"Estimated model size: {model_info['total_size_mb']:.1f} MB")
        
        return model
    
    def setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.get('weight_decay', 1e-6),
            betas=(0.9, 0.999)
        )
    
    def setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.training.get('lr_scheduler') == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_steps,
                eta_min=1e-6
            )
        return None
    
    def setup_dataloader(self):
        """Setup data loader."""
        # Use dummy data for POC if no real data available
        data_dir = Path(self.config.data.data_dir)
        
        if data_dir.exists() and any(data_dir.iterdir()):
            logger.info(f"Loading real data from {data_dir}")
            dataset = TTSDataset(
                data_dir=data_dir,
                subset_size=self.config.data.get('subset_size'),
                max_text_length=256,
                max_mel_length=1000,
                sample_rate=self.config.data.sample_rate,
                n_mels=self.config.data.n_mels,
                cache_dir=self.output_dir / "cache"
            )
        else:
            logger.warning("No data found, using dummy dataset for testing")
            dataset = DummyDataset(
                size=self.config.data.get('subset_size', 100),
                mel_dim=self.config.data.n_mels
            )
        
        return create_dataloader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True
        )
    
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            wandb.init(
                project=self.config.system.wandb_project,
                name=f"m2-tts-{int(time.time())}",
                config=OmegaConf.to_container(self.config, resolve=True)
            )
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(f"Failed to setup wandb: {e}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss {self.best_loss:.6f}")
        
        # Keep only recent checkpoints to save space
        max_checkpoints = self.config.training.get('max_checkpoints', 5)
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoint_files) > max_checkpoints:
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            for old_file in checkpoint_files[:-max_checkpoints]:
                old_file.unlink()
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Loaded checkpoint from step {self.step}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
        
        # Forward pass
        outputs = self.model(
            phoneme_ids=batch['phoneme_ids'],
            phoneme_lengths=batch['text_lengths'],
            target_durations=batch['durations'],
            max_target_length=batch['mel_specs'].size(2)
        )
        
        # Compute loss
        loss_dict = self.criterion(
            mel_pred=outputs['mel_output'],
            mel_target=batch['mel_specs'],
            duration_pred=outputs['duration_pred'],
            duration_target=batch['durations'],
            mel_lengths=batch['mel_lengths']
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        
        # Handle gradient accumulation
        loss = loss_dict['total_loss']
        grad_accum_steps = self.config.training.get('gradient_accumulation_steps', 1)
        
        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps
        
        loss.backward()
        
        # Gradient clipping
        if self.config.training.get('gradient_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_norm
            )
        
        # Only step optimizer every grad_accum_steps
        if (self.step + 1) % grad_accum_steps == 0:
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
        
        # Return metrics
        return {k: v.item() for k, v in loss_dict.items()}
    
    def validate_step(self) -> Dict[str, float]:
        """Validation step - generate sample audio."""
        self.model.eval()
        
        with torch.no_grad():
            # Use a simple test phrase
            from src.utils.text import TextProcessor
            
            text_processor = TextProcessor()
            text_data = text_processor.process_text("Hello world, this is a test.", max_length=50)
            
            phoneme_ids = torch.LongTensor(text_data['phoneme_ids']).unsqueeze(0).to(self.device)
            phoneme_lengths = torch.LongTensor([text_data['length']]).to(self.device)
            
            # Generate mel and audio
            mel_output, audio_output = self.model.inference(phoneme_ids, phoneme_lengths)
            
            # Save sample audio
            if audio_output is not None and audio_output.size(0) > 0:
                audio_np = audio_output[0, 0].cpu().numpy()  # [audio_samples]
                output_path = self.output_dir / f"sample_step_{self.step}.wav"
                save_audio(audio_np, output_path, self.config.data.sample_rate)
                logger.info(f"Saved sample audio: {output_path}")
        
        return {'validation_completed': 1.0}
    
    def train_epoch(self):
        """Train one epoch."""
        epoch_losses = []
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Check thermal state
            if not thermal_monitor.check_thermal_state():
                thermal_monitor.wait_for_cooldown()
            
            # Training step
            try:
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict['total_loss'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'mel': f"{loss_dict['mel_loss']:.4f}",
                    'dur': f"{loss_dict['duration_loss']:.4f}"
                })
                
                # Logging
                if self.step % self.config.system.log_every == 0:
                    self._log_metrics(loss_dict)
                
                # Validation
                if self.step % self.config.training.validate_every == 0:
                    val_metrics = self.validate_step()
                    self._log_metrics(val_metrics, prefix='val')
                
                # Checkpointing  
                if self.step % self.config.training.save_every == 0:
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    is_best = avg_loss < self.best_loss
                    
                    if is_best:
                        self.best_loss = avg_loss
                    
                    self.save_checkpoint(is_best=is_best)
                
                self.step += 1
                
                # Check max steps
                if self.step >= self.config.training.max_steps:
                    logger.info(f"Reached max steps ({self.config.training.max_steps})")
                    return False
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at step {self.step}, clearing cache and continuing")
                    clear_cache()
                    continue
                else:
                    raise e
            
            # Clear cache periodically
            if self.step % 100 == 0:
                clear_cache()
        
        return True
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = 'train'):
        """Log metrics to wandb and console."""
        # Add prefix to metrics
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Add system metrics
        mem_stats = memory_tracker.update()
        prefixed_metrics.update({
            f"{prefix}/memory_gb": mem_stats['current_gb'],
            f"{prefix}/memory_peak_gb": mem_stats['peak_gb'],
            f"{prefix}/step": self.step,
            f"{prefix}/epoch": self.epoch,
            f"{prefix}/lr": self.optimizer.param_groups[0]['lr']
        })
        
        # Log to wandb
        if wandb.run:
            wandb.log(prefixed_metrics, step=self.step)
        
        # Log to console (less frequent)
        if self.step % (self.config.system.log_every * 10) == 0:
            logger.info(f"Step {self.step}: " + 
                       " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Max steps: {self.config.training.max_steps}")
        
        try:
            while self.step < self.config.training.max_steps:
                continue_training = self.train_epoch()
                if not continue_training:
                    break
                
                self.epoch += 1
                
            logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
            
        finally:
            # Save final checkpoint
            self.save_checkpoint(is_best=False)
            
            if wandb.run:
                wandb.finish()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="M2 TTS Model Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/stage1_poc.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup logging
    log_dir = Path(config.paths.get('log_dir', 'outputs/logs'))
    setup_logging(
        log_level=config.system.get('log_level', 'INFO'),
        log_file=log_dir / "train.log"
    )
    
    logger.info(f"Loading configuration from {args.config}")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    # Create trainer
    trainer = M2TTSTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()