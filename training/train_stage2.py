#!/usr/bin/env python3
"""
M2 TTS Model Training Script - Stage 2 (Quality Improvement)
Enhanced training with real data, improved losses, and evaluation
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import time
import numpy as np

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
from training.losses import CombinedTTSLoss, EarlyStopping
from evaluation.metrics import TTSEvaluator, benchmark_model_performance
from utils.device import setup_device, thermal_monitor, memory_tracker, clear_cache
from utils.audio import save_audio, AudioProcessor
from utils.text import TextProcessor

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


class M2TTSTrainerStage2:
    """Enhanced TTS Trainer for Stage 2 - Quality improvement."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with Stage 2 configuration."""
        self.config = config
        self.device = setup_device()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.setup_model()
        
        # Advanced loss function
        self.criterion = CombinedTTSLoss(
            mel_loss_weight=config.training.get('mel_loss_weight', 1.0),
            duration_loss_weight=config.training.get('duration_loss_weight', 0.1),
            adversarial_loss_weight=config.training.get('adversarial_loss_weight', 0.25),
            feature_matching_weight=config.training.get('feature_matching_weight', 2.0),
            spectral_loss_weight=config.training.get('spectral_loss_weight', 1.0),
            perceptual_loss_weight=config.training.get('perceptual_loss_weight', 0.5),
            use_discriminator=True
        )
        
        # Setup optimizers (separate for generator and discriminator)
        self.optimizer_g = self.setup_optimizer(self.model.parameters())
        self.optimizer_d = self.setup_optimizer(self.criterion.get_discriminator_parameters())
        
        self.scheduler_g = self.setup_scheduler(self.optimizer_g)
        self.scheduler_d = self.setup_scheduler(self.optimizer_d)
        
        # Setup data loaders
        self.train_loader, self.val_loader = self.setup_dataloaders()
        
        # Evaluation tools
        self.evaluator = TTSEvaluator(sample_rate=config.data.sample_rate)
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor(
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.get('patience', 10000),
            min_delta=config.training.get('min_delta', 0.001)
        )
        
        # Setup monitoring
        if config.system.get('wandb_project'):
            self.setup_wandb()
        
        logger.info(f"M2TTSTrainerStage2 initialized on {self.device}")
        
    def setup_directories(self):
        """Create necessary directories."""
        self.output_dir = Path(self.config.paths.output_dir)
        self.checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        self.log_dir = Path(self.config.paths.log_dir)
        self.samples_dir = Path(self.config.paths.get('samples_dir', self.output_dir / 'samples'))
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir, self.samples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_model(self) -> M2TTSModel:
        """Initialize and setup model with Stage 2 configuration."""
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
        logger.info(f"Stage 2 model loaded with {model_info['total_params']:,} parameters")
        logger.info(f"Estimated model size: {model_info['total_size_mb']:.1f} MB")
        
        # Log component breakdown
        for component, info in model_info['components'].items():
            logger.info(f"  {component}: {info['total']:,} params ({info['size_mb']:.1f} MB)")
        
        return model
    
    def setup_optimizer(self, parameters) -> optim.Optimizer:
        """Setup optimizer."""
        return optim.AdamW(
            parameters,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.get('weight_decay', 1e-6),
            betas=(0.8, 0.99)  # Optimized for TTS
        )
    
    def setup_scheduler(self, optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.training.get('lr_scheduler') == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_steps,
                eta_min=1e-6
            )
        return None
    
    def setup_dataloaders(self):
        """Setup train and validation data loaders."""
        # Training data
        data_dir = Path(self.config.data.data_dir)
        
        if data_dir.exists() and any(data_dir.iterdir()):
            logger.info(f"Loading real data from {data_dir}")
            train_dataset = TTSDataset(
                data_dir=data_dir,
                subset_size=self.config.data.get('subset_size'),
                max_text_length=256,
                max_mel_length=1000,
                sample_rate=self.config.data.sample_rate,
                n_mels=self.config.data.n_mels,
                cache_dir=self.output_dir / "cache"
            )
            
            # Create validation split (10% of data)
            total_size = len(train_dataset)
            val_size = min(max(total_size // 10, 1), 100)  # 10% but at least 1, max 100
            train_size = total_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
        else:
            logger.warning("No data found, using dummy dataset for testing")
            train_dataset = DummyDataset(
                size=self.config.data.get('subset_size', 1000),
                mel_dim=self.config.data.n_mels
            )
            val_dataset = DummyDataset(
                size=100,
                mel_dim=self.config.data.n_mels
            )
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=min(self.config.data.num_workers, 2),
            pin_memory=self.config.data.pin_memory,
            drop_last=False
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            wandb.init(
                project=self.config.system.wandb_project,
                name=f"m2-tts-stage2-{int(time.time())}",
                config=OmegaConf.to_container(self.config, resolve=True),
                tags=["stage2", "quality-improvement", "m2-optimized"]
            )
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(f"Failed to setup wandb: {e}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with generator and discriminator."""
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
        
        # Generate audio for loss computation
        mel_transposed = outputs['mel_output'].transpose(1, 2)
        audio_pred = self.model.vocoder(mel_transposed)
        
        # Target audio (convert from mel using Griffin-Lim as approximation)
        target_mel_np = batch['mel_specs'].cpu().numpy()
        target_audios = []
        for i in range(target_mel_np.shape[0]):
            target_audio = self.audio_processor.mel_to_audio(target_mel_np[i])
            target_audios.append(target_audio)
        
        audio_target = torch.FloatTensor(np.stack(target_audios)).unsqueeze(1).to(self.device)
        
        # Training discriminator every other step
        train_discriminator = (self.step % 2 == 0)
        
        if train_discriminator:
            # Train discriminator
            self.optimizer_d.zero_grad()
            
            loss_dict = self.criterion(
                mel_pred=outputs['mel_output'],
                mel_target=batch['mel_specs'],
                duration_pred=outputs['duration_pred'],
                duration_target=batch['durations'],
                audio_pred=audio_pred,
                audio_target=audio_target,
                mel_lengths=batch['mel_lengths'],
                optimize_discriminator=True
            )
            
            loss_dict['total_loss'].backward()
            
            if self.config.training.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.criterion.get_discriminator_parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer_d.step()
            if self.scheduler_d:
                self.scheduler_d.step()
                
        else:
            # Train generator
            self.optimizer_g.zero_grad()
            
            loss_dict = self.criterion(
                mel_pred=outputs['mel_output'],
                mel_target=batch['mel_specs'],
                duration_pred=outputs['duration_pred'],
                duration_target=batch['durations'],
                audio_pred=audio_pred,
                audio_target=audio_target,
                mel_lengths=batch['mel_lengths'],
                optimize_discriminator=False
            )
            
            # Handle gradient accumulation
            loss = loss_dict['total_loss']
            grad_accum_steps = self.config.training.get('gradient_accumulation_steps', 1)
            
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            
            loss.backward()
            
            # Only step optimizer every grad_accum_steps
            if (self.step + 1) % grad_accum_steps == 0:
                if self.config.training.get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                
                self.optimizer_g.step()
                if self.scheduler_g:
                    self.scheduler_g.step()
        
        # Return metrics
        metrics = {k: v.item() for k, v in loss_dict.items()}
        metrics['learning_rate_g'] = self.optimizer_g.param_groups[0]['lr']
        metrics['learning_rate_d'] = self.optimizer_d.param_groups[0]['lr']
        
        return metrics
    
    def validate_step(self) -> Dict[str, float]:
        """Validation step with comprehensive evaluation."""
        self.model.eval()
        
        val_metrics = {}
        
        with torch.no_grad():
            # Evaluate on validation set
            if len(self.val_loader) > 0:
                logger.info("Running validation evaluation...")
                val_metrics = benchmark_model_performance(
                    self.model, self.val_loader, self.device, num_samples=50
                )
        
        # Generate sample audio for predefined texts
        eval_texts = self.config.system.get('eval_texts', [
            "Hello world, this is a test of the improved model."
        ])
        
        for i, text in enumerate(eval_texts):
            try:
                # Process text
                text_data = self.text_processor.process_text(text, max_length=100)
                phoneme_ids = torch.LongTensor(text_data['phoneme_ids']).unsqueeze(0).to(self.device)
                phoneme_lengths = torch.LongTensor([text_data['length']]).to(self.device)
                
                # Generate audio
                mel_output, audio_output = self.model.inference(phoneme_ids, phoneme_lengths)
                
                # Save sample audio
                if audio_output is not None and audio_output.size(0) > 0:
                    audio_np = audio_output[0, 0].cpu().numpy()
                    sample_path = self.samples_dir / f"sample_step_{self.step}_text_{i}.wav"
                    save_audio(audio_np, sample_path, self.config.data.sample_rate)
                    
                    # Evaluate audio quality
                    audio_metrics = self.evaluator.estimate_mos_score(audio_np)
                    val_metrics[f'sample_{i}_mos'] = audio_metrics.get('estimated_mos', 0.0)
                    
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
        
        self.model.train()
        return val_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict() if self.scheduler_g else None,
            'scheduler_d_state_dict': self.scheduler_d.state_dict() if self.scheduler_d else None,
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
        max_checkpoints = self.config.training.get('max_checkpoints', 10)
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
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        if self.scheduler_g and checkpoint['scheduler_g_state_dict']:
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        if self.scheduler_d and checkpoint['scheduler_d_state_dict']:
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Loaded checkpoint from step {self.step}")
    
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
                    'mel': f"{loss_dict.get('mel_loss', 0):.4f}",
                    'dur': f"{loss_dict.get('duration_loss', 0):.4f}",
                    'lr_g': f"{loss_dict.get('learning_rate_g', 0):.2e}"
                })
                
                # Logging
                if self.step % self.config.system.log_every == 0:
                    self._log_metrics(loss_dict)
                
                # Validation
                if self.step % self.config.training.validate_every == 0:
                    val_metrics = self.validate_step()
                    self._log_metrics(val_metrics, prefix='val')
                    
                    # Early stopping check
                    if val_metrics and 'estimated_mos' in val_metrics:
                        val_loss = -val_metrics['estimated_mos']  # Negative because higher MOS is better
                        should_stop = self.early_stopping(val_loss)
                        if should_stop:
                            logger.info("Early stopping triggered")
                            return False
                
                # Generate samples
                if (self.step % self.config.system.get('generate_samples_every', 5000) == 0 
                    and self.step > 0):
                    self.validate_step()
                
                # Checkpointing  
                if self.step % self.config.training.save_every == 0:
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
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
        })
        
        # Log to wandb
        if wandb.run:
            wandb.log(prefixed_metrics, step=self.step)
        
        # Log to console (less frequent)
        if self.step % (self.config.system.log_every * 5) == 0:
            logger.info(f"Step {self.step}: " + 
                       " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))]))
    
    def train(self):
        """Main training loop."""
        logger.info("Starting Stage 2 training...")
        logger.info(f"Device: {self.device}")
        
        model_info = self.model.get_model_size()
        logger.info(f"Model parameters: {model_info['total_params']:,}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
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
    parser = argparse.ArgumentParser(description="M2 TTS Model Training - Stage 2")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/stage2_quality.yaml",
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
    log_dir = Path(config.paths.get('log_dir', 'outputs/stage2/logs'))
    setup_logging(
        log_level=config.system.get('log_level', 'INFO'),
        log_file=log_dir / "train_stage2.log"
    )
    
    logger.info(f"Loading Stage 2 configuration from {args.config}")
    logger.info("Stage 2 Focus: Quality improvement with real data training")
    
    # Create trainer
    trainer = M2TTSTrainerStage2(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()