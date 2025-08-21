# M2 TTS Model - Lightweight Text-to-Speech

A lightweight, CPU-optimized text-to-speech model designed specifically to be trainable on an Apple M2 MacBook Pro, inspired by KittenTTS architecture.

## Project Overview

This project implements a minimalist TTS model with the following characteristics:
- **Model Size**: 5-15M parameters (target <15MB)
- **Hardware**: Optimized for Apple M2 Pro/Max with MPS acceleration
- **Memory**: Designed for 16-32GB unified memory constraints
- **Quality Target**: 3.5-4.0 MOS score
- **Inference**: Real-time synthesis on CPU/Neural Engine

## Architecture

- **Text Encoder**: Lightweight transformer (4 layers, 128 hidden dims)
- **Duration Predictor**: Convolutional network for phoneme timing
- **VAE Decoder**: Variational autoencoder for mel-spectrogram generation
- **Vocoder**: Simplified HiFi-GAN for waveform synthesis

## Quick Start

```bash
# Setup environment
pip install -r requirements.txt

# Download sample data
python scripts/download_data.py --dataset ljspeech --subset 100

# Train minimal model
python training/train.py --config configs/stage1_poc.yaml

# Generate sample
python scripts/synthesize.py --text "Hello world" --checkpoint outputs/stage1_model.pt
```

## Hardware Requirements

### Minimum Configuration
- M2 MacBook Pro 14" (M2 Pro, 16GB RAM)
- 50GB available storage
- macOS 13.0+ with MPS support

## Performance Expectations

| Configuration | Training Speed | Model Quality | Training Time |
|---------------|---------------|---------------|---------------|
| M2 Pro (16GB) | 0.5 steps/sec | 3.5-3.7 MOS | 2-4 weeks |
| M2 Max (32GB) | 1-2 steps/sec | 3.7-4.0 MOS | 1-2 weeks |

## Project Structure

```
m2-tts-model/
├── src/                    # Core model implementation
│   ├── models/            # Neural network architectures
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training utilities
│   └── utils/             # Helper functions
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── training/              # Training scripts
├── evaluation/            # Quality assessment tools
├── scripts/               # Utility scripts
└── outputs/               # Generated models and samples
```

## License

Apache License 2.0

## Contributing

This is a research/educational project. Contributions welcome for:
- Memory optimization techniques
- M2-specific performance improvements  
- Quality enhancement methods
- Evaluation metrics and tools

## Acknowledgments

Inspired by:
- KittenTTS lightweight architecture
- VITS variational inference approach
- HiFi-GAN vocoder design
- Apple MPS PyTorch backend