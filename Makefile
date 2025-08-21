# M2 TTS Model Development Makefile

# Python interpreter
PYTHON := python3
PIP := pip3

# Directories
SRC_DIR := src
SCRIPTS_DIR := scripts
TRAINING_DIR := training
CONFIG_DIR := configs
OUTPUT_DIR := outputs

# Default target
.PHONY: help
help:
	@echo "M2 TTS Model Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  install          Install dependencies"
	@echo "  install-dev      Install with development dependencies"
	@echo "  setup            Full development setup"
	@echo ""
	@echo "Development:"
	@echo "  test            Run pipeline tests"
	@echo "  test-quick      Run quick pipeline test"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black and isort"
	@echo ""
	@echo "Training:"
	@echo "  train           Start training with default config"
	@echo "  train-poc       Start POC training (minimal)"
	@echo "  resume          Resume training from latest checkpoint"
	@echo ""
	@echo "Inference:"
	@echo "  synthesize      Generate sample audio"
	@echo "  demo           Generate demo audio samples"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean generated files"
	@echo "  clean-cache    Clean cache files only"
	@echo "  info           Show system information"

# Installation targets
.PHONY: install install-dev setup
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest black isort flake8

setup: install-dev
	@echo "Creating directories..."
	@mkdir -p $(OUTPUT_DIR)/{checkpoints,logs,cache,samples}
	@mkdir -p data
	@echo "Running initial tests..."
	@$(PYTHON) $(SCRIPTS_DIR)/test_pipeline.py

# Development targets
.PHONY: test test-quick lint format
test:
	$(PYTHON) $(SCRIPTS_DIR)/test_pipeline.py

test-quick:
	$(PYTHON) -c "from scripts.test_pipeline import test_device_setup, test_model_creation, setup_device; test_device_setup(); test_model_creation(setup_device())"

lint:
	flake8 $(SRC_DIR) $(SCRIPTS_DIR) $(TRAINING_DIR) --max-line-length=100 --ignore=E203,W503
	
format:
	black $(SRC_DIR) $(SCRIPTS_DIR) $(TRAINING_DIR) --line-length=100
	isort $(SRC_DIR) $(SCRIPTS_DIR) $(TRAINING_DIR) --profile=black

# Training targets  
.PHONY: train train-poc resume
train:
	$(PYTHON) $(TRAINING_DIR)/train.py --config $(CONFIG_DIR)/stage1_poc.yaml

train-poc:
	$(PYTHON) $(TRAINING_DIR)/train.py --config $(CONFIG_DIR)/stage1_poc.yaml

resume:
	@LATEST=$$(ls -t $(OUTPUT_DIR)/stage1/checkpoints/checkpoint_step_*.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Resuming from: $$LATEST"; \
		$(PYTHON) $(TRAINING_DIR)/train.py --config $(CONFIG_DIR)/stage1_poc.yaml --resume "$$LATEST"; \
	else \
		echo "No checkpoints found, starting fresh training"; \
		$(MAKE) train; \
	fi

# Inference targets
.PHONY: synthesize demo
synthesize:
	@if [ -z "$(TEXT)" ]; then \
		echo "Usage: make synthesize TEXT='Hello world'"; \
		exit 1; \
	fi
	@LATEST=$$(ls -t $(OUTPUT_DIR)/stage1/checkpoints/*.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		$(PYTHON) $(SCRIPTS_DIR)/synthesize.py --text "$(TEXT)" --checkpoint "$$LATEST" --output $(OUTPUT_DIR)/synthesized.wav; \
	else \
		echo "No trained model found. Run 'make train' first."; \
	fi

demo:
	@mkdir -p $(OUTPUT_DIR)/demo
	@LATEST=$$(ls -t $(OUTPUT_DIR)/stage1/checkpoints/*.pt 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Generating demo samples..."; \
		$(PYTHON) $(SCRIPTS_DIR)/synthesize.py --text "Hello world, this is M2 TTS" --checkpoint "$$LATEST" --output $(OUTPUT_DIR)/demo/hello.wav; \
		$(PYTHON) $(SCRIPTS_DIR)/synthesize.py --text "The quick brown fox jumps over the lazy dog" --checkpoint "$$LATEST" --output $(OUTPUT_DIR)/demo/pangram.wav; \
		$(PYTHON) $(SCRIPTS_DIR)/synthesize.py --text "This model runs efficiently on Apple Silicon" --checkpoint "$$LATEST" --output $(OUTPUT_DIR)/demo/apple_silicon.wav; \
		echo "Demo samples generated in $(OUTPUT_DIR)/demo/"; \
	else \
		echo "No trained model found. Run 'make train' first."; \
	fi

# Utility targets
.PHONY: clean clean-cache info
clean:
	@echo "Cleaning generated files..."
	@rm -rf $(OUTPUT_DIR)
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf *.egg-info
	@rm -rf .pytest_cache
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@echo "Clean complete"

clean-cache:
	@echo "Cleaning cache files..."
	@rm -rf $(OUTPUT_DIR)/*/cache
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@find . -name "*.pyc" -delete
	@echo "Cache clean complete"

info:
	@echo "System Information"
	@echo "=================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "PyTorch version: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "MPS available: $$($(PYTHON) -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo 'Unknown')"
	@echo "Working directory: $$(pwd)"
	@echo "Disk space: $$(df -h . | tail -1 | awk '{print $$4}') available"
	@echo ""
	@$(PYTHON) -c "from src.utils.device import get_device_info; import json; print('Device info:', json.dumps(get_device_info(), indent=2))" 2>/dev/null || echo "Device info not available"

# Data download (placeholder for future LJSpeech download)
.PHONY: download-data
download-data:
	@echo "Data download not implemented yet"
	@echo "For now, the system will use dummy data for testing"
	@echo "To use real data, download LJSpeech dataset to data/ljspeech/"