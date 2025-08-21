#!/usr/bin/env python3
"""
Setup script for M2 TTS Model
"""
from pathlib import Path
from setuptools import setup, find_packages

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
readme_path = Path(__file__).parent / "README.md"
with open(readme_path, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="m2-tts-model",
    version="0.1.0",
    description="Lightweight Text-to-Speech model optimized for Apple M2 MacBook Pro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="M2 TTS Developer",
    author_email="developer@example.com",
    url="https://github.com/example/m2-tts-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0", 
            "isort>=5.12.0",
            "flake8>=6.0.0"
        ],
        "optional": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "coremltools>=7.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "m2-tts-train=training.train:main",
            "m2-tts-synthesize=scripts.synthesize:main",
            "m2-tts-test=scripts.test_pipeline:run_all_tests"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech"
    ],
    keywords="text-to-speech, tts, neural-networks, pytorch, apple-silicon, m2, macbook",
    project_urls={
        "Bug Reports": "https://github.com/example/m2-tts-model/issues",
        "Source": "https://github.com/example/m2-tts-model",
        "Documentation": "https://github.com/example/m2-tts-model/docs"
    }
)