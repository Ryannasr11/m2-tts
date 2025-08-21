#!/usr/bin/env python3
"""
Data download script for M2 TTS training datasets
"""
import argparse
import logging
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract tar.bz2 or zip archive."""
    logger.info(f"Extracting {archive_path} to {extract_to}")
    
    if archive_path.suffix == '.bz2' or archive_path.name.endswith('.tar.bz2'):
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(extract_to)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_file:
            zip_file.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_ljspeech(data_dir: Path, subset_size: int = None):
    """Download LJSpeech dataset."""
    logger.info("Downloading LJSpeech dataset...")
    
    ljspeech_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    ljspeech_dir = data_dir / "ljspeech"
    archive_path = data_dir / "LJSpeech-1.1.tar.bz2"
    
    # Download if not exists
    if not archive_path.exists():
        download_file(ljspeech_url, archive_path, "Downloading LJSpeech")
    else:
        logger.info(f"Archive already exists: {archive_path}")
    
    # Extract if not exists
    extracted_dir = data_dir / "LJSpeech-1.1"
    if not extracted_dir.exists():
        extract_archive(archive_path, data_dir)
    else:
        logger.info(f"Dataset already extracted: {extracted_dir}")
    
    # Move to standard location
    if not ljspeech_dir.exists():
        extracted_dir.rename(ljspeech_dir)
        logger.info(f"Moved dataset to: {ljspeech_dir}")
    
    # Verify dataset
    metadata_file = ljspeech_dir / "metadata.csv"
    wavs_dir = ljspeech_dir / "wavs"
    
    if not metadata_file.exists() or not wavs_dir.exists():
        raise RuntimeError(f"Invalid LJSpeech dataset structure in {ljspeech_dir}")
    
    # Count samples
    with open(metadata_file, 'r') as f:
        total_samples = len(f.readlines())
    
    logger.info(f"LJSpeech dataset ready: {total_samples} samples")
    
    # Create subset if requested
    if subset_size and subset_size < total_samples:
        create_ljspeech_subset(ljspeech_dir, subset_size)
    
    # Clean up archive
    if archive_path.exists():
        archive_path.unlink()
        logger.info("Removed archive file")
    
    return ljspeech_dir


def create_ljspeech_subset(ljspeech_dir: Path, subset_size: int):
    """Create a subset of LJSpeech for faster training."""
    logger.info(f"Creating LJSpeech subset with {subset_size} samples...")
    
    subset_dir = ljspeech_dir.parent / f"ljspeech_subset_{subset_size}"
    subset_wavs = subset_dir / "wavs"
    subset_wavs.mkdir(parents=True, exist_ok=True)
    
    # Read original metadata
    metadata_file = ljspeech_dir / "metadata.csv"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Take first N samples
    subset_lines = lines[:subset_size]
    
    # Copy metadata
    subset_metadata = subset_dir / "metadata.csv"
    with open(subset_metadata, 'w', encoding='utf-8') as f:
        f.writelines(subset_lines)
    
    # Copy audio files
    logger.info("Copying audio files...")
    for line in tqdm(subset_lines, desc="Copying files"):
        file_id = line.split('|')[0]
        src_file = ljspeech_dir / "wavs" / f"{file_id}.wav"
        dst_file = subset_wavs / f"{file_id}.wav"
        
        if src_file.exists() and not dst_file.exists():
            import shutil
            shutil.copy2(src_file, dst_file)
    
    logger.info(f"Created subset dataset: {subset_dir}")
    return subset_dir


def download_vctk_subset(data_dir: Path, num_speakers: int = 10):
    """Download a subset of VCTK dataset."""
    logger.info(f"VCTK subset download with {num_speakers} speakers not implemented yet")
    logger.info("For Sprint 2, we'll focus on LJSpeech only")
    return None


def main():
    parser = argparse.ArgumentParser(description="Download TTS training datasets")
    parser.add_argument(
        "--dataset",
        choices=["ljspeech", "vctk"],
        default="ljspeech",
        help="Dataset to download"
    )
    parser.add_argument(
        "--subset",
        type=int,
        help="Create subset with N samples (for faster training)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store datasets"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    try:
        if args.dataset == "ljspeech":
            dataset_path = download_ljspeech(data_dir, args.subset)
            logger.info(f"✅ LJSpeech dataset ready at: {dataset_path}")
            
        elif args.dataset == "vctk":
            dataset_path = download_vctk_subset(data_dir, args.subset or 10)
            if dataset_path:
                logger.info(f"✅ VCTK subset ready at: {dataset_path}")
            else:
                logger.warning("VCTK download not implemented")
                
        logger.info("Dataset download completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Dataset download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()