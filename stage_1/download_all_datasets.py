#!/usr/bin/env python3
"""
Unified Dataset Downloader for Foundation Models vs Task-Specific Models Study

This script downloads all required datasets for the three tasks:
1. Classification: TrashNet (train) + RealWaste (test)
2. Detection: TACO (train) + Trash-ICRA19 (test)
3. Segmentation: TACO with masks (train) + BePLi v1 (test)

Usage:
    python download_all_datasets.py --task all
    python download_all_datasets.py --task classification
    python download_all_datasets.py --task detection
    python download_all_datasets.py --task segmentation
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Optional
import urllib.request
import json

# Try to import optional dependencies
try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("Warning: gdown not installed. Install with: pip install gdown")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATASETS_DIR = BASE_DIR / "datasets"

DATASET_CONFIG = {
    "classification": {
        "train": {
            "name": "TrashNet",
            "source": "huggingface",
            "repo_id": "garythung/trashnet",
            "output_dir": "classification/trashnet",
            "description": "Lab-isolated waste classification dataset (6 classes)"
        },
        "test": {
            "name": "RealWaste",
            "source": "uci",
            "url": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
            "output_dir": "classification/realwaste",
            "description": "Real-world waste images with cluttered backgrounds"
        }
    },
    "detection": {
        "train": {
            "name": "TACO",
            "source": "zenodo",
            "url": "https://zenodo.org/records/3587843/files/TACO.zip",
            "doi": "10.5281/zenodo.3587843",
            "output_dir": "detection/taco",
            "description": "Trash Annotations in Context - urban litter detection (2.7 GB)"
        },
        "test": {
            "name": "Trash-ICRA19",
            "source": "umn_conservancy",
            "url": "https://conservancy.umn.edu/bitstreams/0239b06a-512e-49c3-80aa-ba33371e11de/download",
            "doi": "10.13020/x0qn-y082",
            "output_dir": "detection/trash_icra19",
            "description": "Underwater marine debris detection dataset (5,700 images)"
        }
    },
    "segmentation": {
        "train": {
            "name": "TACO (Segmentation)",
            "source": "same_as_detection",
            "output_dir": "segmentation/taco_masks",
            "description": "TACO dataset with pixel-level masks"
        },
        "test": {
            "name": "BePLi v1",
            "source": "seanoe",
            "url": "https://www.seanoe.org/data/00811/92297/data/98753.zip",
            "doi": "10.17882/92297",
            "output_dir": "segmentation/bepli",
            "description": "Beach Plastic Litter segmentation dataset (3708 images with instance segmentation)"
        }
    }
}

# ============================================================================
# Download Utilities
# ============================================================================

class DownloadProgress:
    """Progress bar for downloads"""
    def __init__(self, total_size):
        self.total_size = total_size
        self.downloaded = 0
        
    def update(self, block_size):
        self.downloaded += block_size
        if self.total_size > 0:
            percent = (self.downloaded / self.total_size) * 100
            bar = '=' * int(percent // 2) + '>' + ' ' * (50 - int(percent // 2))
            print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file from URL with progress bar"""
    print(f"\n📥 Downloading: {description or url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get file size
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
        
        # Download with progress
        progress = DownloadProgress(total_size)
        
        def reporthook(block_num, block_size, total_size):
            progress.update(block_size)
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print(f"\n✅ Downloaded to: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip or tar archive"""
    print(f"📦 Extracting: {archive_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(output_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(output_dir)
        else:
            print(f"❌ Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"✅ Extracted to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def clone_git_repo(repo_url: str, output_dir: Path) -> bool:
    """Clone a git repository"""
    print(f"\n📥 Cloning repository: {repo_url}")
    
    if output_dir.exists():
        print(f"⚠️  Directory exists, pulling latest changes...")
        try:
            subprocess.run(['git', '-C', str(output_dir), 'pull'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Failed to pull, removing and re-cloning...")
            shutil.rmtree(output_dir)
    
    try:
        subprocess.run(['git', 'clone', repo_url, str(output_dir)], check=True)
        print(f"✅ Cloned to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Clone failed: {e}")
        return False


# ============================================================================
# Dataset-Specific Downloaders
# ============================================================================

def download_trashnet(output_dir: Path) -> bool:
    """Download TrashNet from Hugging Face"""
    if not HF_AVAILABLE:
        print("❌ huggingface_hub required. Install with: pip install huggingface_hub")
        return False
    
    print("\n" + "="*60)
    print("📥 Downloading TrashNet (Classification - Train)")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        path = snapshot_download(
            repo_id="garythung/trashnet",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ TrashNet downloaded to: {path}")
        
        # Extract if zipped
        for zip_file in output_dir.glob("*.zip"):
            extract_archive(zip_file, output_dir / zip_file.stem)
        
        return True
    except Exception as e:
        print(f"❌ Failed to download TrashNet: {e}")
        return False


def download_realwaste(output_dir: Path) -> bool:
    """Download RealWaste from UCI Repository"""
    print("\n" + "="*60)
    print("📥 Downloading RealWaste (Classification - Test)")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "realwaste.zip"
    
    # Try direct download first
    url = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
    
    if download_file(url, zip_path, "RealWaste dataset"):
        return extract_archive(zip_path, output_dir)
    
    # Fallback: check if already exists locally
    local_zip = BASE_DIR / "RealWaste" / "realwaste.zip"
    if local_zip.exists():
        print(f"Found local copy at: {local_zip}")
        shutil.copy(local_zip, zip_path)
        return extract_archive(zip_path, output_dir)
    
    print("❌ Please download RealWaste manually from:")
    print("   https://archive.ics.uci.edu/dataset/908/realwaste")
    return False


def download_taco(output_dir: Path) -> bool:
    """Download TACO dataset from Zenodo"""
    print("\n" + "="*60)
    print("📥 Downloading TACO (Detection - Train)")
    print("="*60)
    print("Source: Zenodo")
    print("DOI: 10.5281/zenodo.3587843")
    print("URL: https://zenodo.org/records/3587843")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct download link from Zenodo
    url = "https://zenodo.org/records/3587843/files/TACO.zip"
    zip_path = output_dir / "TACO.zip"
    
    # Check if already downloaded
    if zip_path.exists():
        print(f"⚠️  {zip_path.name} already exists")
        user_input = input("Re-download? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Skipping download, extracting existing file...")
            return extract_archive(zip_path, output_dir)
    
    # Try automated download
    print(f"\n📥 Downloading from Zenodo...")
    print(f"   File: TACO.zip (~2.7 GB)")
    print("   This may take a while...")
    
    if download_file(url, zip_path, "TACO Dataset"):
        success = extract_archive(zip_path, output_dir)
        if success:
            print("\n✅ TACO dataset downloaded and extracted successfully!")
            print(f"   Location: {output_dir}")
            print("\n📋 Dataset Info:")
            print("   - ~1,500 images of litter in the wild")
            print("   - 60 categories of litter")
            print("   - Segmentation masks (COCO format)")
            print("   - License: CC BY 4.0")
        return success
    
    # If automated download fails, provide manual instructions
    print("\n" + "="*60)
    print("⚠️  Automated download failed.")
    print("="*60)
    print("\n📋 Please download manually:")
    print("   1. Visit: https://zenodo.org/records/3587843")
    print("   2. Click 'Download' next to TACO.zip")
    print(f"   3. Save the file to: {zip_path}")
    print("   4. Re-run this script")
    print("\n📚 Citation:")
    print("   Pedro F. Proença, & Pedro Simões. (2019). TACO: Trash Annotations")
    print("   in Context Dataset. Zenodo. https://doi.org/10.5281/zenodo.3587843")
    
    return False


def download_trash_icra19(output_dir: Path) -> bool:
    """Download Trash-ICRA19 underwater debris dataset from UMN Conservancy"""
    print("\n" + "="*60)
    print("📥 Downloading Trash-ICRA19 (Detection - Test)")
    print("="*60)
    print("Source: University of Minnesota Data Repository (DRUM)")
    print("DOI: 10.13020/x0qn-y082")
    print("URL: https://conservancy.umn.edu/items/c34b2945-4052-48fa-b7e7-ce0fba2fe649")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct download link from UMN Conservancy
    url = "https://conservancy.umn.edu/bitstreams/0239b06a-512e-49c3-80aa-ba33371e11de/download"
    zip_path = output_dir / "trash_ICRA19.zip"
    
    # Check if already downloaded
    if zip_path.exists():
        print(f"⚠️  {zip_path.name} already exists")
        user_input = input("Re-download? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Skipping download, extracting existing file...")
            return extract_archive(zip_path, output_dir)
    
    # Try automated download
    print(f"\n📥 Downloading from UMN Conservancy...")
    print(f"   File: trash_ICRA19.zip (~980 MB)")
    
    if download_file(url, zip_path, "Trash-ICRA19 Dataset"):
        success = extract_archive(zip_path, output_dir)
        if success:
            print("\n✅ Trash-ICRA19 dataset downloaded and extracted successfully!")
            print(f"   Location: {output_dir}")
            print("\n📋 Dataset Info:")
            print("   - 5,700 underwater images")
            print("   - Bounding box annotations")
            print("   - Classes: trash, bio (biological), ROV")
            print("   - License: Free for academic use")
        return success
    
    # If automated download fails, provide manual instructions
    print("\n" + "="*60)
    print("⚠️  Automated download failed.")
    print("="*60)
    print("\n📋 Please download manually:")
    print("   1. Visit: https://conservancy.umn.edu/items/c34b2945-4052-48fa-b7e7-ce0fba2fe649")
    print("   2. Click on 'trash_ICRA19.zip' under 'View/Download File'")
    print(f"   3. Save the file to: {zip_path}")
    print("   4. Re-run this script")
    print("\n📚 Citation:")
    print("   Fulton, Hong, Sattar (2020). Trash-ICRA19: A Bounding Box Labeled")
    print("   Dataset of Underwater Trash. https://doi.org/10.13020/x0qn-y082")
    
    return False


def download_bepli(output_dir: Path) -> bool:
    """Download BePLi v1 beach plastic litter dataset from SEANOE"""
    print("\n" + "="*60)
    print("📥 Downloading BePLi v1 (Segmentation - Test)")
    print("="*60)
    print("Source: SEANOE (DOI: 10.17882/92297)")
    print("URL: https://www.seanoe.org/data/00811/92297/")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SEANOE direct download link for BePLi v1
    seanoe_url = "https://www.seanoe.org/data/00811/92297/data/98753.zip"
    zip_path = output_dir / "bepli_v1.zip"
    
    # Check if already downloaded
    if zip_path.exists():
        print(f"⚠️  {zip_path.name} already exists")
        user_input = input("Re-download? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Skipping download, extracting existing file...")
            return extract_archive(zip_path, output_dir)
    
    # Try automated download
    print(f"\n📥 Downloading from SEANOE...")
    print(f"   File: 98753.zip (~1-2 GB)")
    
    if download_file(seanoe_url, zip_path, "BePLi v1 Dataset"):
        success = extract_archive(zip_path, output_dir)
        if success:
            print("\n✅ BePLi v1 dataset downloaded and extracted successfully!")
            print(f"   Location: {output_dir}")
            print("\n📋 Dataset Info:")
            print("   - 3708 original beach images")
            print("   - Instance-based and pixel-level annotations")
            print("   - MSCOCO format annotations")
        return success
    
    # If automated download fails, provide manual instructions
    print("\n" + "="*60)
    print("⚠️  Automated download failed.")
    print("="*60)
    print("\n📋 Please download manually:")
    print("   1. Visit: https://www.seanoe.org/data/00811/92297/")
    print("   2. Click 'Télécharger' (Download) button next to '98753.zip'")
    print(f"   3. Save the file to: {zip_path}")
    print("   4. Re-run this script")
    print("\n📚 Citation:")
    print("   Hidaka et al. (2023). BePLi Dataset v1: Beach Plastic Litter Dataset version 1.")
    print("   SEANOE. https://doi.org/10.17882/92297")
    
    return False


# ============================================================================
# Main Download Functions
# ============================================================================

def download_classification_datasets() -> dict:
    """Download all classification datasets"""
    results = {}
    
    # TrashNet (Train)
    trashnet_dir = DATASETS_DIR / "classification" / "trashnet"
    results['trashnet'] = download_trashnet(trashnet_dir)
    
    # RealWaste (Test)
    realwaste_dir = DATASETS_DIR / "classification" / "realwaste"
    results['realwaste'] = download_realwaste(realwaste_dir)
    
    return results


def download_detection_datasets() -> dict:
    """Download all detection datasets"""
    results = {}
    
    # TACO (Train)
    taco_dir = DATASETS_DIR / "detection" / "taco"
    results['taco'] = download_taco(taco_dir)
    
    # Trash-ICRA19 (Test)
    icra_dir = DATASETS_DIR / "detection" / "trash_icra19"
    results['trash_icra19'] = download_trash_icra19(icra_dir)
    
    return results


def download_segmentation_datasets() -> dict:
    """Download all segmentation datasets"""
    results = {}
    
    # TACO Masks (Train) - use same as detection
    taco_seg_dir = DATASETS_DIR / "segmentation" / "taco_masks"
    detection_taco = DATASETS_DIR / "detection" / "taco"
    
    if detection_taco.exists():
        print("\n📋 Linking TACO for segmentation...")
        taco_seg_dir.parent.mkdir(parents=True, exist_ok=True)
        if not taco_seg_dir.exists():
            taco_seg_dir.symlink_to(detection_taco)
        results['taco_masks'] = True
    else:
        results['taco_masks'] = download_taco(taco_seg_dir)
    
    # BePLi (Test)
    bepli_dir = DATASETS_DIR / "segmentation" / "bepli"
    results['bepli'] = download_bepli(bepli_dir)
    
    return results


def download_all_datasets() -> dict:
    """Download all datasets for all tasks"""
    results = {
        'classification': download_classification_datasets(),
        'detection': download_detection_datasets(),
        'segmentation': download_segmentation_datasets()
    }
    return results


# ============================================================================
# Dataset Verification
# ============================================================================

def verify_datasets():
    """Verify all downloaded datasets"""
    print("\n" + "="*60)
    print("📊 Dataset Verification")
    print("="*60)
    
    status = []
    
    for task, datasets in DATASET_CONFIG.items():
        print(f"\n{task.upper()}:")
        for role, config in datasets.items():
            dataset_dir = DATASETS_DIR / config['output_dir']
            exists = dataset_dir.exists()
            
            if exists:
                # Count files
                n_files = sum(1 for _ in dataset_dir.rglob('*') if _.is_file())
                status_str = f"✅ {config['name']}: {n_files} files"
            else:
                status_str = f"❌ {config['name']}: Not found"
            
            print(f"  [{role.upper()}] {status_str}")
            status.append((config['name'], exists))
    
    # Summary
    n_ok = sum(1 for _, ok in status if ok)
    print(f"\n📊 Summary: {n_ok}/{len(status)} datasets ready")
    
    return status


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Foundation Models study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_all_datasets.py --task all
    python download_all_datasets.py --task classification
    python download_all_datasets.py --task detection
    python download_all_datasets.py --verify
        """
    )
    
    parser.add_argument(
        '--task',
        choices=['all', 'classification', 'detection', 'segmentation'],
        default='all',
        help='Which task datasets to download'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing datasets'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for datasets'
    )
    
    args = parser.parse_args()
    
    # Set custom output directory
    global DATASETS_DIR
    if args.output_dir:
        DATASETS_DIR = Path(args.output_dir)
    
    print("="*60)
    print("🗃️  Foundation Models Study - Dataset Downloader")
    print("="*60)
    print(f"Output directory: {DATASETS_DIR}")
    
    if args.verify:
        verify_datasets()
        return
    
    # Download datasets
    if args.task == 'all':
        results = download_all_datasets()
    elif args.task == 'classification':
        results = {'classification': download_classification_datasets()}
    elif args.task == 'detection':
        results = {'detection': download_detection_datasets()}
    elif args.task == 'segmentation':
        results = {'segmentation': download_segmentation_datasets()}
    
    # Verify after download
    print("\n" + "="*60)
    verify_datasets()
    
    print("\n✅ Download complete!")
    print("\nNext steps:")
    print("1. Verify all datasets are properly downloaded")
    print("2. Run preprocessing scripts if needed")
    print("3. Start training experiments")


if __name__ == "__main__":
    main()
