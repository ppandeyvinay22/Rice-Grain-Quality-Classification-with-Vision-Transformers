"""
GPU-Accelerated Dataset Preprocessing
Converts 112×224 images to 224×224 by rotation and stacking
Saves to new directory for fast training
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
import multiprocessing as mp
from functools import partial


def preprocess_single_image(args):
    """
    Preprocess a single image
    
    Args:
        args: Tuple of (src_path, dst_path)
    
    Returns:
        Success status
    """
    src_path, dst_path = args
    
    try:
        # Read image
        img = cv2.imread(src_path)
        if img is None:
            return False
        
        # Rotate 90 degrees clockwise
        img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        # Stack vertically to create 224×224
        img_224 = np.vstack([img_rotated, img_rotated])
        
        # Save
        cv2.imwrite(dst_path, img_224)
        return True
        
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def preprocess_dataset(config):
    """
    Preprocess entire dataset
    
    Args:
        config: Configuration dictionary
    """
    src_dir = config['dataset']['data_dir']
    
    # Create destination directory name
    src_path = Path(src_dir)
    dst_dir = str(src_path.parent / f"{src_path.name}_preprocessed")
    
    print("\n" + "="*80)
    print("DATASET PREPROCESSING")
    print("="*80)
    print(f"Source:      {src_dir}")
    print(f"Destination: {dst_dir}")
    print("="*80 + "\n")
    
    # Check if destination exists
    if os.path.exists(dst_dir):
        response = input(f"⚠️  Destination directory exists: {dst_dir}\nOverwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Preprocessing cancelled.")
            return
        print("Overwriting existing directory...\n")
    
    # Create destination directory structure
    os.makedirs(dst_dir, exist_ok=True)
    
    # Collect all image paths
    image_pairs = []
    total_images = 0
    
    print("Scanning directories...")
    for class_name in config['dataset']['classes']:
        src_class_dir = os.path.join(src_dir, class_name)
        dst_class_dir = os.path.join(dst_dir, class_name)
        
        if not os.path.exists(src_class_dir):
            print(f"⚠️  Warning: Class directory not found: {src_class_dir}")
            continue
        
        # Create destination class directory
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # Get all images in class
        images = [f for f in os.listdir(src_class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            src_path = os.path.join(src_class_dir, img_name)
            dst_path = os.path.join(dst_class_dir, img_name)
            image_pairs.append((src_path, dst_path))
        
        total_images += len(images)
        print(f"  {class_name}: {len(images)} images")
    
    print(f"\nTotal images to process: {total_images:,}")
    print("="*80 + "\n")
    
    # Determine number of workers
    num_workers = config['preprocess'].get('num_workers', mp.cpu_count())
    print(f"Using {num_workers} CPU workers for parallel processing\n")
    
    # Process images in parallel
    print("Processing images...")
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(preprocess_single_image, image_pairs),
            total=len(image_pairs),
            desc="Preprocessing",
            unit="img"
        ))
    
    # Count successes
    successful = sum(results)
    failed = len(results) - successful
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Successful: {successful:,} images")
    print(f"Failed:     {failed:,} images")
    print(f"Output:     {dst_dir}")
    print("="*80 + "\n")
    
    # Update config file suggestion
    print("📝 To use preprocessed data, update config.yaml:")
    print(f'   data_dir: "{dst_dir}"')
    print("\n")


def verify_preprocessing(config, num_samples=5):
    """
    Verify preprocessing by checking a few random images
    
    Args:
        config: Configuration dictionary
        num_samples: Number of samples to verify
    """
    src_dir = config['dataset']['data_dir']
    dst_dir = str(Path(src_dir).parent / f"{Path(src_dir).name}_preprocessed")
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    if not os.path.exists(dst_dir):
        print(f"❌ Preprocessed directory not found: {dst_dir}")
        return
    
    verified = 0
    for class_name in config['dataset']['classes'][:3]:  # Check first 3 classes
        dst_class_dir = os.path.join(dst_dir, class_name)
        if not os.path.exists(dst_class_dir):
            continue
        
        images = [f for f in os.listdir(dst_class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
        
        for img_name in images:
            img_path = os.path.join(dst_class_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                h, w = img.shape[:2]
                if h == 224 and w == 224:
                    verified += 1
                else:
                    print(f"⚠️  Wrong size: {img_name} is {w}×{h} (expected 224×224)")
    
    print(f"✅ Verified {verified} sample images (all 224×224)")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess rice grain dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify preprocessed images instead of preprocessing'
    )
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.verify:
        # Verify existing preprocessing
        verify_preprocessing(config)
    else:
        # Run preprocessing
        preprocess_dataset(config)
        
        # Auto-verify after preprocessing
        verify_preprocessing(config)