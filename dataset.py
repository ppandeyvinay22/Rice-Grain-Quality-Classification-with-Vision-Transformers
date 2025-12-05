"""
Rice Grain Dataset Class
Handles image loading, preprocessing (224x112 -> 224x224), and augmentation
OPTIMIZED VERSION with Enhanced Augmentations
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
import yaml


class RiceGrainDataset(Dataset):
    """
    Custom Dataset for Rice Grain Classification
    - Loads 224x224 preprocessed images
    - Applies augmentation
    """
    
    def __init__(self, 
                 root_dir: str, 
                 class_names: List[str],
                 transform=None,
                 is_train: bool = True):
        """
        Args:
            root_dir: Path to dataset root
            class_names: List of class folder names
            transform: Augmentation transforms
            is_train: Training or validation mode
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.is_train = is_train
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # Load all image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"{'Train' if is_train else 'Val'} Dataset: {len(self.samples)} images from {len(class_names)} classes")
    
    def _load_samples(self):
        """Load all image paths and labels"""
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all images in class folder
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label
        
        Returns:
            image: Tensor of shape (3, 224, 224)
            label: Class index
        """
        img_path, label = self.samples[idx]
        
        # Load image (already 224×224 from preprocessing)
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(config: Dict, is_train: bool = True):
    """
    Get augmentation transforms based on config
    ENHANCED with perspective and sharpness augmentations
    
    Args:
        config: Configuration dictionary
        is_train: Training or validation mode
        
    Returns:
        Composed transforms
    """
    if is_train:
        aug_config = config['augmentation']['train']
        transform_list = [
            transforms.RandomRotation(aug_config['random_rotation']),
            transforms.RandomHorizontalFlip(aug_config['random_horizontal_flip']),
            transforms.RandomVerticalFlip(aug_config['random_vertical_flip']),
            transforms.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter'].get('hue', 0)
            ),
            transforms.RandomAffine(
                degrees=aug_config['random_affine']['degrees'],
                translate=tuple(aug_config['random_affine']['translate']),
                scale=tuple(aug_config['random_affine']['scale'])
            ),
        ]
        
        # Add Gaussian Blur if configured
        if 'gaussian_blur' in aug_config:
            from torchvision.transforms import GaussianBlur
            transform_list.append(
                GaussianBlur(
                    kernel_size=aug_config['gaussian_blur']['kernel_size'],
                    sigma=tuple(aug_config['gaussian_blur']['sigma'])
                )
            )
        
        # NEW: Add Random Perspective if configured
        if 'random_perspective' in aug_config:
            transform_list.append(
                transforms.RandomPerspective(
                    distortion_scale=aug_config['random_perspective']['distortion_scale'],
                    p=aug_config['random_perspective']['p']
                )
            )
        
        # NEW: Add Random Adjust Sharpness if configured
        if 'random_adjust_sharpness' in aug_config:
            transform_list.append(
                transforms.RandomAdjustSharpness(
                    sharpness_factor=aug_config['random_adjust_sharpness']['sharpness_factor'],
                    p=aug_config['random_adjust_sharpness']['p']
                )
            )
        
        # Add ToTensor and Normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Add random noise after normalization if configured
        if 'random_noise' in aug_config:
            noise_std = aug_config['random_noise']
            transform_list.append(
                transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_std)
            )
    else:
        # Validation: only normalize
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)


def calculate_class_weights(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for balanced training
    
    Formula: weight = total_samples / (num_classes * samples_per_class)
    
    Args:
        dataset: Training dataset
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Calculate weights
    total_samples = len(dataset)
    class_weights = total_samples / (num_classes * class_counts)
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION & WEIGHTS")
    print("="*60)
    for idx, (count, weight) in enumerate(zip(class_counts, class_weights)):
        class_name = dataset.class_names[idx]
        print(f"{class_name:10s}: {int(count):6d} images | Weight: {weight:.3f}")
    print("="*60 + "\n")
    
    return class_weights


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Create train and validation dataloaders with class weights
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader, class_weights
    """
    # Load config
    data_dir = config['dataset']['data_dir']
    class_names = config['dataset']['classes']
    batch_size = config['training']['batch_size']
    num_workers = config['gpu']['num_workers']
    pin_memory = config['gpu']['pin_memory']
    seed = config['data_split']['random_seed']
    train_ratio = config['data_split']['train_ratio']
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get transforms
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    # Create full dataset
    full_dataset = RiceGrainDataset(
        root_dir=data_dir,
        class_names=class_names,
        transform=None,  # Will be applied per split
        is_train=True
    )
    
    # Split into train and val
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transforms to splits
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    print(f"\nDataset Split:")
    print(f"  Training:   {train_size:,} images ({train_ratio*100:.0f}%)")
    print(f"  Validation: {val_size:,} images ({(1-train_ratio)*100:.0f}%)")
    
    # Calculate class weights from training set
    class_weights = None
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(
            train_dataset.dataset, 
            config['dataset']['num_classes']
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    # Test dataset loading
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader, class_weights = get_dataloaders(config)
    
    print(f"\nDataloader Test:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
