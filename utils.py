"""
Utility Functions
Training helpers, metrics, visualization, and checkpointing
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Dict, List
import json
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard samples
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        weight: Class weights (optional)
    """
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: True labels (batch_size)
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """Early stopping to stop training when validation accuracy stops improving"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_acc: float, epoch: int) -> bool:
        """
        Check if should stop training
        
        Args:
            val_acc: Current validation accuracy
            epoch: Current epoch number
            
        Returns:
            True if should stop, False otherwise
        """
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered! No improvement for {self.patience} epochs.")
                print(f"Best validation accuracy: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return False


class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
    def update(self, metrics: Dict):
        """Update metrics history"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self, filename: str = 'training_history.json'):
        """Save metrics to JSON"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {filepath}")
    
    def plot_metrics(self, save_path: str):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, config, filename):
    """
    Save model checkpoint
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        epoch: Current epoch
        val_acc: Validation accuracy
        config: Configuration dictionary
        filename: Checkpoint filename
    """
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance (optional)
        scheduler: Scheduler instance (optional)
        
    Returns:
        epoch, val_acc
    """
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    print(f"Checkpoint loaded from epoch {epoch} with val_acc: {val_acc:.4f}")
    
    return epoch, val_acc


def evaluate_model(model, dataloader, device, class_names):
    """
    Comprehensive model evaluation
    
    Args:
        model: Model instance
        dataloader: Validation dataloader
        device: torch.device
        class_names: List of class names
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels) * 100
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'},
        square=True
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_classification_report(report, class_names, save_path):
    """
    Save classification report to text file
    
    Args:
        report: Classification report dictionary
        class_names: List of class names
        save_path: Path to save report
    """
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RICE GRAIN CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics
        f.write(f"Overall Accuracy: {report['accuracy']:.4f}\n\n")
        
        # Per-class metrics
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for class_name in class_names:
            metrics = report[class_name]
            f.write(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1-score']:<12.4f} {metrics['support']:<10.0f}\n")
        
        f.write("-"*80 + "\n")
        
        # Macro average
        macro = report['macro avg']
        f.write(f"{'Macro Avg':<15} {macro['precision']:<12.4f} {macro['recall']:<12.4f} "
               f"{macro['f1-score']:<12.4f} {macro['support']:<10.0f}\n")
        
        # Weighted average
        weighted = report['weighted avg']
        f.write(f"{'Weighted Avg':<15} {weighted['precision']:<12.4f} {weighted['recall']:<12.4f} "
               f"{weighted['f1-score']:<12.4f} {weighted['support']:<10.0f}\n")
        
        f.write("="*80 + "\n")
    
    print(f"Classification report saved to {save_path}")


def print_training_summary(config, train_loader, val_loader):
    """Print training configuration summary"""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Model:           {config['model']['name']}")
    print(f"Input Size:      {config['dataset']['image_size']}x{config['dataset']['image_size']}")
    print(f"Num Classes:     {config['dataset']['num_classes']}")
    print(f"Batch Size:      {config['training']['batch_size']}")
    print(f"Epochs:          {config['training']['num_epochs']}")
    print(f"Learning Rate:   {config['training']['learning_rate']}")
    print(f"Weight Decay:    {config['training']['weight_decay']}")
    print(f"Gradient Clip:   {config['training'].get('gradient_clip_norm', 'None')}")
    print(f"Optimizer:       {config['training']['optimizer'].upper()}")
    print(f"Scheduler:       {config['training']['scheduler']['type'].upper()}")
    print(f"Class Weights:   {'Enabled' if config['training']['use_class_weights'] else 'Disabled'}")
    print(f"Early Stopping:  {'Enabled' if config['training']['early_stopping']['enabled'] else 'Disabled'}")
    print(f"Device:          {config['gpu']['device'].upper()}")
    print(f"Train Batches:   {len(train_loader)}")
    print(f"Val Batches:     {len(val_loader)}")
    print("="*80 + "\n")
