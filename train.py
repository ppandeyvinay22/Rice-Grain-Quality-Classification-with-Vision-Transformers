"""
Main Training Script for Rice Grain Classification
MobileViT with Class Weights, Label Smoothing, and Comprehensive Evaluation
OPTIMIZED VERSION
"""

import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from datetime import datetime

from dataset import get_dataloaders
from model import get_model, get_optimizer, get_scheduler
from utils import (
    EarlyStopping, MetricsTracker, save_checkpoint, 
    evaluate_model, plot_confusion_matrix, save_classification_report,
    print_training_summary, FocalLoss
)


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, config):
    """
    Train for one epoch
    
    Args:
        model: Model instance
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: torch.device
        epoch: Current epoch number
        config: Configuration dictionary
        
    Returns:
        Average loss and accuracy
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        if 'gradient_clip_norm' in config['training'] and config['training']['gradient_clip_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip_norm']
            )
        
        optimizer.step()
        
        # Update scheduler (step-wise for cosine)
        if config['training']['scheduler']['type'] == 'cosine':
            scheduler.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update running loss
        running_loss += loss.item()
        
        # Update progress bar
        if batch_idx % config['logging']['log_interval'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%",
                'lr': f"{current_lr:.6f}"
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch, config):
    """
    Validate model
    
    Args:
        model: Model instance
        dataloader: Validation dataloader
        criterion: Loss function
        device: torch.device
        epoch: Current epoch number
        config: Configuration dictionary
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Val]  ")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train(config):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Setup device
    device = torch.device(config['gpu']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*80}\n")
    
    # Create output directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, class_weights = get_dataloaders(config)
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(config, device)
    
    # Setup loss function WITH LABEL SMOOTHING
    if config['training'].get('use_focal_loss', False):
        focal_gamma = config['training'].get('focal_gamma', 2.0)
        
        if config['training']['use_class_weights'] and class_weights is not None:
            class_weights = class_weights.to(device)
            criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
            print(f"\nUsing Focal Loss (gamma={focal_gamma}) with Class Weights")
        else:
            criterion = FocalLoss(gamma=focal_gamma)
            print(f"\nUsing Focal Loss (gamma={focal_gamma})")
    else:
        # UPDATED: Added label_smoothing=0.1
        if config['training']['use_class_weights'] and class_weights is not None:
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=0.1  # ADDED!
            )
            print("\nUsing weighted CrossEntropyLoss with Label Smoothing (0.1)")
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=0.1  # ADDED!
            )
            print("\nUsing standard CrossEntropyLoss with Label Smoothing (0.1)")
    
    # Setup optimizer
    optimizer = get_optimizer(model, config)
    
    # Setup scheduler
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
        print(f"\nEarly Stopping enabled (patience: {config['training']['early_stopping']['patience']})")
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(config['paths']['log_dir'])
    
    # Print training summary
    print_training_summary(config, train_loader, val_loader)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    print("Starting training...\n")
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, config
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, config
        )
        
        # Update scheduler (epoch-wise for step/plateau)
        # UPDATED: Plateau now monitors val_acc instead of val_loss
        if config['training']['scheduler']['type'] == 'step':
            scheduler.step()
        elif config['training']['scheduler']['type'] == 'plateau':
            scheduler.step(val_loss)
        
        # Get current learning rate (convert to float for JSON serialization)
        current_lr = float(optimizer.param_groups[0]['lr'])
        
        # Update metrics
        metrics_tracker.update({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rates': current_lr
        })
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"{'='*80}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            # Always save best with accuracy in filename
            best_filename = f'best_model_{val_acc:.2f}_epoch_{epoch+1}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, config,
                best_filename
            )
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%\n")
        
        # Save all models above threshold with accuracy in filename
        save_threshold = config['logging'].get('save_threshold', 95.0)
        if val_acc >= save_threshold:
            checkpoint_filename = f'model_{val_acc:.2f}_epoch_{epoch+1}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc, config,
                checkpoint_filename
            )
            print(f"✓ Saved model (Val Acc: {val_acc:.2f}% >= {save_threshold}%)\n")
        
        # Early stopping check
        if early_stopping:
            if early_stopping(val_acc, epoch + 1):
                print(f"\nTraining stopped early at epoch {epoch + 1}")
                break
    
    # Training complete
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch}")
    print("="*80 + "\n")
    
    # Save final metrics
    metrics_tracker.save()
    
    # Plot training curves
    print("\nGenerating training curves...")
    metrics_tracker.plot_metrics(
        os.path.join(config['paths']['results_dir'], 'training_curves.png')
    )
    
    # Final evaluation on validation set
    print("\nPerforming final evaluation...")
    
    # Load best model
    best_checkpoint = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Evaluate
    eval_results = evaluate_model(
        model, val_loader, device, config['dataset']['classes']
    )
    
    print(f"\nFinal Validation Accuracy: {eval_results['accuracy']:.2f}%")
    
    # Save confusion matrix
    plot_confusion_matrix(
        eval_results['confusion_matrix'],
        config['dataset']['classes'],
        os.path.join(config['paths']['results_dir'], 'confusion_matrix.png')
    )
    
    # Save classification report
    save_classification_report(
        eval_results['classification_report'],
        config['dataset']['classes'],
        os.path.join(config['paths']['results_dir'], 'classification_report.txt')
    )
    
    print("\n" + "="*80)
    print("All results saved!")
    print(f"  Checkpoints: {config['paths']['checkpoint_dir']}")
    print(f"  Logs: {config['paths']['log_dir']}")
    print(f"  Results: {config['paths']['results_dir']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Load configuration
    config_path = 'config.yaml'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Start training
    train(config)
