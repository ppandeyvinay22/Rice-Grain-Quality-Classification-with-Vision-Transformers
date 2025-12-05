"""
Evaluate Saved Model
Load checkpoint and generate confusion matrix + classification report
"""

import os
import yaml
import torch
import argparse
from dataset import get_dataloaders
from model import get_model
from utils import evaluate_model, plot_confusion_matrix, save_classification_report


def evaluate_checkpoint(checkpoint_path, config):
    """
    Evaluate a saved checkpoint
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data (validation set only)
    print("Loading validation dataset...")
    _, val_loader, _ = get_dataloaders(config)
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(config, device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    saved_val_acc = checkpoint['val_acc']
    
    print(f"  Epoch: {epoch + 1}")
    print(f"  Saved Val Accuracy: {saved_val_acc:.2f}%")
    
    # Evaluate
    print("\n" + "="*80)
    print("RUNNING EVALUATION...")
    print("="*80 + "\n")
    
    eval_results = evaluate_model(
        model, val_loader, device, config['dataset']['classes']
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Validation Accuracy: {eval_results['accuracy']:.2f}%")
    print("="*80 + "\n")
    
    # Extract checkpoint name for output files
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    # Save confusion matrix
    cm_path = os.path.join(
        config['paths']['results_dir'], 
        f'confusion_matrix_{checkpoint_name}.png'
    )
    plot_confusion_matrix(
        eval_results['confusion_matrix'],
        config['dataset']['classes'],
        cm_path
    )
    
    # Save classification report
    report_path = os.path.join(
        config['paths']['results_dir'],
        f'classification_report_{checkpoint_name}.txt'
    )
    save_classification_report(
        eval_results['classification_report'],
        config['dataset']['classes'],
        report_path
    )
    
    # Print per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print("-"*60)
    report = eval_results['classification_report']
    for class_name in config['dataset']['classes']:
        if class_name in report:
            acc = report[class_name]['precision'] * 100
            support = report[class_name]['support']
            print(f"{class_name:10s}: {acc:6.2f}% (support: {int(support):5d})")
    print("-"*60)
    
    # Find most confused classes
    print("\nMOST CONFUSED CLASSES (Top 5):")
    print("-"*60)
    cm = eval_results['confusion_matrix']
    
    confusions = []
    for i in range(len(config['dataset']['classes'])):
        for j in range(len(config['dataset']['classes'])):
            if i != j and cm[i][j] > 0:
                confusions.append({
                    'true': config['dataset']['classes'][i],
                    'pred': config['dataset']['classes'][j],
                    'count': cm[i][j]
                })
    
    # Sort by count
    confusions.sort(key=lambda x: x['count'], reverse=True)
    
    for conf in confusions[:5]:
        print(f"{conf['true']:10s} → {conf['pred']:10s}: {int(conf['count']):4d} misclassifications")
    print("-"*60)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print(f"  Confusion Matrix: {cm_path}")
    print(f"  Classification Report: {report_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a saved model checkpoint')
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='checkpoints/best_model.pth',
        help='Path to checkpoint file (default: checkpoints/best_model.pth)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"\n❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = config['paths']['checkpoint_dir']
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            for ckpt in sorted(checkpoints):
                print(f"  - {os.path.join(checkpoint_dir, ckpt)}")
        exit(1)
    
    # Run evaluation
    evaluate_checkpoint(args.checkpoint, config)