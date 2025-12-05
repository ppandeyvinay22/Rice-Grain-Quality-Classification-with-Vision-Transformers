"""
MobileViT Model Setup
Loads pretrained MobileViT and adapts for rice grain classification
"""

import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification, MobileViTConfig
from transformers import AutoModelForImageClassification  # ADD THIS
from typing import Dict


class RiceGrainMobileViT(nn.Module):
    """
    MobileViT wrapper for rice grain classification
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super(RiceGrainMobileViT, self).__init__()
        
        model_name = config['model']['name']
        num_classes = config['dataset']['num_classes']
        pretrained = config['model']['pretrained']
        
        print(f"\nInitializing MobileViT Model:")
        print(f"  Model: {model_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Num Classes: {num_classes}")
        
        if pretrained:
            # Load pretrained model (works with ANY vision model!)
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # Allow classifier head replacement
            )
        else:
            # Load from scratch
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(model_name)
            model_config.num_labels = num_classes
            self.model = AutoModelForImageClassification.from_config(model_config)
        # Get model info
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Total params: {self.num_params:,}")
        print(f"  Trainable params: {self.num_trainable:,}")
    
    def forward(self, pixel_values):
        """
        Forward pass
        
        Args:
            pixel_values: Tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
    
    def freeze_backbone(self):
        """Freeze all layers except classifier head"""
        print("\nFreezing backbone layers...")
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        self.num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable params after freezing: {self.num_trainable:,}")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        print("\nUnfreezing all layers...")
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable params after unfreezing: {self.num_trainable:,}")


def get_model(config: Dict, device: torch.device) -> RiceGrainMobileViT:
    """
    Initialize model and move to device
    
    Args:
        config: Configuration dictionary
        device: torch.device
        
    Returns:
        Model instance
    """
    model = RiceGrainMobileViT(config)
    model = model.to(device)
    
    # Multi-GPU support
    if config['gpu']['multi_gpu'] and torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    return model


def get_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Get optimizer based on config
    
    Args:
        model: Model instance
        config: Configuration dictionary
        
    Returns:
        Optimizer
    """
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer_name = config['training']['optimizer'].lower()
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"\nOptimizer: {optimizer_name.upper()}")
    print(f"  Learning Rate: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict, steps_per_epoch: int):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        steps_per_epoch: Number of training steps per epoch
        
    Returns:
        Scheduler
    """
    scheduler_type = config['training']['scheduler']['type'].lower()
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training']['warmup_epochs']
    
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    if scheduler_type == 'cosine':
        eta_min = config['training']['scheduler']['cosine_eta_min']
        
        # Warmup + Cosine Annealing
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(eta_min / config['training']['learning_rate'], 
                          0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == 'step':
        step_size = config['training']['scheduler'].get('step_size', 30)
        gamma = config['training']['scheduler'].get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    print(f"\nScheduler: {scheduler_type.upper()}")
    print(f"  Warmup Epochs: {warmup_epochs}")
    print(f"  Total Steps: {total_steps:,}")
    
    return scheduler


if __name__ == "__main__":
    # Test model loading
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize model
    model = get_model(config, device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
