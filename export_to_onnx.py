"""
Export PyTorch Model to ONNX
Step 1 of PyTorch → OpenVINO conversion
"""

import torch
import yaml
import argparse
from model import RiceGrainMobileViT


def export_to_onnx(checkpoint_path, output_path, config):
    """
    Export PyTorch model to ONNX format
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        output_path: Output ONNX file path
        config: Configuration dict
    """
    print("\n" + "="*80)
    print("PYTORCH → ONNX EXPORT")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_path}")
    
    # Setup device
    device = torch.device('cpu')  # Export on CPU
    
    # Load model
    print("\nLoading model...")
    model = RiceGrainMobileViT(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model.model,  # The actual HuggingFace model
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model saved: {output_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to .pth checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model.onnx',
        help='Output ONNX file path (default: model.onnx)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Export
    export_to_onnx(args.checkpoint, args.output, config)
    
    print("Next step: Convert ONNX → OpenVINO")
    print(f"Run: mo --input_model {args.output} --output_dir openvino_model/")