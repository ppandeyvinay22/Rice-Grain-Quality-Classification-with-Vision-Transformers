#!/bin/bash

# Convert ONNX to OpenVINO IR (.xml + .bin)
# Requires: openvino-dev toolkit

ONNX_MODEL=$1
OUTPUT_DIR=${2:-"openvino_model"}

if [ -z "$ONNX_MODEL" ]; then
    echo "Usage: ./convert_to_openvino.sh <onnx_model_path> [output_dir]"
    exit 1
fi

echo "=================================="
echo "ONNX → OpenVINO IR Conversion"
echo "=================================="
echo "Input:  $ONNX_MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Convert using Model Optimizer
mo --input_model $ONNX_MODEL \
   --output_dir $OUTPUT_DIR \
   --input_shape [1,3,224,224] \
   --model_name rice_grain_model

echo ""
echo "✅ Conversion complete!"
echo "Output files:"
echo "  - $OUTPUT_DIR/rice_grain_model.xml"
echo "  - $OUTPUT_DIR/rice_grain_model.bin"
echo ""
echo "Test inference:"
echo "  python test_openvino.py --model $OUTPUT_DIR/rice_grain_model.xml"