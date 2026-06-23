#!/bin/bash

# ==============================================================================
# SautiEdge: Master Orchestration Pipeline Script
# Sequentially executes data preparation, training, edge export, validation, and submission formatting checks.
# ==============================================================================

# Stop immediately if any command exits with a non-zero status
set -e

echo "======================================================================"
echo "          Starting SautiEdge Unified Edge ASR Pipeline                "
echo "======================================================================"

# 1. Environment Activation
echo "--> Step 1/6: Setting up the python environment..."
if [ -d ".venv" ]; then
    echo "Activating virtual environment found in .venv/..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment found in venv/..."
    source venv/bin/activate
else
    echo "WARNING: No virtual environment found. Running with global python."
fi

# Print Python version for confirmation
python_ver=$(python --version)
echo "Using environment: $python_ver"

# Ensure HF_TOKEN is noted if present
if [ -z "$HF_TOKEN" ]; then
    echo "NOTE: HF_TOKEN environment variable is not set. Hugging Face downloads will run unauthenticated (subject to rate limits)."
else
    echo "HF_TOKEN detected. Token authentication will be applied to Hugging Face Hub operations."
fi

# 2. Data Preparation and Cleaning
echo ""
echo "--> Step 2/6: Executing Data Preparation..."
echo "Scanning raw data, mapping ISO codes, filtering garbage, and split partitioning..."
python src/prepare_data.py

# 3. Model Fine-Tuning (LoRA)
echo ""
echo "--> Step 3/6: Executing LoRA Fine-Tuning..."
echo "Starting unified multilingual Whisper-Small PEFT training..."
python src/train.py

# 4. ONNX Model Export & Quantization
echo ""
echo "--> Step 4/6: Executing ONNX Edge Export and Quantization..."
echo "Merging adapters, exporting to ONNX, and applying dynamic INT8 quantization..."
python src/export_onnx.py

# 5. Offline Inference and Submission Generation
echo ""
echo "--> Step 5/6: Executing Edge Inference..."
echo "Running offline model generation on the test set and generating submission.csv..."
python src/inference.py

# 6. Hardware Validation Benchmarks
echo ""
echo "--> Step 6/6: Running Hardware Validation Benchmarks..."
echo "Measuring Peak RAM usage and Real-Time Factor (RTF) constraints..."
python src/hardware_validator.py

# 7. Kaggle Submission Verification
echo ""
echo "--> Verification: Checking submission.csv format compliance..."
python src/submission_check.py

echo ""
echo "======================================================================"
echo "          SautiEdge Pipeline Completed Successfully!                  "
echo "======================================================================"
