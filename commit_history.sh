#!/bin/bash
set -e

echo "Starting granular commits..."

# Commit 1: Documentation and License
git add LICENSE* README.md PROPOSAL.md reports/technical_blog_post.md
git commit -m "docs: add project documentation, technical blog, proposal, and Apache 2.0 license" || true

# Commit 2: Configurations
git add configs/
git commit -m "chore: add model configuration, hyperparameters, and edge optimization settings" || true

# Commit 3: Core Data Pipeline
git add src/data_processing.py src/dataset.py src/prepare_data.py
git commit -m "feat: implement data ingestion, audio cleaning, and unified PyTorch dataset pipeline" || true

# Commit 4: Model Training
git add src/train.py
git commit -m "feat: add unified LoRA fine-tuning script for 6 East African languages" || true

# Commit 5: Edge Optimization and Inference
git add src/export_onnx.py src/inference.py
git commit -m "feat: implement ONNX export, INT8 quantization, and edge inference engine" || true

# Commit 6: Validation and Orchestration
git add src/hardware_validator.py src/submission_check.py run_pipeline.sh
git commit -m "feat: add hardware validation, Kaggle submission checker, and master orchestration script" || true

# Commit 7: Gitignore
git add .gitignore
git commit -m "chore: add gitignore to exclude raw data, model weights, and python artifacts" || true

echo "All granular commits completed successfully!"
