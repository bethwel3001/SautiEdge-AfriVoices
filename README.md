# SautiEdge: Unified Offline Edge ASR for East African Languages

SautiEdge is a unified, fully offline speech recognition (ASR) system designed to transcribe six East African languages—Swahili, Kikuyu, Luo, Somali, Maasai, and Kalenjin—locally on low-resource edge devices (like a Raspberry Pi 4 or entry-level Android devices).

It serves as our official submission for the **Insurance Solutions Hackathon (Insurtech Nairobi)**, specifically targeting the **Insurance Education & Customer Awareness** and **Microinsurance Coverage** tracks by enabling natural voice-based interfaces in native local languages without any internet requirements.

---

## 🔗 Key Documentation
- **Technical Research & Methodology:** Read our full write-up in [technical_blog_post.md](file:///home/bethwel/Desktop/SautiEdge-AfriVoices/reports/technical_blog_post.md).
- **Project Proposal & Constraints:** Review the core objectives in [PROPOSAL.md](file:///home/bethwel/Desktop/SautiEdge-AfriVoices/PROPOSAL.md).
- **Target Hardware Validation Report:** View the judge-ready benchmark report in [hardware_validation.md](file:///home/bethwel/Desktop/SautiEdge-AfriVoices/reports/hardware_validation.md).
- **Latest Benchmark Validation Run:** See the hardware profiling results in [hardware_validation_results.txt](file:///home/bethwel/Desktop/SautiEdge-AfriVoices/reports/hardware_validation_results.txt) (generated during pipeline runs).

---

## ⚙️ Project Architecture
SautiEdge uses a decoupled approach to achieve high accuracy within edge resource limitations:
1. **Acoustic Adapter (Hearing):** A base `openai/whisper-small` (244M parameters) fine-tuned using Low-Rank Adaptation (LoRA) on projection layers (`q_proj`, `v_proj`, `k_proj`, `out_proj`).
2. **Dynamic Language Routing (Understanding):** A custom speech sequence data collator and custom ASR training loop that dynamically guides target output tokens based on the source language, bypassing heavy external compile dependencies.
3. **Dynamic INT8 Quantization:** The model is merged and exported to multi-part ONNX runtime graphs and dynamically quantized to INT8, shrinking the disk footprint from ~960MB to ~240MB and accelerating inference.

---

## 🚀 Reproduction and Execution Guide

### Prerequisites
Ensure your system has the following installed:
- Python 3.9 or higher (Tested and fully compatible with Python 3.11 and 3.13)
- Bash shell environment
- FFmpeg (for audio loading fallbacks)

### 1. Setup & Installation
Clone this repository to your local system and navigate to the project root:
```bash
git clone https://github.com/bethwel3001/SautiEdge-AfriVoices.git
cd SautiEdge-AfriVoices
```

Initialize your virtual environment and install the required dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Your Datasets
Before running the pipeline, place your raw language datasets under `data/raw/` in their respective folders (e.g. `data/raw/kikuyu/`, `data/raw/maasai/`, `data/raw/swahili/`).
*Note:* The scripts support loading datasets in Hugging Face format (via `save_to_disk`) as well as folders containing raw audio files (`.wav`, `.mp3`, `.flac`) paired with metadata transcriptions (CSV, TSV, or parallel `.txt` files).

### 3. Run the Master Orchestration Pipeline
Execute the master orchestration shell script. This script executes all steps sequentially with immediate stop-on-error (`set -e`) enabled:
```bash
bash run_pipeline.sh
```

This single command automates the following actions:
1. **`prepare_data.py`**: Scans the raw folders, maps them to ISO codes, applies clipping and silence filters, partitions into 90/5/5 splits, and saves to `data/processed/`.
2. **`train.py`**: Loads the data, applies LoRA, runs training, evaluates validation WER, and saves the adapter weights to `checkpoints/lora_unified/`.
3. **`export_onnx.py`**: Merges LoRA adapters into the base Whisper weights, exports to ONNX, and applies dynamic INT8 quantization, saving output models to `models/quantized/`.
4. **`inference.py`**: transcribes the test audio files in `data/test/` using the quantized model on CPU, producing a submission-ready `submission.csv`.
5. **`hardware_validator.py`**: Profiles Peak RAM usage and Real-Time Factor (RTF) during CPU inference, verifying strict constraint bounds (RAM $\le$ 8GB, RTF $\le$ 2.0) and outputting [hardware_validation_results.txt](file:///home/bethwel/Desktop/SautiEdge-AfriVoices/reports/hardware_validation_results.txt).
6. **`submission_check.py`**: Performs Kaggle submission formatting checks (ID uniqueness, permitted ISO languages, missing value checks) to ensure your submission is valid.

---

## ⚖️ License and Data Compliance
- **Software License:** SautiEdge is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- **Data Compliance:** All datasets used are compliant with the CC BY 4.0 license, maintaining strict user data privacy, regulatory compliance, and responsible AI practices under the Insurance Regulatory Authority (IRA) guidelines.
