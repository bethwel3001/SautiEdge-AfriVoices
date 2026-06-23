# SautiEdge: Unified Offline Edge ASR for East African Languages

## 1. Project Context
SautiEdge is a unified, fully offline speech recognition (ASR) system designed to bridge the digital divide in East Africa. By running on local edge hardware without cloud dependency, it enables digital insurance education, customer awareness, and microinsurance access for native speakers in Swahili, Kikuyu, Luo, Somali, Maasai, and Kalenjin.

## 2. The Problem
Modern speech-to-text (ASR) systems are heavily biased toward global, high-resource languages and rely on continuous cloud connectivity. In East Africa, millions of potential insurance users (such as informal traders, smallholder farmers, and boda-boda riders) are excluded because:
- High-quality, fast internet is expensive or unavailable in rural areas.
- They speak local languages or dialects (such as Maasai, Kalenjin, Luo, Kikuyu, Somali) which are unsupported by mainstream cloud providers.
- Traditional app-based insurance enrollment has high literacy and technical entry barriers compared to natural voice interfaces.

## 3. The Innovation & Function
SautiEdge introduces a decoupled architecture:
- **Hearing (Lightweight Acoustic Model):** A fine-tuned, quantized Whisper-Small (244M parameters) model processes raw audio on the fly.
- **Understanding (N-gram Language Model Fusion):** During decoding, shallow fusion with lightweight KenLM / pyctcdecode models guides the output. This provides language-specific grammatical structure and vocabulary constraints, significantly reducing Word Error Rate (WER) for low-resource languages.
- **Edge Deployment:** The pipeline uses ONNX runtime with INT8 dynamic quantization to deliver fast CPU inference on entry-level edge hardware. SautiEdge requires no GPU and no cloud backend, running locally on a $50 Raspberry Pi 4.

## 4. Strict Hackathon Constraints
SautiEdge strictly adheres to the following guidelines:
1. **Model Parameter Count:** Strictly under 1 Billion parameters (Whisper-Small utilizes ~244M parameters).
2. **Execution Environment:** CPU-only inference (No GPU required).
3. **Target Edge Hardware:** Raspberry Pi 4 (8GB RAM) or entry-level Android devices.
4. **RAM Utilization:** Peak memory during inference is strictly <= 8GB.
5. **Latency (Real-Time Factor):** Latency is <= 2x real-time audio duration (RTF <= 2.0).
6. **Output Format:** Clean Kaggle submission CSV with exactly three columns: `id`, `language`, `transcription`.
7. **ISO Language Codes:** Explicitly 3-letter ISO 639-3 codes:
   - `swa` - Swahili
   - `kik` - Kikuyu
   - `luo` - Luo
   - `som` - Somali
   - `mas` - Maasai
   - `kln` - Kalenjin
