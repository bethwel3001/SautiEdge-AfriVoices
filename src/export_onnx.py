import os
import shutil
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def main():
    base_model_name = "openai/whisper-small"
    adapter_path = "checkpoints/lora_unified"
    merged_path = "checkpoints/merged_model"
    onnx_raw_path = "models/onnx_raw"
    quantized_path = "models/quantized"
    
    # Create directories if they do not exist
    Path("models").mkdir(exist_ok=True)
    
    # 1. Merge LoRA Adapters into Base Model
    print(f"Loading base Whisper model: {base_model_name}")
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    
    if os.path.exists(adapter_path):
        print(f"LoRA adapters found at {adapter_path}. Merging weights...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()
    else:
        print(f"WARNING: Adapter path {adapter_path} not found. Exporting base Whisper model.")
        merged_model = base_model
        
    print(f"Saving merged model weights to {merged_path}...")
    merged_model.save_pretrained(merged_path)
    
    # Save the processor configuration alongside the model
    processor = WhisperProcessor.from_pretrained(base_model_name)
    processor.save_pretrained(merged_path)
    
    # 2. Export the merged PyTorch model to ONNX format
    print("Exporting model to ONNX using Optimum ORTModelForSpeechSeq2Seq...")
    # This automatically exports both encoder and decoder components
    onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(merged_path, export=True)
    print(f"Saving raw ONNX models to {onnx_raw_path}...")
    onnx_model.save_pretrained(onnx_raw_path)
    
    # 3. Apply dynamic INT8 quantization to all components
    print("Applying dynamic INT8 quantization...")
    
    # The quantizer will load all ONNX models in onnx_raw_path
    quantizer = ORTQuantizer.from_pretrained(onnx_raw_path)
    
    # Dynamic INT8 configuration is optimal for speech translation on edge CPUs
    qconfig = AutoQuantizationConfig.ort_config(
        is_static=False,            # Dynamic quantization
        weights_only=False          # Quantize both weights and activations dynamically
    )
    
    # Run the quantization pipeline for all sub-models (encoder, decoder, decoder_with_past)
    quantizer.quantize(
        save_dir=quantized_path,
        quantization_config=qconfig
    )
    
    # Save the processor to the final quantized directory
    processor.save_pretrained(quantized_path)
    
    # 4. Cleanup temporary folders to conserve edge storage space
    print("Cleaning up temporary directories...")
    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)
    if os.path.exists(onnx_raw_path):
        shutil.rmtree(onnx_raw_path)
        
    print(f"\nModel preparation complete. Quantized ONNX model saved to: {quantized_path}")

if __name__ == "__main__":
    main()
