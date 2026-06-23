import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

from src.data_processing import AudioProcessor

# Mapping of strict ISO codes to Whisper generation target languages
ISO_TO_WHISPER_LANG = {
    "swa": "swahili",
    "som": "somali",
    "kik": "swahili",  # Proxy to Swahili
    "luo": "swahili",  # Proxy to Swahili
    "mas": "swahili",  # Proxy to Swahili
    "kln": "swahili",  # Proxy to Swahili
}

def parse_filename(file_path):
    """
    Parses the audio filename to extract language and ID.
    Supports formats:
      - audio_{lang}_{id}.wav
      - {lang}_{id}.wav
    """
    stem = file_path.stem
    parts = stem.split("_")
    
    if len(parts) >= 3 and parts[0] == "audio":
        lang = parts[1]
        audio_id = "_".join(parts[2:])
    elif len(parts) >= 2:
        lang = parts[0]
        audio_id = "_".join(parts[1:])
    else:
        lang = "swa"  # Default fallback
        audio_id = stem
        
    return audio_id, lang

def main():
    parser = argparse.ArgumentParser(description="SautiEdge Offline Inference Script")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to directory containing test audio files")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Path to save submission CSV")
    args = parser.parse_args()

    model_path = "models/quantized"
    test_path = Path(args.test_dir)
    
    # 1. Load the quantized ONNX model and processor
    print(f"Loading quantized ONNX model from {model_path}...")
    if not os.path.exists(model_path):
        raise ValueError(f"Quantized model path '{model_path}' does not exist. Please run export_onnx.py first.")
        
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path, provider="CPUExecutionProvider")
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # Initialize our robust audio loader (target rate 16kHz)
    audio_proc = AudioProcessor(target_sr=16000)

    # 2. Find test audio files
    audio_extensions = ["wav", "mp3", "flac"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(test_path.glob(f"*.{ext}"))
        
    if not audio_files:
        print(f"No audio files found in directory: {test_path}")
        # Create an empty template submission if directory is empty or missing
        df = pd.DataFrame(columns=["id", "language", "transcription"])
        df.to_csv(args.output_csv, index=False)
        print(f"Created empty submission template at {args.output_csv}")
        return

    print(f"Found {len(audio_files)} test audio files. Starting transcription loop...")
    results = []

    # 3. Transcribe test audio files
    for file_path in tqdm(audio_files, desc="Transcribing"):
        try:
            # Parse ID and language from filename
            audio_id, lang_code = parse_filename(file_path)
            
            # Map language code to Whisper's vocabulary
            whisper_lang = ISO_TO_WHISPER_LANG.get(lang_code.lower(), "swahili")
            
            # Load and resample audio
            y = audio_proc.load_audio(str(file_path))
            
            # Extract Log-Mel feature representation
            inputs = processor(y, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features
            
            # Get forced decoder IDs for target language prefix
            forced_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
            
            # Run inference on the quantized ONNX runtime CPU provider
            predicted_ids = model.generate(
                input_features, 
                forced_decoder_ids=forced_ids,
                max_length=225
            )
            
            # Decode predictions to text transcription
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            results.append({
                "id": audio_id,
                "language": lang_code,
                "transcription": transcription.strip()
            })
            
        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")
            # Append empty transcription on failure to preserve submission structure
            audio_id, lang_code = parse_filename(file_path)
            results.append({
                "id": audio_id,
                "language": lang_code,
                "transcription": ""
            })

    # 4. Save results to submission.csv
    df = pd.DataFrame(results)
    df = df[["id", "language", "transcription"]]  # Ensure strict column ordering
    df.to_csv(args.output_csv, index=False)
    print(f"\nInference complete. Submission file saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
