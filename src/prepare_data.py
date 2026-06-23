import os
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import datasets
from datasets import Dataset, DatasetDict
import soundfile as sf
import numpy as np
import hashlib

from src.data_processing import AudioProcessor, GarbageAudioError

# Mapping of folder names/variations to strict 3-letter ISO codes
FOLDER_TO_ISO = {
    "swahili": "swa", "swa": "swa", "sw": "swa",
    "kikuyu": "kik", "kik": "kik",
    "luo": "luo",
    "somali": "som", "som": "som",
    "maasai": "mas", "mas": "mas",
    "kalenjin": "kln", "kln": "kln"
}

def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the AudioProcessor with target rate and duration constraints
    audio_proc = AudioProcessor(target_sr=16000, min_duration=0.5, max_duration=20.0)
    
    combined_data = []
    
    # Scan raw_dir for folders
    lang_folders = [d for d in raw_dir.iterdir() if d.is_dir()]
    
    if not lang_folders:
        print("No language folders found in data/raw/!")
        return

    for folder in lang_folders:
        folder_name = folder.name.lower()
        if folder_name not in FOLDER_TO_ISO:
            print(f"Skipping unknown folder: {folder.name}")
            continue
            
        lang_code = FOLDER_TO_ISO[folder_name]
        print(f"\n--- Processing folder: {folder.name} -> ISO: {lang_code} ---")
        
        lang_samples = []
        
        # Scenario A: Folder contains a saved Hugging Face dataset on disk
        if (folder / "dataset_info.json").exists() or (folder / "state.json").exists():
            print(f"Loading HF dataset from disk for {folder.name}...")
            try:
                hf_data = datasets.load_from_disk(str(folder))
                # Concatenate divisions (train, dev, test) if it is a DatasetDict
                if isinstance(hf_data, datasets.DatasetDict):
                    from datasets import concatenate_datasets
                    hf_dataset = concatenate_datasets(list(hf_data.values()))
                else:
                    hf_dataset = hf_data
                
                cols = hf_dataset.column_names
                possible_text_cols = ["transcription", "text", "sentence", "transcript"]
                text_col = next((c for c in cols if c in possible_text_cols), None)
                
                if "audio" not in cols or not text_col:
                    print(f"Dataset in {folder.name} missing 'audio' or text column. Found columns: {cols}")
                    continue
                
                # Output folder for saving embedded audio arrays to WAV files
                extracted_audio_dir = folder / "extracted_audio"
                extracted_audio_dir.mkdir(exist_ok=True)
                
                for item in tqdm(hf_dataset, desc=f"Extracting {lang_code} HF dataset"):
                    transcript = item[text_col]
                    if not transcript or not isinstance(transcript, str) or len(transcript.strip()) == 0:
                        continue
                        
                    audio_info = item["audio"]
                    audio_path = audio_info.get("path")
                    
                    # If audio is embedded (no path or doesn't exist on disk), write to WAV
                    if not audio_path or not os.path.exists(audio_path):
                        array = audio_info.get("array")
                        sr = audio_info.get("sampling_rate", 16000)
                        if array is not None:
                            h = hashlib.sha256(array.tobytes()).hexdigest()[:16]
                            audio_path = str(extracted_audio_dir / f"{h}.wav")
                            if not os.path.exists(audio_path):
                                sf.write(audio_path, array, sr)
                        else:
                            continue
                            
                    # Validate audio using AudioProcessor
                    try:
                        y = audio_proc.load_audio(audio_path)
                        audio_proc.validate_audio(y)
                        lang_samples.append({
                            "audio_path": audio_path,
                            "transcription": transcript.strip(),
                            "language_code": lang_code
                        })
                    except GarbageAudioError:
                        continue
                    except Exception:
                        continue
            except Exception as e:
                print(f"Failed to load HF dataset for {folder.name}: {e}")
                
        # Scenario B: Folder has raw audio files and metadata CSV/TXT files
        else:
            print(f"Scanning directory {folder.name} for raw audio files...")
            metadata_file = None
            for ext in ["csv", "tsv", "txt"]:
                for f in folder.glob(f"*.{ext}"):
                    if "metadata" in f.name.lower() or "transcript" in f.name.lower():
                        metadata_file = f
                        break
                if metadata_file:
                    break
                    
            paired_files = {}
            if metadata_file:
                print(f"Found metadata file: {metadata_file.name}")
                try:
                    if metadata_file.suffix == ".csv":
                        df = pd.read_csv(metadata_file)
                    elif metadata_file.suffix == ".tsv":
                        df = pd.read_csv(metadata_file, sep="\t")
                    else:
                        df = pd.read_csv(metadata_file, sep="|", header=None, names=["file_name", "transcription"])
                        if df.shape[1] < 2:
                            df = pd.read_csv(metadata_file, sep="\t", header=None, names=["file_name", "transcription"])
                            
                    df.columns = [c.lower() for c in df.columns]
                    file_col = next((c for c in df.columns if "file" in c or "audio" in c or "id" in c), None)
                    text_col = next((c for c in df.columns if "text" in c or "trans" in c or "sent" in c), None)
                    
                    if file_col and text_col:
                        for _, row in df.iterrows():
                            fn = str(row[file_col])
                            txt = str(row[text_col])
                            paired_files[fn] = txt
                except Exception as e:
                    print(f"Error parsing metadata file {metadata_file}: {e}")
            
            # Match files
            audio_extensions = ["wav", "mp3", "flac"]
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(folder.rglob(f"*.{ext}"))
                
            for af in tqdm(audio_files, desc=f"Scanning {lang_code} audio files"):
                af_str = str(af)
                transcript = None
                
                if af.name in paired_files:
                    transcript = paired_files[af.name]
                elif af.stem in paired_files:
                    transcript = paired_files[af.stem]
                else:
                    # Fallback to look for a parallel .txt file
                    txt_path = af.with_suffix(".txt")
                    if txt_path.exists():
                        try:
                            with open(txt_path, "r", encoding="utf-8") as f:
                                transcript = f.read().strip()
                        except Exception:
                            pass
                            
                if not transcript or not isinstance(transcript, str) or len(transcript.strip()) == 0:
                    continue
                    
                # Validate audio using AudioProcessor
                try:
                    y = audio_proc.load_audio(af_str)
                    audio_proc.validate_audio(y)
                    lang_samples.append({
                        "audio_path": af_str,
                        "transcription": transcript.strip(),
                        "language_code": lang_code
                    })
                except GarbageAudioError:
                    continue
                except Exception:
                    continue
                    
        print(f"Found {len(lang_samples)} valid samples for {lang_code}.")
        combined_data.extend(lang_samples)
        
    if not combined_data:
        print("No valid audio samples found across all languages!")
        return
        
    print(f"\nTotal combined valid samples across all languages: {len(combined_data)}")
    
    # Create unified HF Dataset
    dataset = Dataset.from_list(combined_data)
    
    # Splitting: 90% train, 5% val, 5% test
    # First split 10% out as temp
    train_temp = dataset.train_test_split(test_size=0.10, seed=42)
    train_dataset = train_temp["train"]
    
    # Split the 10% temp split 50/50 into validation and test
    val_test = train_temp["test"].train_test_split(test_size=0.50, seed=42)
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]
    
    split_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    output_path = processed_dir / "afrivoices_unified"
    split_dataset.save_to_disk(str(output_path))
    print(f"\nSuccessfully saved split datasets to {output_path}")

if __name__ == "__main__":
    main()
