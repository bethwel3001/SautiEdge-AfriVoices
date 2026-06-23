import os
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

def generate_noise_audio(duration, sample_rate=16000):
    """Generates a random float32 noise signal within acceptable amplitude bounds."""
    # Scale to prevent silence detection (<0.001 RMS) and clipping detection (>=0.99)
    num_samples = int(duration * sample_rate)
    # Standard deviation of 0.05 gives RMS around 0.05, avoiding silence/clipping flags
    audio = np.random.normal(0, 0.05, num_samples).astype(np.float32)
    # Clip to be safe
    audio = np.clip(audio, -0.5, 0.5)
    return audio

def main():
    print("==================================================")
    print("        SautiEdge Dummy Data Generator            ")
    print("==================================================")
    
    # 1. Define target folders and transcripts
    languages = {
        "swahili": {
            "iso": "swa",
            "phrases": ["habari ya asubuhi", "jambo sana rafiki", "ninataka bima ya afya", "asante kwa msaada"]
        },
        "kikuyu": {
            "iso": "kik",
            "phrases": ["wimwega mwega", "nĩndenda ũteithio waku", "ũhoro mwega mũno", "ũhoro wĩna wĩra"]
        },
        "luo": {
            "iso": "luo",
            "phrases": ["amadhano osiepe", "aneno maber ahinya", "achamo chiemo", "ang\'eyo ni ibiro konyo"]
        },
        "somali": {
            "iso": "som",
            "phrases": ["subax wanaagsan", "maalin wanaagsan", "aad baad u mahadsantahay", "waxaan u baahanahay caymis"]
        },
        "maasai": {
            "iso": "mas",
            "phrases": ["supat olchere", "aji apa ninye", "kaaji ng\'ole sidai", "loipoki lenye sidai"]
        },
        "kalenjin": {
            "iso": "kln",
            "phrases": ["chamgei mising", "abere iwendi ano", "o karibu kabisa", "kagechamgei amu wendi"]
        }
    }
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    total_raw_files = 0
    
    # 2. Generate raw training audio and metadata for each language
    for folder_name, lang_info in languages.items():
        lang_dir = raw_dir / folder_name
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_records = []
        
        print(f"Generating dummy files for: {folder_name} ({lang_info['iso']})...")
        
        for idx, phrase in enumerate(lang_info["phrases"]):
            filename = f"audio_{lang_info['iso']}_{idx:03d}.wav"
            filepath = lang_dir / filename
            
            # Generate dummy audio with a random duration between 2 and 5 seconds
            duration = np.random.uniform(2.0, 5.0)
            audio = generate_noise_audio(duration, sample_rate=16000)
            
            sf.write(str(filepath), audio, 16000)
            
            metadata_records.append({
                "file_name": filename,
                "transcription": phrase
            })
            total_raw_files += 1
            
        # Write metadata.csv for this language
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(lang_dir / "metadata.csv", index=False)
        print(f" - Saved metadata.csv and {len(metadata_records)} WAV files to {lang_dir}")
        
    # 3. Generate dummy test audio files
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_languages = ["swa", "kik", "luo", "som", "mas", "kln"]
    total_test_files = 0
    print("\nGenerating dummy files for test set (data/test/)...")
    
    for idx, lang in enumerate(test_languages):
        # Generate 2 test files per language
        for sub_idx in range(2):
            filename = f"audio_{lang}_{idx * 2 + sub_idx:03d}.wav"
            filepath = test_dir / filename
            
            duration = np.random.uniform(1.5, 4.0)
            audio = generate_noise_audio(duration, sample_rate=16000)
            
            sf.write(str(filepath), audio, 16000)
            total_test_files += 1
            
    print(f" - Saved {total_test_files} test WAV files to {test_dir}")
    
    print("\n==================================================")
    print(f" SUCCESS: Dummy datasets created successfully!    ")
    print(f" Total training samples: {total_raw_files}        ")
    print(f" Total test samples:     {total_test_files}       ")
    print("==================================================")

if __name__ == "__main__":
    main()
