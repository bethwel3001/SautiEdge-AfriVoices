import torch
from torch.utils.data import Dataset
import numpy as np
from src.data_processing import AudioProcessor, GarbageAudioError

class SautiEdgeDataset(Dataset):
    def __init__(self, audio_paths, transcripts, language_codes, processor, 
                 audio_processor=None, apply_augment=False, pre_filter=False):
        """
        SautiEdge PyTorch Dataset for loading, validating, and tokenizing audio and transcripts on the fly.
        
        Args:
            audio_paths (list of str): Paths to audio files.
            transcripts (list of str): Transcriptions corresponding to the audio.
            language_codes (list of str): 3-letter ISO 639-3 language codes (e.g. 'swa', 'kik').
            processor (WhisperProcessor): Whisper processor for feature extraction and tokenization.
            audio_processor (AudioProcessor, optional): Pre-configured AudioProcessor.
            apply_augment (bool): Whether to apply data augmentation during training.
            pre_filter (bool): If True, filters out invalid audio files at initialization.
        """
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.language_codes = language_codes
        self.processor = processor
        
        if audio_processor is None:
            self.audio_processor = AudioProcessor(apply_augment=apply_augment)
        else:
            self.audio_processor = audio_processor
            
        self.indices = list(range(len(self.audio_paths)))
        
        if pre_filter:
            self.filter_dataset()

    def filter_dataset(self):
        """
        Pre-filters the dataset by checking all files and removing invalid / garbage files.
        This is run at initialization to prevent runtime overhead during DataLoader loops.
        """
        valid_indices = []
        print("Pre-filtering dataset files for garbage audio...")
        for idx in range(len(self.audio_paths)):
            file_path = self.audio_paths[idx]
            try:
                y = self.audio_processor.load_audio(file_path)
                self.audio_processor.validate_audio(y)
                valid_indices.append(idx)
            except Exception:
                # Silently skip files that fail loading or validation
                continue
                
        self.indices = valid_indices
        print(f"Filtering complete: {len(self.indices)} / {len(self.audio_paths)} samples are valid.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Loads and processes audio, tokenizes target text, and returns model-ready inputs.
        Implements a circular fallback mechanism in case of runtime file errors.
        """
        attempts = 0
        max_attempts = len(self.indices)
        
        while attempts < max_attempts:
            current_idx = self.indices[idx]
            file_path = self.audio_paths[current_idx]
            text = self.transcripts[current_idx]
            lang_code = self.language_codes[current_idx]
            
            try:
                # 1. Load, validate, and augment audio
                y = self.audio_processor.process(file_path)
                
                # 2. Extract Log-Mel Spectrogram features
                # WhisperFeatureExtractor outputs a dictionary with a list of numpy arrays
                input_features = self.processor.feature_extractor(
                    y, sampling_rate=self.audio_processor.target_sr
                ).input_features[0]
                
                # 3. Tokenize label text
                labels = self.processor.tokenizer(text).input_ids
                
                return {
                    "input_features": torch.tensor(input_features, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "language_code": lang_code
                }
            except Exception as e:
                # Log file error and try next sample in list (circularly)
                # print(f"Warning: Skipping index {current_idx} ({file_path}) due to processing error: {e}")
                idx = (idx + 1) % len(self.indices)
                attempts += 1
                
        raise ValueError("SautiEdgeDataset: No valid audio files found in the dataset list.")
