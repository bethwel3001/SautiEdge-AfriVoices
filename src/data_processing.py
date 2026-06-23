import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from audiomentations import Compose, AddGaussianNoise, PitchShift

class GarbageAudioError(ValueError):
    """Custom exception raised when audio is flagged as garbage/invalid."""
    pass

class AudioProcessor:
    def __init__(self, target_sr=16000, min_duration=0.5, max_duration=20.0, 
                 clipping_threshold=0.99, max_clipping_ratio=0.1, 
                 silence_threshold=0.001, apply_augment=False):
        """
        Initializes the AudioProcessor.
        
        Args:
            target_sr (int): Target sampling rate (default: 16000).
            min_duration (float): Minimum audio duration in seconds.
            max_duration (float): Maximum audio duration in seconds.
            clipping_threshold (float): Amplitude threshold to detect clipped samples.
            max_clipping_ratio (float): Maximum ratio of clipped samples allowed.
            silence_threshold (float): Minimum RMS amplitude to consider audio not silent.
            apply_augment (bool): Whether to apply augmentations by default.
        """
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.clipping_threshold = clipping_threshold
        self.max_clipping_ratio = max_clipping_ratio
        self.silence_threshold = silence_threshold
        self.apply_augment = apply_augment
        
        # Define basic augmentations compatible with python 3.13 (no C++ compiler issues)
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        ])

    def load_audio(self, file_path):
        """
        Loads and resamples audio to target_sr mono waveform as a numpy array.
        
        Args:
            file_path (str): Path to the audio file.
            
        Returns:
            np.ndarray: Waveform array (float32).
        """
        if not os.path.exists(file_path):
            raise GarbageAudioError(f"Audio file does not exist: {file_path}")
            
        try:
            # Attempt load using torchaudio
            waveform, sr = torchaudio.load(file_path)
            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample if needed
            if sr != self.target_sr:
                resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
            y = waveform.squeeze(0).numpy()
        except Exception as e:
            # Fallback load using soundfile
            try:
                data, sr = sf.read(file_path, dtype='float32')
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if sr != self.target_sr:
                    tensor_waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                    resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
                    resampled_waveform = resampler(tensor_waveform)
                    y = resampled_waveform.squeeze(0).numpy()
                else:
                    y = data
            except Exception as e_inner:
                raise GarbageAudioError(f"Failed to load audio file {file_path}. Torchaudio error: {e}. Soundfile error: {e_inner}")
        
        return y

    def validate_audio(self, y):
        """
        Checks if the audio waveform is 'garbage' according to duration, clipping, and silence.
        
        Args:
            y (np.ndarray): Waveform array.
            
        Returns:
            bool: True if audio is valid.
            
        Raises:
            GarbageAudioError: If audio fails any validation check.
        """
        duration = len(y) / self.target_sr
        
        # 1. Duration check
        if duration < self.min_duration:
            raise GarbageAudioError(f"Audio duration {duration:.2f}s is below minimum limit {self.min_duration}s")
        if duration > self.max_duration:
            raise GarbageAudioError(f"Audio duration {duration:.2f}s exceeds maximum limit {self.max_duration}s")
            
        # 2. Extreme clipping check
        clipping_samples = np.sum(np.abs(y) >= self.clipping_threshold)
        clipping_ratio = clipping_samples / len(y)
        if clipping_ratio > self.max_clipping_ratio:
            raise GarbageAudioError(f"Audio has extreme clipping (ratio {clipping_ratio:.2%})")
            
        # 3. Dead silence check
        rms = np.sqrt(np.mean(y**2))
        if rms < self.silence_threshold:
            raise GarbageAudioError(f"Audio is dead silence (RMS {rms:.6f} < threshold {self.silence_threshold})")
            
        return True

    def augment_audio(self, y):
        """
        Applies basic augmentations using audiomentations.
        
        Args:
            y (np.ndarray): Waveform array.
            
        Returns:
            np.ndarray: Augmented waveform array.
        """
        return self.augment(samples=y, sample_rate=self.target_sr)

    def process(self, file_path, apply_augment=None):
        """
        Loads, validates, and optionally augments audio from file_path.
        
        Args:
            file_path (str): Path to the audio file.
            apply_augment (bool, optional): Override the class default apply_augment.
            
        Returns:
            np.ndarray: Processed waveform.
        """
        y = self.load_audio(file_path)
        self.validate_audio(y)
        
        should_augment = apply_augment if apply_augment is not None else self.apply_augment
        if should_augment:
            y = self.augment_audio(y)
            
        return y
