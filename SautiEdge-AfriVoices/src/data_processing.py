# Data loading, cleaning, and augmentation pipeline
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

def get_augmentation_pipeline():
    return Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    ])
