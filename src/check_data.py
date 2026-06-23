from datasets import load_dataset

print("Downloading a tiny slice of open Swahili data (Google FLEURS)...")
# google/fleurs is completely open, no token needed. sw_ke is Swahili.
dataset = load_dataset("google/fleurs", "sw_ke", split="test[:3]")

print("\nSuccess! Here is what the data looks like:\n")

for i, row in enumerate(dataset):
    print(f"--- Sample {i+1} ---")
    print(f"Transcript: {row['transcription']}")
    
    audio_array = row['audio']['array']
    sampling_rate = row['audio']['sampling_rate']
    duration = len(audio_array) / sampling_rate
    
    print(f"Audio Length: {duration:.2f} seconds")
    print("-" * 30)
