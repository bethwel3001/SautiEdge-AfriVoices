import os
from datasets import load_dataset
from pathlib import Path

# Make sure you export your NEW token in the terminal before running this
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable.")

DATASETS_TO_DOWNLOAD = [
    "Anv-ke/Maasai",
    "Anv-ke/kikuyu", 
    "Anv-ke/Kalenjin",
]

def download_and_save_datasets():
    base_dir = Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS_TO_DOWNLOAD:
        print(f"\n--- Downloading {dataset_name} ---")
        try:
            dataset = load_dataset(dataset_name, token=HF_TOKEN)
            lang_folder = base_dir / dataset_name.split("/")[-1].lower()
            lang_folder.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(lang_folder))
            print(f"Successfully saved {dataset_name} to {lang_folder}")
        except Exception as e:
            print(f"Failed to download {dataset_name}. Error: {e}")

if __name__ == "__main__":
    download_and_save_datasets()
