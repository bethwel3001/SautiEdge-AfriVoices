import os
import yaml
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model

from src.dataset import SautiEdgeDataset

# Mapping of strict ISO codes to Whisper language identifiers
ISO_TO_WHISPER = {
    "swa": "sw",
    "som": "so",
    "kik": "sw",  # Use Swahili as a linguistic and phonetic proxy for Kikuyu
    "luo": "sw",  # Use Swahili as a proxy for Luo
    "mas": "sw",  # Use Swahili as a proxy for Maasai
    "kln": "sw",  # Use Swahili as a proxy for Kalenjin
}

@dataclass
class SautiEdgeDataCollator:
    """
    Custom speech collator that dynamically structures decoder input targets per item.
    Injects language-specific special tokens in a thread-safe manner.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Collate and pad the audio log-mel features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2. Re-structure labels with language prefixes
        labels_list = []
        for feature in features:
            lang_code = feature["language_code"]
            raw_labels = feature["labels"]
            if isinstance(raw_labels, torch.Tensor):
                raw_labels = raw_labels.tolist()

            # Slice off default prefix [50258, 50363] (<|startoftranscript|>, <|notimestamps|>)
            if len(raw_labels) >= 2 and raw_labels[0] == 50258 and raw_labels[1] == 50363:
                sliced_labels = raw_labels[2:]
            else:
                if len(raw_labels) > 0 and raw_labels[0] == 50258:
                    sliced_labels = raw_labels[1:]
                else:
                    sliced_labels = raw_labels

            # Map ISO code to Whisper language token ID
            whisper_lang = ISO_TO_WHISPER.get(lang_code, "sw")
            lang_token = self.processor.tokenizer.lang_code_to_id.get(whisper_lang, 50318) # default to Swahili

            # Reconstruct target labels: [language, transcribe, notimestamps] + transcription text + [eos]
            # We omit <|startoftranscript|> because shift_tokens_right in Whisper automatically prepends it
            target_labels = [lang_token, 50359, 50363] + sliced_labels
            labels_list.append({"input_ids": target_labels})

        # Pad labels using tokenizer padding
        labels_batch = self.processor.tokenizer.pad(labels_list, return_tensors="pt")

        # Mask padding in labels with -100 so CrossEntropyLoss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        # Preserve language codes in the batch to update generation config during evaluation
        batch["language_code"] = [f["language_code"] for f in features]

        return batch


class SautiEdgeTrainer(Seq2SeqTrainer):
    """
    Subclass Hugging Face Seq2SeqTrainer to pop custom metadata fields (like language_code) 
    before forward passes and dynamically configure Whisper's generation target language during validation.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        # Remove language_code metadata from model inputs to prevent forward call failures
        inputs.pop("language_code", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Extract language_code to set target language dynamically for generation evaluation
        language_codes = inputs.pop("language_code", None)
        
        if not prediction_loss_only and language_codes is not None:
            # Detect language of the batch (using the first sample)
            lang = language_codes[0]
            whisper_lang = ISO_TO_WHISPER.get(lang, "sw")
            
            # Configure Whisper model generation config dynamically for this batch
            model.generation_config.language = whisper_lang
            model.generation_config.task = "transcribe"
            model.generation_config.forced_decoder_ids = None

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


def main():
    # 1. Load configuration file
    config_path = "configs/model_config.yaml"
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    batch_size = config["training"]["batch_size"]
    learning_rate = float(config["training"]["learning_rate"])
    epochs = config["training"]["epochs"]
    lora_r = config["training"]["lora_rank"]
    lora_alpha = config["training"]["lora_alpha"]
    beam_size = config["decoding"]["beam_size"]

    print(f"Base model: {model_name}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    # 2. Load processor and base model
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Disable KV cache for training compatibility with gradient checkpointing/PEFT
    model.config.use_cache = False

    # 3. Configure LoRA PEFT adapters
    print("Initializing LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load datasets from disk
    dataset_path = "data/processed/afrivoices_unified"
    print(f"Loading processed dataset from {dataset_path}...")
    hf_dataset = load_from_disk(dataset_path)

    # Initialize PyTorch Datasets
    train_dataset = SautiEdgeDataset(
        audio_paths=hf_dataset["train"]["audio_path"],
        transcripts=hf_dataset["train"]["transcription"],
        language_codes=hf_dataset["train"]["language_code"],
        processor=processor,
        apply_augment=True,
    )
    val_dataset = SautiEdgeDataset(
        audio_paths=hf_dataset["validation"]["audio_path"],
        transcripts=hf_dataset["validation"]["transcription"],
        language_codes=hf_dataset["validation"]["language_code"],
        processor=processor,
        apply_augment=False,
    )

    # 5. Define evaluation metrics (WER)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace padding labels with tokenizer pad token ID
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode to text strings
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate Word Error Rate (WER)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 6. Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints/whisper-small-sautiedge",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=50,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=beam_size,
        logging_steps=10,
        report_to="none",  # Avoid external dependencies like wandb
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Thread-safe collation
        remove_unused_columns=False,
    )

    # Initialize custom collator
    data_collator = SautiEdgeDataCollator(processor=processor)

    # 7. Initialize Trainer
    trainer = SautiEdgeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # 8. Run training
    print("Starting LoRA fine-tuning training loop...")
    trainer.train()

    # 9. Save final adapters and processor config
    output_adapters_dir = "checkpoints/lora_unified/"
    print(f"Training completed. Saving best model adapters to {output_adapters_dir}...")
    model.save_pretrained(output_adapters_dir)
    processor.save_pretrained(output_adapters_dir)
    print("Save complete. SautiEdge Model is ready.")


if __name__ == "__main__":
    main()
