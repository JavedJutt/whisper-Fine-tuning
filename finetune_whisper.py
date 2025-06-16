import os
import json
import numpy as np
from datasets import Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import torch
from typing import Any, Dict, List, Union
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence

# === Load and Process Dataset ===
jsonl_dir = 'whisper_Dataset'
all_data = []

print(f"Starting to process files in directory: {jsonl_dir}")

for filename in os.listdir(jsonl_dir):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(jsonl_dir, filename)
        print(f"Processing file: {filename}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entry['audio']['array'] = np.array(entry['audio']['array'], dtype=np.float32)
                all_data.append(entry)

print(f"Finished processing all files.")
print(f"Total entries loaded: {len(all_data)}")

dataset = Dataset.from_list(all_data)

# === Load Whisper Components ===
print("Loading feature extractor and tokenizer...")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
print("Feature extractor loaded.")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", task="transcribe")
print("Tokenizer loaded.")

print("Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
print("Whisper model loaded.")
model.config.forced_decoder_ids = None

# === Dataset Preprocessing ===
def prepare_dataset(batch):
    batch["input_features"] = feature_extractor(
        batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preparing dataset...")
dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])
print("Dataset prepared.")

# === Custom Data Collator ===
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data Collator for Whisper fine-tuning with audio inputs + label padding
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Wrap processor
processor = SimpleNamespace(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding()
data_collator.processor = processor
print("Data collator set up.")

# === Training Arguments ===
print("Setting up training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=3000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)
print("Training arguments set up.")

# === Evaluation Metric (WER) ===
print("Loading WER metric...")
metric = evaluate.load("wer")
print("WER metric loaded.")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 🛠 If it's a tuple (some models return tuple), get the first element
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # 🛠 Convert logits to predicted token IDs if needed
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    # 🛠 Replace label padding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # ✅ Decode properly
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# === Trainer Setup ===
print("Setting up trainer...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,  # Ideally use a separate validation set
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
print("Trainer set up.")

# === Start Training ===
print("Starting training...")
trainer.train()
print("Training completed.")

# === Save Model ===
print("Saving model...")
trainer.save_model("./whisper-large-finetuned")
print("Model saved.")
