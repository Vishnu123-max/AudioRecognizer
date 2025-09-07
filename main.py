import os
import torch
import torchaudio
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer

# 1. Pretrained model for speaker ID
MODEL_NAME = "superb/hubert-base-superb-sid"

# 2. Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# Define resampler (convert any sr -> 16000)
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

# 3. Prepare dataset
def load_audio(file_path, label):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = resampler(waveform)
        sr = 16000
    return {"speech": waveform.squeeze().numpy(), "sampling_rate": sr, "labels": label}

# Local dataset
data_files = [
    ("data/Vishnu.wav", 0),
    ("data/Raviraj.wav", 1)
]

dataset = Dataset.from_list([load_audio(f, l) for f, l in data_files])

# 4. Preprocess
def preprocess(batch):
    inputs = feature_extractor(
        batch["speech"],
        sampling_rate=16000,
        return_tensors="pt"
    )
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    return batch

dataset = dataset.map(preprocess)

# 5. Label mapping
id2label = {0: "Vishnu", 1: "Raviraj"}
label2id = {"Vishnu": 0, "Raviraj": 1}

# 6. Load model
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 7. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 8. Custom collate function to handle variable-length audio
def collate_fn(batch):
    input_values = [torch.tensor(x["input_values"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = torch.tensor([x["labels"] for x in batch])

    # pad sequences to max length in batch
    input_values = pad_sequence(input_values, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=collate_fn
)

# 10. Train
trainer.train()

# 11. Inference
test_file = "data/test_2.wav"
waveform, sr = torchaudio.load(test_file)
if sr != 16000:
    waveform = resampler(waveform)

inputs = feature_extractor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class = id2label[int(torch.argmax(logits))]
print(f"Prediction: {predicted_class}")
