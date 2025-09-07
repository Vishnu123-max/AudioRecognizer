# AudioRecognizer
Audio Recognizer is a lightweight project that demonstrates how to fine-tune and evaluate speech/audio classification models using Hugging Face’s transformers, datasets, and PyTorch/Torchaudio. It uses the HuBERT model  for audio sequence classification and can be adapted to speaker identification, speech emotion recognition, or other custom tasks.

# 🎤 Audio Recognizer

Audio Recognizer is a speech/audio classification project built with [Hugging Face Transformers](https://huggingface.co/transformers/), [PyTorch](https://pytorch.org/), and [Torchaudio](https://pytorch.org/audio/stable/index.html).  
It demonstrates how to train, fine-tune, and evaluate models like **HuBERT** on custom audio datasets.

---

## ✨ Features
- Uses Hugging Face `transformers` and `datasets`
- Works with local datasets (no need for cloud storage)
- Configurable training pipeline with `TrainingArguments`
- Built-in support for padding/collation of variable-length audio inputs
- Ready for fine-tuning on tasks like:
  - Speaker Identification
  - Emotion Recognition
  - General Audio Classification

---

## 📂 Project Structure
