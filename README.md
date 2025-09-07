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
- Huggingface_projects/     #you can name your project 
- │── main.py # Training & evaluation script
- │── requirements.txt # Dependencies
- │── myenv310/ # Virtual environment (not uploaded to repo)
- │── data/ # Local audio dataset 


---

## ⚡ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/audio-recognizer.git
cd audio-recognizer
```

### 2. Create a virtual environment
```bash
python -m venv myenv310
# Activate (Windows)
myenv310\Scripts\activate
# Activate (Linux/macOS)
source myenv310/bin/activate
```

### 3. Install dependencies
pip install -r requirements.txt

### ▶️ Usage
Run Training/Evaluation
```bash
python main.py
```
Your dataset should be placed under the data/ folder. Update the paths inside main.py if needed.


### 🛠️ Tech Stack

Python 3.10

PyTorch + Torchaudio

Hugging Face Transformers

Hugging Face Datasets

### 📜 License

This project is released under the MIT License.

### 🙌 Acknowledgements

Hugging Face Transformers

HuBERT Model

PyTorch
