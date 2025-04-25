# Medical Chatbot

A lightweight, offline-capable medical Q&A chatbot using the Llama-Doctor 3.2 3B Instruct GGUF model via llama.cpp.

## Prerequisites
- Python 3.9+
- `git` and Hugging Face account

## Setup
1. **Clone & enter repo**
   ```bash
   git clone https://github.com/yourusername/medical_chatbot.git
   cd medical_chatbot
   ```
2. **Create & activate venv**
   ```bash
   python3 -m venv venv && source venv/bin/activate
   ```
3. **Install deps**
   ```bash
   pip install -r requirements.txt
   huggingface-cli login
   ```

## Download Model
```bash
https://huggingface.co/bartowski/Llama-Doctor-3.2-3B-Instruct-GGUF
```

## Run Chatbot
```bash
python run_bot.py --model ./models/Llama-Doctor-3.2-3B-Instruct.gguf
```


