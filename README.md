# Medical Chatbot (RAG-Powered)

A Retrieval-Augmented Generation (RAG) medical chatbot that can answer health-related questions using either the **Google Gemini API** (cloud, fast, requires API key) or a **local GGUF LLM** (runs on your machine, no API needed, requires model download).  
The chatbot saves conversation history, supports semantic search over previous chats, and can be easily switched between cloud and local models.

---

## Features

- **Retrieval-Augmented Generation:** Answers are grounded in your document database.
- **Conversation Memory:** Remembers previous exchanges for context.
- **Timeout Handling:** If the model takes too long, you get a timeout message.
- **Conversation Logs:** All chats are saved as JSON files for review.
- **Switchable Backend:** Use Gemini API or run fully local with GGUF models.
- **Command Support:** Save, list, and load previous conversations.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Vkushwahaa/Medical_chatbot.git
cd medical_chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in your project root with the following content:

```
GEMINI_API=your_gemini_api_key_here
DB_FAISS_PATH=/absolute/path/to/your/vectorstore/db_faiss
CONVERSATIONS_PATH=/absolute/path/to/your/conversations
MODEL = your downloaded model path
```

- **GEMINI_API**: Your Google Gemini API key ([get from Google AI Studio](https://aistudio.google.com/app/apikey)).
- **DB_FAISS_PATH**: Path to your FAISS vectorstore (for document retrieval).
- **CONVERSATIONS_PATH**: Directory to save conversation logs.

---

## Running the Chatbot

```bash
python iteration_6.py
```

You’ll see a prompt. Type your medical question, or use commands like `help`, `save`, `list`, `load <id>`, or `exit`.

---

## Switching Between Gemini API and Local GGUF Model

### A. Using Gemini API (Default)

- The code uses Gemini API if `load_llm()` is set to `ChatGoogleGenerativeAI`.
- Requires a valid API key in `.env`.

### B. Using Local GGUF Model with llama.cpp

1. **Download a GGUF Model**

   - Visit [TheBloke's Hugging Face page](https://huggingface.co/TheBloke) or another trusted source.
   - Download a model file (e.g., `gemma-2b-it.gguf` or `llama-2-7b-chat.gguf`).
   - Place it in a folder, e.g., `models/`.

   Example download command:

   ```bash
   wget https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf 
   ```
   
<img width="1317" alt="Screenshot 2025-05-05 at 3 28 35 PM" src="https://github.com/user-attachments/assets/e6132dc4-5309-405b-b9a9-54967cbc00b5" />
You may sometimes have to fill out a form for the usage of this model.


2. **Install llama-cpp-python**

   ```bash
   pip install llama-cpp-python
   ```

3. **Edit the `load_llm()` function in your code:**

   ```python
   # Comment out Gemini and uncomment LlamaCpp:
   # from langchain_community.llms import LlamaCpp

   def load_llm():
       return LlamaCpp(
           model_path=os.environ.get("MODEL"),
           n_gpu_layers=-1,
           n_batch=256,
           f16_kv=True,
           n_ctx=1024,
           max_tokens=256,
           temperature=0.6,
           top_p=0.9,
           verbose=False,
       )
   ```

4. **Run the chatbot again:**

   ```bash
   python iteration_6.py
   ```

---

## Commands

- **Ask a question:** Just type your medical question.
- **help:** Show available commands.
- **save:** Save the current conversation.
- **list:** List all saved conversations.
- **load <id>:** Load a previous conversation by ID.
- **exit/quit/bye:** End the conversation and save.

---

## Data Storage

- **Conversation logs:** Saved as JSON files in the directory specified by `CONVERSATIONS_PATH`.
- **Vectorstore:** FAISS vector database at `DB_FAISS_PATH` for semantic retrieval.

---

## Troubleshooting

- **Model not found:** Double-check your `model_path` and that the file is a valid `.gguf`.
- **API errors:** Make sure your API key is correct and you have internet access.
- **Slow responses:** Try a smaller model, reduce `n_ctx`, or lower `max_tokens`.

---



