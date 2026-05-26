# NUST Admission Assistant

An offline NUST admissions chatbot that runs locally on your machine. It uses official FAQ data, a local FAISS index, and a FastAPI backend with a plain HTML/CSS/JS frontend.

## Overview

NUST Admission Assistant helps students get quick answers about:

- NET and admission procedures
- Eligibility criteria
- Scholarships and financial aid
- Hostels and accommodation
- Undergraduate programmes
- Reserved seats and related admission policies

The bot is designed to work without internet after setup. Normal FAQ questions are answered directly from the local FAQ data and semantic index, so responses are typically well under 10 seconds.

## How It Works

1. The student asks a question in the browser UI.
2. The frontend sends the question to the local FastAPI backend at `/api/chat`.
3. The chatbot first tries a fast direct/fuzzy FAQ match.
4. If needed, it embeds the question with `all-MiniLM-L6-v2` and searches the offline FAISS index.
5. The best local FAQ answer is returned with source chunks.

Ollama is no longer required for normal FAQ responses. It can still be enabled as an optional fallback by setting:

```bash
USE_OLLAMA_FALLBACK=1
```

## Project Structure

```text
nust_admission_chatbot/
|-- data/
|   `-- faqs.json              # Official NUST FAQ Q&A pairs
|-- index/
|   |-- faqs.index             # FAISS vector index
|   `-- faqs_texts.json        # Raw indexed FAQ chunks
|-- static/
|   |-- index.html             # Browser chat UI
|   |-- styles.css             # Frontend styling
|   `-- app.js                 # Frontend chat logic
|-- app.py                     # FastAPI backend
|-- chatbot.py                 # Local FAQ query engine
|-- build_index.py             # Rebuilds the FAISS index
`-- requirements.txt
```

## Tech Stack

| Component | Tool / Library |
| --- | --- |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector search | FAISS (`faiss-cpu`) |
| Optional LLM fallback | `qwen2.5:1.5b` via Ollama |
| Language | Python |

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Build or rebuild the FAISS index if needed.

```bash
python build_index.py
```

4. Start the local FastAPI server.

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

5. Open the chatbot.

```text
http://127.0.0.1:8000/
```

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Serves the local chat UI |
| `/api/health` | GET | Checks local backend status and FAQ count |
| `/api/chat` | POST | Answers a question from local FAQ data |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Are there any quota seats?\"}"
```

## Offline Notes

- The FAQ data and FAISS index are stored locally.
- The embedding model must already be available in the local Hugging Face cache.
- The app sets offline Hugging Face environment flags in `chatbot.py` and `build_index.py`.
- Internet is not required for normal chatbot use once dependencies, model files, and the index are present.

## Performance

The chatbot avoids calling a local LLM for normal FAQ answers. Direct FAQ matches are usually answered in a few milliseconds after startup, while semantic FAQ matches are typically far below the 10 second target.

First startup can take a few seconds because the embedding model is loaded into memory.
