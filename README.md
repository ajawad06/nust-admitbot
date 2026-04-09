markdown

# nust-admission-assistant

A fully offline RAG-based chatbot that answers NUST admissions questions using verified FAQ data. No internet required at runtime.

![UI Screenshot](screenshot.png)

---

## What it does

Students ask questions about NUST admissions — NET, eligibility, scholarships, hostels, programmes — and the bot answers strictly from NUST's official FAQ data. No hallucinations, no cloud, no internet.

---

## How it works

**Retrieval-Augmented Generation (RAG) pipeline:**

1. Student types a question
2. Question is embedded into a semantic vector
3. FAISS searches an offline index for the 3 most relevant FAQ chunks
4. Chunks are passed to a local LLM (`qwen2.5:1.5b`) as context
5. LLM generates a grounded answer — strictly from FAQ data

---

## Project structure

```
nust-admission-assistant/
│
├── data/
│   └── faqs.json              # NUST FAQ Q&A pairs
│
├── index/
│   ├── faqs.index             # FAISS vector index
│   └── faqs_texts.json        # raw text chunks
│
├── build_index.py             # builds FAISS index from faqs.json
├── chatbot.py                 # RAG query engine
├── app.py                     # Streamlit chat UI
└── requirements.txt
```

---

## Tech stack

| Component | Tool |
|---|---|
| LLM | qwen2.5:1.5b via Ollama (local, CPU) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector search | FAISS (faiss-cpu) |
| UI | Streamlit |
| Language | Python 3.14 |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/nust-admission-assistant
cd nust-admission-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull the model**
```bash
ollama pull qwen2.5:1.5b
```

**4. Build the index**
```bash
python build_index.py
```

**5. Run the app**
```bash
python -m streamlit run app.py
```

Open **http://localhost:8501**

---

## Hardware

- CPU: Intel Core i5 (no dedicated GPU)
- RAM: 8GB
- OS: Windows 11
- 100% offline after setup

---

## Competition

Built for **Local Chatbot Competition 2026** — NUST Islamabad  
Organized by Dr. Sohail Iqbal

---

## License

MIT
