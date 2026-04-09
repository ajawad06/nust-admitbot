# NUST Admission Assistant

**A fully offline, RAG-based chatbot for answering NUST admissions questions - powered entirely by official FAQ data.**  

---

## 🚀 Overview

**NUST Admission Assistant** allows students to get instant, accurate answers about:

- NET & admission procedures  
- Eligibility criteria  
- Scholarships & financial aid  
- Hostels & accommodation  
- Programmes & course details  

All answers are **grounded in NUST's official FAQ data** — no internet, no cloud, no hallucinations.  

---

## 💡 How It Works

This project uses a **Retrieval-Augmented Generation (RAG)** pipeline:

1. Student asks a question in the chat.  
2. The question is embedded into a semantic vector.  
3. **FAISS** searches the offline index for the 3 most relevant FAQ chunks.  
4. These chunks are provided to a **local LLM (`qwen2.5:1.5b`)** as context.  
5. The LLM generates a precise, grounded answer strictly based on the FAQ data.  

---

## 🗂 Project Structure
```bash
nust-admission-assistant/
│
├── data/
│ └── faqs.json # Official NUST FAQ Q&A pairs
│
├── index/
│ ├── faqs.index # FAISS vector index
│ └── faqs_texts.json # Raw text chunks for indexing
│
├── build_index.py # Script to build FAISS index from faqs.json
├── chatbot.py # RAG query engine
├── app.py # Streamlit-based chat UI
└── requirements.txt
```

---

## 🛠 Tech Stack

| Component       | Tool / Library |
|-----------------|----------------|
| Local LLM       | `qwen2.5:1.5b` via Ollama (CPU) |
| Embeddings      | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Search   | FAISS (faiss-cpu) |
| UI              | Streamlit |
| Language        | Python 3.14 |

---

## ⚡ Setup Instructions

**1. Clone the repository**  

```bash
git clone https://github.com/ajawad06/nust-admission-bot.git
cd nust-admission-bot
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Pull the local LLM**

```bash
ollama pull qwen2.5:1.5b
```

**4. Build the FAISS index**

```bash
python build_index.py
```

**5. Launch the Streamlit app**

```bash
python -m streamlit run app.py
```

---

## 💻 Hardware Requirements

- CPU: Intel Core i5
- RAM: Min. 8GB
- OS: Windows 11 (or linux/macOS)
- Fully offline after setup

---
