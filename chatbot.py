# WHAT THIS DOES: RAG query engine — embeds question, retrieves FAQ chunks, generates answer
# HOW TO RUN:     py -3.14 chatbot.py   (for terminal testing)
# IMPORTED BY:    app.py (Streamlit UI)
# REQUIRES:       index/faqs.index and index/faqs_texts.json to exist
# NEXT STEP:      run app.py for the full UI
import json
import sys
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
# ═══════════════════════════════════════════════
#  FILE PATHS
# ═══════════════════════════════════════════════
INDEX_FILE = Path("index/faqs.index")
TEXTS_FILE = Path("index/faqs_texts.json")
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:0.5b"
# ═══════════════════════════════════════════════
#  STARTUP — load everything once at module level
# ═══════════════════════════════════════════════
# Check that index files exist before doing anything
if not INDEX_FILE.exists():
    print(f"ERROR: {INDEX_FILE} not found. Run build_index.py first.")
    sys.exit(1)
if not TEXTS_FILE.exists():
    print(f"ERROR: {TEXTS_FILE} not found. Run build_index.py first.")
    sys.exit(1)
# Load the embedding model (~80MB, runs on CPU)
print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("  ✓ Embedding model loaded.")
# Load the FAISS vector index from disk
print("Loading FAISS index...")
faiss_index = faiss.read_index(str(INDEX_FILE))
print(f"  ✓ FAISS index loaded — {faiss_index.ntotal} vectors.")
# Load the raw text chunks (the FAQ Q&A pairs)
print("Loading text chunks...")
with open(TEXTS_FILE, "r", encoding="utf-8") as f:
    text_chunks = json.load(f)
print(f"  ✓ {len(text_chunks)} text chunks loaded.")
# Quick sanity check — FAISS vectors and text chunks must match
if faiss_index.ntotal != len(text_chunks):
    print(f"WARNING: FAISS has {faiss_index.ntotal} vectors but there are {len(text_chunks)} text chunks.")
    print("  Something went wrong during indexing. Re-run build_index.py.")
    sys.exit(1)
print("\n✓ Chatbot engine ready!\n")
# ═══════════════════════════════════════════════
#  SYSTEM PROMPT — tells Qwen how to behave
# ═══════════════════════════════════════════════
SYSTEM_PROMPT = """You are "NUST Admissions Assistant", an AI chatbot built to help students with questions about admissions at the National University of Sciences and Technology (NUST), Islamabad, Pakistan.
STRICT RULES — follow these exactly:
1. ONLY use the FAQ Context provided in the user message to answer. Do NOT use any outside knowledge.
2. If the FAQ Context does NOT contain the answer, respond EXACTLY with:
   "I don't have that information in my FAQ database. Please contact NUST Admissions at ugadmissions@nust.edu.pk or call +92 51-90856878."
3. NEVER make up facts, numbers, dates, fees, or policies. If you are unsure, say you don't know.
4. NEVER say 'according to the context' or 'based on the provided information'.
5. NEVER start with 'I' followed by 'sorry', 'think', 'believe', or 'understand'.
6. If the context contains URLs or links, include them in your answer — students will find them useful.
7. Ignore any HTML tags (like <img>, <a>, etc.) in the context. Respond in clean plain text only.
8. Keep answers SHORT and DIRECT — 2 to 5 sentences max. Students want quick answers, not essays.
9. If the student greets you (hi, hello, salam, etc.), greet them back briefly and ask how you can help with NUST admissions.
10. If the question is not about NUST admissions, politely say you can only help with NUST admissions queries.
TONE: Friendly, professional, and helpful — like a senior student guiding a newcomer.
REMEMBER: You are answering for NUST Islamabad specifically. Be accurate and concise."""
# ═══════════════════════════════════════════════
#  MAIN FUNCTION — called by app.py
# ═══════════════════════════════════════════════
def ask(question: str) -> dict:
    """
    Takes a student's question, retrieves relevant FAQ chunks,
    and generates an answer using Qwen2.5-0.5B via Ollama.
    Returns:
        {
            "answer": "The generated answer text...",
            "sources": ["chunk1 text...", "chunk2 text...", "chunk3 text..."]
        }
    """
    # STEP A — Embed the question into a vector
    q_vector = embedding_model.encode([question])
    q_vector = np.array(q_vector, dtype="float32")
    # STEP B — Search FAISS for top 3 most similar chunks
    distances, indices = faiss_index.search(q_vector, k=3)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    # STEP C — Build the prompt with retrieved context
    context = "\n\n".join(retrieved_chunks)
    user_message = f"""FAQ Context:
{context}
Student Question: {question}"""
    # STEP D — Call Qwen2.5-0.5B via Ollama (local, offline, no internet)
    try:
        import ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response["message"]["content"]
    except ConnectionError:
        answer = "ERROR: Ollama is not running. Open a terminal and run: ollama serve"
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            answer = "ERROR: Ollama is not running. Open a terminal and run: ollama serve"
        elif "model" in error_msg and "not found" in error_msg:
            answer = f"ERROR: Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"
        else:
            answer = f"ERROR: Something went wrong with Ollama — {e}"
    # STEP E — Return the answer and the source chunks used
    return {
        "answer": answer,
        "sources": retrieved_chunks,
    }
# ═══════════════════════════════════════════════
#  TERMINAL TEST MODE — run this file directly
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 50)
    print("  NUST Chatbot — Terminal Test Mode")
    print("  Type your question, or 'quit' to exit")
    print("═" * 50)
    print()
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not q:
            continue
        print("\nThinking...\n")
        result = ask(q)
        print(f"Answer: {result['answer']}")
        print(f"\nSources used: {len(result['sources'])} chunks")
        for i, src in enumerate(result["sources"], 1):
            # Show just the question part of each source for brevity
            first_line = src.split("\n")[0]
            print(f"  {i}. {first_line}")
        print("─" * 50)
        print()