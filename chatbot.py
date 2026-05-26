"""Local FAQ query engine for the NUST admissions chatbot.

The fast path answers directly from the local FAQ dataset and FAISS index. This
keeps normal FAQ responses comfortably below 10 seconds without requiring
internet access or an LLM call.
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("data/faqs.json")
INDEX_FILE = Path("index/faqs.index")
TEXTS_FILE = Path("index/faqs_texts.json")
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:1.5b"
OFF_TOPIC_THRESHOLD = 1.5
DIRECT_MATCH_THRESHOLD = 0.76
SEMANTIC_DIRECT_THRESHOLD = 1.15
USE_OLLAMA_FALLBACK = os.getenv("USE_OLLAMA_FALLBACK", "0").lower() in {"1", "true", "yes"}


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"ERROR: {path} not found. {hint}")
        sys.exit(1)


def _clean_text(text: str) -> str:
    return (
        text.replace("Â\xa0", " ")
        .replace("Â", "")
        .replace("\xa0", " ")
        .replace("\t", " ")
        .strip()
    )


def _normalize(text: str) -> str:
    text = _clean_text(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_faqs() -> list[dict]:
    _require_file(DATA_FILE, "Keep data/faqs.json in the project folder.")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    cleaned = []
    for item in faqs:
        question = _clean_text(item.get("question", ""))
        answer = _clean_text(item.get("answer", ""))
        if question and answer:
            cleaned.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": _clean_text(item.get("source", "NUST FAQ")),
                    "normalized_question": _normalize(question),
                }
            )
    return cleaned


_require_file(INDEX_FILE, "Run build_index.py first.")
_require_file(TEXTS_FILE, "Run build_index.py first.")

embedding_model = SentenceTransformer(MODEL_NAME, local_files_only=True)
faiss_index = faiss.read_index(str(INDEX_FILE))

with open(TEXTS_FILE, "r", encoding="utf-8") as f:
    text_chunks = [_clean_text(chunk) for chunk in json.load(f)]

faqs = _load_faqs()

if faiss_index.ntotal != len(text_chunks):
    print(f"WARNING: FAISS has {faiss_index.ntotal} vectors but {len(text_chunks)} text chunks.")
    print("Re-run build_index.py to fix this.")
    sys.exit(1)


SYSTEM_PROMPT = """You are "NUST Admissions Assistant", an AI chatbot for NUST Islamabad admissions.

STRICT RULES:
1. ONLY use the FAQ Context provided to answer. Do NOT use outside knowledge.
2. If the answer is not in the FAQ Context, say ONLY: "I don't have that information. Contact ugadmissions@nust.edu.pk or call +92 51-90856878."
3. NEVER make up facts, numbers, dates, fees, or policies.
4. Keep answers SHORT - 2 to 4 sentences.

TONE: Direct and factual. No filler words. No fluff. Just the answer."""


def _faq_to_source(faq: dict) -> str:
    source = faq.get("source") or "NUST FAQ"
    return f"Q: {faq['question']}\nA: {faq['answer']}\nSource: {source}"


def _extract_answer(chunk: str) -> str:
    if "\nA:" in chunk:
        return _clean_text(chunk.split("\nA:", 1)[1])
    return _clean_text(chunk)


def _direct_match(question: str) -> tuple[dict | None, float]:
    normalized = _normalize(question)
    if not normalized:
        return None, 0.0

    best_faq = None
    best_score = 0.0
    for faq in faqs:
        faq_question = faq["normalized_question"]
        if normalized == faq_question:
            return faq, 1.0
        if normalized in faq_question or faq_question in normalized:
            score = min(len(normalized), len(faq_question)) / max(len(normalized), len(faq_question))
            score = max(score, 0.88)
        else:
            score = SequenceMatcher(None, normalized, faq_question).ratio()
        if score > best_score:
            best_faq = faq
            best_score = score

    if best_score >= DIRECT_MATCH_THRESHOLD:
        return best_faq, best_score
    return None, best_score


def _semantic_search(question: str, k: int = 3) -> tuple[float, list[int], list[str]]:
    q_vector = np.array(embedding_model.encode([question]), dtype="float32")
    distances, indices = faiss_index.search(q_vector, k=k)
    valid_indices = [int(i) for i in indices[0] if int(i) >= 0]
    chunks = [text_chunks[i] for i in valid_indices]
    return float(distances[0][0]), valid_indices, chunks


def _ollama_answer(question: str, chunks: list[str]) -> str:
    import ollama

    context = "\n\n".join([c[:500] for c in chunks])
    user_message = f"""FAQ Context:
{context}
Student Question: {question}
Answer directly in 2-4 sentences using only the FAQ context above:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 90,
            "top_p": 0.9,
            "stop": ["\nQ:", "\nQuestion:", "\nFAQ:", "Student Question:"],
        },
    )
    return response["message"]["content"].strip()


def ask(question: str) -> dict:
    """Answer an admissions question from local FAQ data."""
    question = _clean_text(question)
    if not question:
        return {"answer": "Please type a NUST admissions question.", "sources": [], "response_time_ms": 0}

    if _normalize(question) in {"hi", "hello", "hey", "salam", "assalam o alaikum"}:
        return {
            "answer": "Hello. Ask me about NUST admissions, NET, eligibility, scholarships, or related UG admission queries.",
            "sources": [],
            "response_time_ms": 0,
        }

    matched_faq, match_score = _direct_match(question)
    if matched_faq:
        return {
            "answer": matched_faq["answer"],
            "sources": [_faq_to_source(matched_faq)],
            "match": "direct",
            "score": round(match_score, 3),
        }

    best_distance, indices, retrieved_chunks = _semantic_search(question)
    if best_distance > OFF_TOPIC_THRESHOLD:
        return {
            "answer": "I can only help with NUST admissions queries. Contact ugadmissions@nust.edu.pk",
            "sources": [],
            "match": "off_topic",
            "distance": round(best_distance, 3),
        }

    if best_distance <= SEMANTIC_DIRECT_THRESHOLD or not USE_OLLAMA_FALLBACK:
        return {
            "answer": _extract_answer(retrieved_chunks[0]),
            "sources": retrieved_chunks,
            "match": "semantic",
            "distance": round(best_distance, 3),
        }

    try:
        answer = _ollama_answer(question, retrieved_chunks)
        if not answer:
            answer = _extract_answer(retrieved_chunks[0])
    except Exception:
        answer = _extract_answer(retrieved_chunks[0])

    return {
        "answer": answer,
        "sources": retrieved_chunks,
        "match": "ollama_fallback",
        "distance": round(best_distance, 3),
    }


if __name__ == "__main__":
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        result = ask(q)
        print(f"Answer: {result['answer']}")
        if result.get("sources"):
            print(f"\nSources used: {len(result['sources'])}")
