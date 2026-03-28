"""Build FAISS index from FAQ data for semantic search."""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("data/faqs.json")
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "faqs.index"
TEXTS_FILE = INDEX_DIR / "faqs_texts.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# STEP 1: Load FAQ data
if not DATA_FILE.exists():
    print("ERROR: data/faqs.json not found!")
    exit(1)
    
with open(DATA_FILE, "r", encoding="utf-8") as f:
    faqs = json.load(f)
    
if not faqs:
    print("ERROR: data/faqs.json is empty!")
    exit(1)


# STEP 2: Build text chunks (Q + A combined)
chunks = []
for item in faqs:
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()
    if not question or not answer:
        continue
    chunk = f"Q: {question}\nA: {answer}"
    chunks.append(chunk)


# STEP 3: Load embedding model
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit(1)


# STEP 4: Embed all chunks
try:
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
except Exception as e:
    print(f"ERROR during embedding: {e}")
    exit(1)


# STEP 5: Build FAISS index
try:
    dimension = embeddings.shape[1]
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
except Exception as e:
    print(f"ERROR building FAISS index: {e}")

    exit(1)


# STEP 6: Save outputs
INDEX_DIR.mkdir(exist_ok=True)

try:
    faiss.write_index(index, str(INDEX_FILE))
except Exception as e:
    print(f"ERROR saving FAISS index: {e}")
    exit(1)

try:
    with open(TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
except Exception as e:
    print(f"ERROR saving texts file: {e}")
    exit(1)
