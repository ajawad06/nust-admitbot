"""RAG query engine — embeds question, retrieves FAQ chunks, generates answer."""
import json
import sys
import numpy as np
from pathlib import Path


INDEX_PATH = Path("index/faqs.index")
TEXTS_PATH = Path("index/faqs_texts.json")


# STEP 1: Check required files exist
if not INDEX_PATH.exists() or not TEXTS_PATH.exists():
    print("ERROR: Index files not found.")
    print("  Missing: index/faqs.index or index/faqs_texts.json")
    sys.exit(1)


# STEP 2: Load embedding model
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"ERROR: Could not load embedding model: {e}")
    sys.exit(1)


# STEP 3: Load FAISS index
try:
    import faiss
    faiss_index = faiss.read_index(str(INDEX_PATH))
except Exception as e:
    print(f"ERROR: Could not load FAISS index: {e}")
    sys.exit(1)


# STEP 4: Load FAQ text chunks
try:
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        faq_texts = json.load(f)
except Exception as e:
    print(f"ERROR: Could not load FAQ texts: {e}")
    sys.exit(1)


SYSTEM_PROMPT = """You are a helpful NUST admissions assistant.
Answer the student's question using ONLY the FAQ context provided below.
If the answer is not in the context, say:
"I don't have that information in my FAQ database. Please contact NUST admissions directly at admission@nust.edu.pk"
Be concise, friendly, and accurate. Do not make up information.
Do not mention that you are using a context or FAQ list — just answer naturally."""


def ask(question):
    """Embed question, retrieve relevant FAQs, and generate answer.
    
    Args:
        question: Student question string
        
    Returns:
        dict: {"answer": str, "sources": list of FAQ chunks}
    """
    # Step A: Embed the question
    q_vector = embedding_model.encode([question])
    q_vector = np.array(q_vector, dtype="float32")

    # Step B: Retrieve top 3 similar FAQ chunks
    distances, indices = faiss_index.search(q_vector, k=3)
    retrieved_chunks = [faq_texts[i] for i in indices[0] if i < len(faq_texts)]

    # Step C: Build prompt with retrieved context
    context = "\n\n---\n\n".join(retrieved_chunks)
    user_message = f"""FAQ Context:
{context}
Student Question: {question}"""
    
    # Step D: Get response from Ollama
    try:
        import ollama
        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        answer = response["message"]["content"]
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "refused" in error_str or "socket" in error_str:
            answer = (
                "ERROR: Ollama is not running.\n"
                "Open a new terminal and run:  ollama serve\n"
                "Then try your question again."
            )
        else:
            answer = f"ERROR: Could not get answer from Ollama.\nDetails: {e}"
    
    # Step E: Return answer with sources
    return {
        "answer": answer,
        "sources": retrieved_chunks
    }


if __name__ == "__main__":
    # Terminal test mode
    while True:
        # Get user input
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Skip empty input
        if not question:
            continue

        # Handle exit commands
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Get answer from chatbot
        result = ask(question)
        print(f"Answer:\n{result['answer']}")
        print()

        # Show referenced sources
        for i, chunk in enumerate(result["sources"], 1):
            first_line = chunk.split("\n")[0]
            print(f"  [{i}] {first_line}")
        print()