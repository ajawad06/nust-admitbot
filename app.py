"""FastAPI backend for the local NUST admissions chatbot."""

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatbot import ask, faqs

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="NUST Admissions Chatbot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=800)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    response_time_ms: int
    match: str | None = None


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "faq_count": len(faqs), "mode": "offline"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> dict:
    started = time.perf_counter()
    result = ask(payload.question)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "response_time_ms": elapsed_ms,
        "match": result.get("match"),
    }
