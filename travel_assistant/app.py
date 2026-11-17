"""Local study workspace for text-based textbooks."""
from collections import OrderedDict
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4
from typing import Literal
from .answers import DEFAULT_ANSWER_TEMPERATURE, MAX_ANSWER_TOKENS, answer_question
from .retrieval import MODEL, embed, hybrid_scores
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

app = FastAPI(title="Textbook RAG")
assets = Path(__file__).parent
HOSTED = os.getenv("VERCEL") == "1"
storage = TemporaryDirectory(prefix="textbook-rag-")
books = OrderedDict()
lock = Lock()
MAX_BYTES = 500 * 1024 * 1024
MAX_PAGES = 5000
MAX_CHUNKS = 100_000


def split_page(text, size, overlap, strategy, separators):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if strategy != "fixed" and end < len(text):
            for separator in separators:
                boundary = text.rfind(separator, start + max(size // 2, overlap), end)
                if boundary >= 0:
                    end = boundary + len(separator)
                    break
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break
        start = end - overlap


class Book:
    def __init__(self, path, name, size, overlap, strategy, separators, hybrid=False):
        self.path = path
        self.name = name
        self.chunks = []
        self.parents = {}
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("This PDF is encrypted.")
        self.pages = len(reader.pages)
        if self.pages > MAX_PAGES:
            raise ValueError(f"Use a PDF with at most {MAX_PAGES:,} pages.")
        self.text_pages = 0
        self.page_text = {}
        for page_index, page in enumerate(reader.pages, 1):
            text = page.extract_text().strip()
            if not text:
                continue
            self.text_pages += 1
            self.page_text[page_index] = text
            for start, end in split_page(text, size, overlap, strategy, separators):
                parent_id = None
                if strategy == "parent-child":
                    block = start // (size * 3)
                    parent_id = (page_index, block)
                    if parent_id not in self.parents:
                        self.parents[parent_id] = text[block * size * 3:(block + 1) * size * 3 + size].strip()
                self.chunks.append({"id": len(self.chunks) + 1, "page": page_index,
                                    "text": text[start:end], "start": start, "end": end,
                                    "parent": parent_id})
                if len(self.chunks) > MAX_CHUNKS:
                    raise ValueError("Too many passages. Increase the chunk size.")
        if not self.chunks:
            raise ValueError("No selectable text found. Run OCR on scanned pages and upload again.")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                          max_features=60_000, dtype=np.float32)
        try:
            self.matrix = self.vectorizer.fit_transform(chunk["text"] for chunk in self.chunks)
        except ValueError as error:
            raise ValueError("The PDF has no searchable words.") from error
        dimensions = min(64, min(self.matrix.shape) - 1)
        self.model = TruncatedSVD(n_components=dimensions, random_state=42) if dimensions >= 2 else None
        self.embeddings = embed([chunk["text"] for chunk in self.chunks]) if hybrid else None
        self.vectors = normalize(self.model.fit_transform(self.matrix)) if self.model else None

    def preview(self, index):
        chunk = self.chunks[index]
        if self.embeddings is not None:
            vector = self.embeddings[index, :8]
        else:
            vector = self.vectors[index][:8] if self.model else self.matrix[index, :8].toarray()[0]
        point = vector[:2].tolist()
        return {**chunk, "vector": vector.tolist(), "x": point[0], "y": point[1] if len(point) > 1 else 0}

    def search(self, query, limit):
        vector = self.vectorizer.transform([query])
        lexical = (self.matrix @ vector.T).toarray().ravel()
        scores = lexical
        eligible = lexical > 0
        query_vector = vector[0, :8].toarray()[0]
        dimensions = vector.shape[1]
        if self.model:
            dense = normalize(self.model.transform(vector))[0]
            query_vector = dense[:8]
            dimensions = len(dense)
            semantic = self.vectors @ dense
            scores = .65 * lexical + .35 * semantic
            if vector.nnz:
                eligible |= semantic >= .55
        if self.embeddings is not None:
            try:
                dense = embed([query])[0]
            except (OSError, ValueError, KeyError) as error:
                raise HTTPException(503, "Local embeddings are unavailable. Check Ollama and retry.") from error
            scores = hybrid_scores(lexical, self.embeddings @ dense)
            eligible = scores > 0
            query_vector = dense[:8]
            dimensions = len(dense)
        ordered = np.argsort(-scores, kind="stable")
        matches = []
        seen = set()
        for index in ordered:
            if not eligible[index]:
                continue
            chunk = self.preview(index)
            excerpt = self.parents.get(chunk["parent"], chunk["text"])
            key = (chunk["page"], excerpt)
            if key in seen:
                continue
            seen.add(key)
            matches.append({**chunk, "excerpt": excerpt, "score": float(scores[index])})
            if len(matches) == limit:
                break
        query_point = query_vector[:2]
        point = np.pad(query_point, (0, max(0, 2 - len(query_point))))
        return {"matches": matches, "point": point.tolist(), "searched": len(self.chunks), "query_vector": query_vector.tolist(), "dimensions": dimensions,
                "score_type": "Fusion score" if self.embeddings is not None else "Similarity"}


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    limit: int = Field(default=5, ge=1, le=12)


def get_book(book_id):
    with lock:
        if book_id not in books:
            raise HTTPException(404, "This index expired. Upload the PDF again.")
        books.move_to_end(book_id)
        return books[book_id]


@app.get("/")
@app.get("/experiment")
def home():
    return FileResponse(assets / "index.html")


@app.get("/dashboard.js")
def javascript():
    return FileResponse(assets / "dashboard.js", media_type="text/javascript")


@app.get("/dashboard.css")
def stylesheet():
    return FileResponse(assets / "dashboard.css", media_type="text/css")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/capabilities")
def capabilities():
    return {"hosted": HOSTED}


@app.post("/books")
def upload(file: UploadFile = File(...), size: int = Form(700, ge=150, le=3500),
           overlap: int = Form(80, ge=0, le=500), strategy: str = Form("recursive"),
           hybrid: bool = Form(False), separators: str = Form(r"\n\n|\n| ")):
    if HOSTED and hybrid:
        raise HTTPException(422, "Hybrid search requires the local Ollama installation.")
    if overlap >= size or strategy not in {"recursive", "fixed", "parent-child"}:
        raise HTTPException(422, "Choose an overlap smaller than the chunk size.")
    boundaries = [separator.replace(r"\n", "\n") for separator in separators[:200].split("|") if separator]
    book_id = uuid4().hex
    path = Path(storage.name) / f"{book_id}.pdf"
    try:
        total = 0
        with path.open("wb") as output:
            for block in iter(lambda: file.file.read(1024 * 1024), b""):
                if total == 0 and b"%PDF-" not in block[:1024]:
                    raise ValueError("Upload a valid PDF document.")
                total += len(block)
                if total > (4_000_000 if HOSTED else MAX_BYTES):
                    raise ValueError("The PDF exceeds the upload limit.")
                output.write(block)
        book = Book(path, Path(file.filename or "Textbook.pdf").name, size, overlap, strategy, boundaries, hybrid)
    except (OSError, KeyError) as error:
        path.unlink(missing_ok=True)
        if hybrid:
            raise HTTPException(503, "Indexing failed. Check Ollama and install all-minilm, or turn hybrid search off.") from error
        raise HTTPException(422, "The PDF could not be read.") from error
    except Exception as error:
        path.unlink(missing_ok=True)
        message = str(error) if isinstance(error, ValueError) else "The PDF could not be read."
        raise HTTPException(422, message) from error

    with lock:
        books[book_id] = book
        if len(books) > 8:
            _, evicted = books.popitem(last=False)
            evicted.path.unlink(missing_ok=True)
    return {"id": book_id, "name": book.name, "pages": book.pages, "textPages": book.text_pages,
            "chunkCount": len(book.chunks), "model": MODEL + " + TF-IDF" if hybrid else ("TF-IDF + latent semantic analysis" if book.model else "TF-IDF"),
            "dimensions": book.embeddings.shape[1] if hybrid else (book.vectors.shape[1] if book.model else book.matrix.shape[1]),
            "preview": [book.preview(index) for index in np.linspace(
                0, len(book.chunks) - 1, min(120, len(book.chunks)), dtype=int)]}


@app.get("/books/{book_id}/pdf")
def pdf(book_id: str):
    return FileResponse(get_book(book_id).path, media_type="application/pdf")


@app.post("/books/{book_id}/search")
def search(book_id: str, request: Question):
    return get_book(book_id).search(request.question, request.limit)


class AnswerQuestion(Question):
    mode: Literal["extractive", "generated"] = "extractive"
    temperature: float = Field(default=DEFAULT_ANSWER_TEMPERATURE, ge=0, le=1)
    max_tokens: int = Field(default=MAX_ANSWER_TOKENS, ge=300, le=MAX_ANSWER_TOKENS)


@app.post("/books/{book_id}/answer")
def answer(book_id: str, request: AnswerQuestion):
    if HOSTED and request.mode == "generated":
        raise HTTPException(422, "Generated explanations require the local Nemotron installation.")
    book = get_book(book_id)
    limit = max(8, request.limit) if request.mode == "generated" else request.limit
    matches = book.search(request.question, limit)["matches"]
    if request.mode == "generated":
        for match in matches:
            page = book.page_text[match["page"]]
            start = max(0, match["start"] - 500)
            end = min(len(page), match["end"] + 650)
            if start:
                boundary = page.find(" ", start, match["start"])
                if boundary >= 0:
                    start = boundary + 1
            if end < len(page):
                boundary = page.rfind(" ", match["end"], end)
                if boundary >= 0:
                    end = boundary
            match["excerpt"] = page[start:end]
    try:
        return answer_question(request.question, matches, request.mode, request.temperature, request.max_tokens)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (OSError, KeyError, TypeError) as error:
        raise HTTPException(502, "The local answer model could not complete this request.") from error
