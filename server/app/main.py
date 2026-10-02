from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncIterator, Literal

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import hnswlib
except ImportError:
    hnswlib = None
try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None

DATA_DIR = Path(os.getenv("VECTOR_ATLAS_DATA", Path(__file__).parents[1] / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_FILE = DATA_DIR / "vectors.json"
DOCUMENT_FILE = DATA_DIR / "documents.json"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

Metric = Literal["cosine", "euclidean", "manhattan"]
Algorithm = Literal["brute_force", "kd_tree", "hnsw"]

class VectorIn(BaseModel):
    id: str | None = None
    label: str = Field(min_length=1, max_length=80)
    values: list[float] = Field(min_length=2, max_length=2048)
    group: str = Field(default="Custom", max_length=40)

class SearchIn(BaseModel):
    query: list[float] = Field(min_length=2, max_length=2048)
    k: int = Field(default=5, ge=1, le=50)
    metric: Metric = "cosine"
    algorithms: list[Algorithm] = ["brute_force", "kd_tree", "hnsw"]

class BenchmarkIn(BaseModel):
    queries: int = Field(default=50, ge=5, le=1000)
    k: int = Field(default=5, ge=1, le=50)
    metric: Metric = "cosine"

class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    text: str = Field(min_length=1, max_length=300_000)
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = Field(default=700, ge=150, le=2000)
    overlap: int = Field(default=100, ge=0, le=500)

class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2:1b"
    k: int = Field(default=4, ge=1, le=10)

def read_json(path: Path, default):
    if not path.exists(): return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def seed_vectors():
    rng = np.random.default_rng(42)
    centers = [(1.7, 2.5, "Indigo"), (-2.2, 1.3, "Sage"), (0.4, -2.3, "Ochre")]
    rows = []
    for cx, cy, group in centers:
        for i in range(12):
            x, y = rng.normal((cx, cy), 0.48)
            z = rng.normal(0.35 if group == "Indigo" else (-0.5 if group == "Sage" else 0.1), 0.42)
            rows.append({"id": f"{group.lower()}-{i+1}", "label": f"{group} {i+1}", "values": [round(float(x), 4), round(float(y), 4), round(float(z), 4)], "group": group})
    return rows

vectors: list[dict] = read_json(VECTOR_FILE, seed_vectors())
# Keep collections created by the earlier 2D demo usable in the 3D explorer.
if vectors and len(vectors[0]["values"]) == 2:
    for vector in vectors:
        vector["values"].append(0.0)
write_json(VECTOR_FILE, vectors)
documents: list[dict] = read_json(DOCUMENT_FILE, [])
document_hnsw = None
document_lookup: list[tuple[dict, int]] = []

def invalidate_document_index():
    global document_hnsw, document_lookup
    document_hnsw, document_lookup = None, []

def search_document_index(query: np.ndarray, k: int):
    """Retrieve document chunks from a cached HNSW index, with an exact fallback."""
    global document_hnsw, document_lookup
    if document_hnsw is None:
        rows, lookup = [], []
        for doc in documents:
            for chunk_number, vector in enumerate(doc["embeddings"]):
                rows.append(vector); lookup.append((doc, chunk_number))
        if not rows: return []
        document_lookup = lookup
        embedded = np.asarray(rows, dtype=np.float32)
        if hnswlib is not None:
            document_hnsw = hnswlib.Index(space="cosine", dim=embedded.shape[1])
            document_hnsw.init_index(max_elements=len(embedded), ef_construction=200, M=16)
            document_hnsw.add_items(embedded, np.arange(len(embedded)))
            document_hnsw.set_ef(max(64, k))
        else:
            document_hnsw = embedded
    if hnswlib is not None:
        ids, distances = document_hnsw.knn_query(query, k=min(k, len(document_lookup)))
        return [(float(distance), *document_lookup[int(idx)]) for idx, distance in zip(ids[0], distances[0])]
    scores = distance(document_hnsw, query, "cosine")
    return [(float(scores[i]), *document_lookup[int(i)]) for i in np.argsort(scores)[:k]]

def matrix(): return np.asarray([v["values"] for v in vectors], dtype=np.float32)

def distance(rows: np.ndarray, query: np.ndarray, metric: Metric):
    if metric == "euclidean": return np.linalg.norm(rows - query, axis=1)
    if metric == "manhattan": return np.abs(rows - query).sum(axis=1)
    denom = np.linalg.norm(rows, axis=1) * max(np.linalg.norm(query), 1e-12)
    return 1 - (rows @ query) / np.maximum(denom, 1e-12)

def search_exact(query: np.ndarray, k: int, metric: Metric):
    rows = matrix(); scores = distance(rows, query, metric); idx = np.argsort(scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]

def search_index(query: np.ndarray, k: int, metric: Metric, algorithm: Algorithm):
    rows = matrix()
    if not len(rows): return [], "empty"
    note = "exact"
    if algorithm == "kd_tree" and KDTree is not None and metric == "euclidean":
        dist, idx = KDTree(rows).query(query, k=min(k, len(rows)))
        idx, dist = np.atleast_1d(idx), np.atleast_1d(dist)
        return [(int(i), float(d)) for i, d in zip(idx, dist)], "KD-tree"
    if algorithm == "hnsw" and hnswlib is not None and metric in ("cosine", "euclidean"):
        space = "cosine" if metric == "cosine" else "l2"
        index = hnswlib.Index(space=space, dim=rows.shape[1])
        index.init_index(max_elements=len(rows), ef_construction=100, M=16)
        index.add_items(rows, np.arange(len(rows))); index.set_ef(max(50, k))
        ids, distances = index.knn_query(query, k=min(k, len(rows)))
        return [(int(i), float(d)) for i, d in zip(ids[0], distances[0])], "HNSW"
    result = search_exact(query, k, metric)
    suffix = "fallback: exact" if algorithm != "brute_force" else "exact"
    return result, suffix

def hydrate(matches):
    return [{**vectors[i], "distance": round(d, 6)} for i, d in matches]

def chunk_text(text: str, size: int, overlap: int):
    text = " ".join(text.split())
    chunks, cursor = [], 0
    while cursor < len(text):
        end = min(len(text), cursor + size)
        if end < len(text):
            cut = text.rfind(". ", cursor, end)
            if cut > cursor + size // 2: end = cut + 1
        chunks.append(text[cursor:end].strip())
        if end == len(text): break
        cursor = end - overlap
    return [c for c in chunks if c]

async def embed(inputs: list[str], model: str):
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{OLLAMA_URL}/api/embed", json={"model": model, "input": inputs})
            response.raise_for_status()
            return response.json()["embeddings"]
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:300]
        raise HTTPException(503, f"Ollama could not load embedding model '{model}'. {detail or error}")
    except (httpx.HTTPError, KeyError) as error:
        raise HTTPException(503, f"Ollama embedding request failed. Start Ollama and pull '{model}'. Details: {error}")

app = FastAPI(title="Vector Atlas", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/v1/health")
def health(): return {"status": "ok", "ollama_url": OLLAMA_URL}

@app.get("/api/v1/status")
async def model_status():
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
        models = response.json().get("models", [])
        return {"ollama": "online", "models": [{"name": m["name"], "capabilities": m.get("capabilities", [])} for m in models], "embedding_default": "nomic-embed-text", "chat_default": "llama3.2:1b"}
    except httpx.HTTPError as error:
        return {"ollama": "offline", "detail": str(error), "models": []}

@app.get("/api/v1/vectors")
def get_vectors(): return {"vectors": vectors, "dimensions": len(vectors[0]["values"]) if vectors else 0}

@app.post("/api/v1/vectors", status_code=201)
def insert_vector(item: VectorIn):
    global vectors
    if vectors and len(item.values) != len(vectors[0]["values"]): raise HTTPException(422, "Vector dimension must match the collection")
    key = item.id or f"custom-{uuid.uuid4().hex[:8]}"
    if any(v["id"] == key for v in vectors): raise HTTPException(409, "Vector id already exists")
    row = {"id": key, "label": item.label, "values": item.values, "group": item.group}
    vectors.append(row); write_json(VECTOR_FILE, vectors); return row

@app.delete("/api/v1/vectors/{vector_id}")
def delete_vector(vector_id: str):
    global vectors
    before = len(vectors); vectors = [v for v in vectors if v["id"] != vector_id]
    if len(vectors) == before: raise HTTPException(404, "Vector not found")
    write_json(VECTOR_FILE, vectors); return {"deleted": vector_id}

@app.post("/api/v1/vectors/search")
def vector_search(body: SearchIn):
    if vectors and len(body.query) != len(vectors[0]["values"]): raise HTTPException(422, "Query dimension must match the collection")
    query = np.asarray(body.query, dtype=np.float32); response = {}
    for algorithm in body.algorithms:
        started = time.perf_counter(); matches, implementation = search_index(query, body.k, body.metric, algorithm)
        response[algorithm] = {"implementation": implementation, "elapsed_ms": round((time.perf_counter()-started)*1000, 3), "results": hydrate(matches)}
    return {"query": body.query, "metric": body.metric, "searches": response}

@app.post("/api/v1/vectors/benchmark")
def benchmark(body: BenchmarkIn):
    if not vectors: raise HTTPException(400, "Collection is empty")
    rows = matrix(); rng = np.random.default_rng(9); queries = rows[rng.integers(0, len(rows), body.queries)] + rng.normal(0, .2, (body.queries, rows.shape[1]))
    output = []
    exact_sets = [set(i for i, _ in search_exact(q, body.k, body.metric)) for q in queries]
    for algorithm in ("brute_force", "kd_tree", "hnsw"):
        started = time.perf_counter(); hits = [] ; implementation = ""
        for q, truth in zip(queries, exact_sets):
            result, implementation = search_index(q, body.k, body.metric, algorithm)
            hits.append(len({i for i, _ in result} & truth) / body.k)
        total = (time.perf_counter()-started)*1000
        output.append({"algorithm": algorithm, "implementation": implementation, "total_ms": round(total, 2), "mean_ms": round(total/body.queries, 3), "recall_at_k": round(float(np.mean(hits)), 3)})
    return {"queries": body.queries, "metric": body.metric, "k": body.k, "results": output}

@app.get("/api/v1/stats")
def stats():
    dims = len(vectors[0]["values"]) if vectors else 0
    return {"vectors": len(vectors), "dimensions": dims, "groups": dict((g, sum(v["group"] == g for v in vectors)) for g in sorted({v["group"] for v in vectors})), "documents": len(documents), "chunks": sum(len(d["chunks"]) for d in documents), "engines": {"brute_force": "NumPy exact", "kd_tree": "SciPy" if KDTree else "exact fallback", "hnsw": "hnswlib" if hnswlib else "exact fallback", "document_retrieval": "cached HNSW" if hnswlib else "NumPy exact fallback"}}

@app.get("/api/v1/documents")
def list_documents(): return {"documents": [{k: v for k, v in doc.items() if k not in ("chunks", "embeddings")} | {"chunk_count": len(doc["chunks"])} for doc in documents]}

@app.post("/api/v1/documents", status_code=201)
async def insert_document(body: DocumentIn):
    chunks = chunk_text(body.text, body.chunk_size, body.overlap)
    embeddings = await embed(chunks, body.embedding_model)
    doc = {"id": uuid.uuid4().hex[:12], "title": body.title, "created_at": int(time.time()), "embedding_model": body.embedding_model, "chunks": chunks, "embeddings": embeddings}
    documents.append(doc); invalidate_document_index(); write_json(DOCUMENT_FILE, documents)
    return {"id": doc["id"], "title": doc["title"], "chunk_count": len(chunks)}

@app.delete("/api/v1/documents/{document_id}")
def delete_document(document_id: str):
    global documents
    before = len(documents); documents = [d for d in documents if d["id"] != document_id]
    if len(documents) == before: raise HTTPException(404, "Document not found")
    invalidate_document_index(); write_json(DOCUMENT_FILE, documents); return {"deleted": document_id}

@app.post("/api/v1/documents/ask")
async def ask(body: AskIn):
    if not documents: raise HTTPException(400, "Add a document before asking a question")
    query = np.asarray((await embed([body.question], body.embedding_model))[0], dtype=np.float32)
    top = search_document_index(query, body.k)
    contexts = [{"document_id": d["id"], "title": d["title"], "chunk": i + 1, "distance": round(s, 4), "text": d["chunks"][i]} for s, d, i in top]
    prompt = "Answer only from the supplied local sources. If the sources do not establish an answer, say so plainly. Cite sources as [Title, chunk N].\\n\\n" + "\\n\\n".join(f"[{c['title']}, chunk {c['chunk']}]\\n{c['text']}" for c in contexts) + f"\\n\\nQuestion: {body.question}\\nAnswer:"
    async def stream() -> AsyncIterator[str]:
        yield json.dumps({"type": "context", "items": contexts}) + "\n"
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={"model": body.chat_model, "prompt": prompt, "stream": True, "options": {"temperature": 0.2}}) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            packet = json.loads(line)
                            if packet.get("response"): yield json.dumps({"type": "token", "text": packet["response"]}) + "\n"
                            if packet.get("done"): yield json.dumps({"type": "done"}) + "\n"
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:300]
            yield json.dumps({"type": "error", "message": f"Ollama could not generate with '{body.chat_model}'. {detail or error}"}) + "\n"
        except httpx.HTTPError as error:
            yield json.dumps({"type": "error", "message": f"Ollama generation failed: {error}"}) + "\n"
    return StreamingResponse(stream(), media_type="application/x-ndjson")

# When the React client is built, one Python process can serve both the API and UI.
CLIENT_DIST = Path(__file__).parents[2] / "client" / "dist"
if CLIENT_DIST.exists():
    app.mount("/", StaticFiles(directory=CLIENT_DIST, html=True), name="client")
