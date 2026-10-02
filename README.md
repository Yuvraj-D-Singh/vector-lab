# Vector Atlas

Local vector search and retrieval lab: a FastAPI service with a React browser interface. It is designed to make vector indexes tangible: search the same sample collection with brute force, KD-tree, and HNSW, then inspect the points and result sets side by side.

## What is included

- Interactive 2D vector explorer with seeded clustered data
- Orbitable 3D vector explorer: drag to orbit, use the wheel to zoom, right-drag to pan, and reset the camera at any time
- Cosine, Euclidean, and Manhattan metrics
- Brute-force exact search, KD-tree (SciPy when installed), and HNSW (hnswlib when installed)
- Timed benchmark endpoint and index statistics
- Local document chunking, Ollama embeddings, vector retrieval, and streamed RAG answers
- JSON-on-disk persistence with no cloud services or database required

## Quick start on Windows

Prerequisites: Python 3.11+, Node 20+, and [Ollama](https://ollama.com/download/windows) only for the Documents workspace.

```powershell
cd server
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

In a second PowerShell window:

```powershell
cd client
npm install
npm run dev
```

Open the URL printed by Vite (normally `http://localhost:5173`). The vector explorer works immediately. For document retrieval, pull the recommended local models:

```powershell
ollama pull nomic-embed-text
ollama pull mistral:latest
```

Or use the one-command PowerShell helper from the project root: `./run-local.ps1`. Add `-Production` to build React and serve the entire app from `http://127.0.0.1:8000` as a single Python deployment.

`nomic-embed-text` uses Ollama's current `/api/embed` API; change model names in the Documents panel if you use another local model. HNSW and KD-tree gracefully fall back to exact search if their optional acceleration library is unavailable.

## REST API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/vectors` | list demo vectors |
| `POST` | `/api/v1/vectors/search` | search one or all indexes |
| `POST` | `/api/v1/vectors` | insert a vector |
| `DELETE` | `/api/v1/vectors/{id}` | delete a vector |
| `POST` | `/api/v1/vectors/benchmark` | compare runtime/recall |
| `GET` | `/api/v1/stats` | collection and index statistics |
| `POST` | `/api/v1/documents` | embed and store text chunks |
| `GET` | `/api/v1/documents` | list stored documents |
| `DELETE` | `/api/v1/documents/{id}` | delete a document and chunks |
| `POST` | `/api/v1/documents/ask` | stream a local RAG response as NDJSON |

## Architecture

```
React + Vite  ── JSON / streaming NDJSON ──>  FastAPI + NumPy
                                               ├─ Brute force / KD-tree / HNSW
                                               ├─ local JSON persistence
                                               └─ Ollama `/api/embed` + `/api/generate`
```

The service never uploads content; Ollama is contacted only at its configurable local address (default `http://127.0.0.1:11434`).

## Using the 3D explorer

The seeded collection now has X, Y, and Z coordinates. In the browser, open **Vector explorer**, then drag inside the plot to orbit the point cloud, use the mouse wheel to zoom, and right-drag to pan. Edit X, Y, and Z under **Search point** and choose **Run comparison** to send the same 3D query through every index.
