# Vector Atlas

Local vector search and retrieval lab: a FastAPI service with a React browser interface. It is designed to make vector indexes tangible: search the same sample collection with brute force, KD-tree, and HNSW, then inspect the points and result sets side by side.

## What is included

- Interactive 2D scatter plot with seeded, color-coded vector clusters
- Click a plotted vector to use it as the query and compare neighboring results side by side
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
ollama pull llama3.2:1b
```

Or use the one-command PowerShell helper from the project root: `./run-local.ps1`. Add `-Production` to build React and serve the entire app from `http://127.0.0.1:8000` as a single Python deployment.

`nomic-embed-text` uses Ollama's current `/api/embed` API; change model names in the Documents panel if you use another local model. HNSW and KD-tree gracefully fall back to exact search if their optional acceleration library is unavailable.

## Deploy to Render for free

The repository includes a Render Blueprint and Dockerfile for deploying the React UI and FastAPI API together. The Blueprint uses Render's free web-service tier and does not provision a persistent disk, so this is suitable for a public demo rather than durable storage.

1. Push this repository to GitHub.
2. In Render, choose **New > Blueprint**, connect the repository, and deploy the `render.yaml` blueprint. It selects the **Free** plan.
3. Open the service URL after the build and health check complete.

Free web services spin down after 15 minutes without traffic and can take about a minute to start again. Their filesystem is ephemeral: uploaded documents and vector changes can be lost when the service sleeps, restarts, or redeploys. Do not store important or sensitive data.

The app has no user authentication. Anyone who can access the public service can modify its vectors and documents, so keep it demo-only unless you add access control.

The Docker build creates the client bundle and the server serves it alongside the API. It builds hnswlib without CPU-specific compiler optimizations so the extension remains compatible with Render's runtime hardware. The deployment starts with the demo vectors; changes are stored only in the service's temporary filesystem.

### Optional: connect Ollama for document retrieval

Document embedding and RAG require an Ollama server reachable from Render. To use Ollama on your Windows PC, follow the protected local tunnel setup below, then add `OLLAMA_URL`, `OLLAMA_BASIC_AUTH_USERNAME`, and `OLLAMA_BASIC_AUTH_PASSWORD` in the Render service's **Environment** settings and redeploy. These settings are optional; the vector explorer works without them. Quick Tunnels are temporary, so the Ollama features stop working if the tunnel is unavailable.

### Connect Ollama running on your Windows PC

Ollama itself is free, but it runs on your PC in this setup. Document embedding and RAG are available only while your PC is on, Ollama is running, and the tunnel is connected. The included Caddy configuration protects the Ollama endpoint with Basic Auth; do not expose Ollama's port 11434 directly to the internet.

1. Install and start [Ollama for Windows](https://ollama.com/download/windows), [Caddy](https://caddyserver.com/download), and [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/). In a PowerShell window, download the two models:

   ```powershell
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
   ```

2. In a PowerShell window at the repository root, choose a username and a strong, unique password. Generate a Caddy password hash and start the protected local proxy:

   ```powershell
   $env:OLLAMA_TUNNEL_USER = "vector-atlas"
   $env:OLLAMA_BASIC_AUTH_PASSWORD = Read-Host "Choose a long, unique password"
   $env:OLLAMA_TUNNEL_PASSWORD_HASH = caddy hash-password --plaintext $env:OLLAMA_BASIC_AUTH_PASSWORD
   caddy run --config .\ollama-tunnel.Caddyfile --adapter caddyfile
   ```

   Keep this window open. The password (not the Caddy hash) is needed for Render; do not commit or share it.

3. In a second PowerShell window, start a temporary Cloudflare Quick Tunnel to the protected proxy:

   ```powershell
   cloudflared tunnel --url http://127.0.0.1:11435
   ```

   Keep this window open too. Copy the `https://....trycloudflare.com` URL it prints.

4. In Render, open the `vector-atlas` service's **Environment** settings and set `OLLAMA_URL` to that HTTPS URL, `OLLAMA_BASIC_AUTH_USERNAME` to the username from step 2, and `OLLAMA_BASIC_AUTH_PASSWORD` to the password from step 2. Save changes and let Render redeploy.
5. In Vector Atlas, the Documents workspace should now show the Ollama models. If the Quick Tunnel is stopped or restarted, repeat step 3 and update `OLLAMA_URL` if its URL changed.

Cloudflare Quick Tunnels are temporary and intended for development/testing, not a guaranteed always-on production connection. This application itself does not require user authentication, so anyone who can reach the public service can modify its vectors and documents. Keep it demo-only unless you add access control, and do not upload sensitive documents.

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

## Using the vector explorer

The demo collection contains 36 vectors arranged in three visible clusters. Choose cosine, Euclidean, or Manhattan distance, adjust the query coordinates and neighbor count, then run all three search strategies against the same point. Hover over a plotted vector to inspect it or select it to run a search at its coordinates. If a custom collection has more than two dimensions, the scatter plot displays X/Y while the additional dimensions remain editable and participate in search.
