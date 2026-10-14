import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const api = async (path: string, init?: RequestInit) => {
  const r = await fetch(`/api/v1${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok)
    throw new Error(
      (await r.json().catch(() => ({ detail: r.statusText }))).detail,
    );
  return r.json();
};
type Vector = {
  id: string;
  label: string;
  values: number[];
  group: string;
  distance?: number;
};
const colors: Record<string, string> = {
  Indigo: "#5661ab",
  Sage: "#5e8371",
  Ochre: "#b47c36",
  Custom: "#ae5e71",
};

function VectorScatterPlot({
  vectors,
  query,
  onChooseQuery,
}: {
  vectors: Vector[];
  query: number[];
  onChooseQuery: (point: Vector) => void;
}) {
  const groups = Array.from(new Set(vectors.map((vector) => vector.group)));
  return (
    <div className="scatter-plot" aria-label="2D scatter plot of vector clusters">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 24, bottom: 14, left: 8 }}>
          <CartesianGrid stroke="#e1e2db" strokeDasharray="3 4" />
          <XAxis
            type="number"
            dataKey="x"
            name="X"
            tick={{ fill: "#777b73", fontSize: 11 }}
            axisLine={{ stroke: "#bfc2b8" }}
            tickLine={false}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Y"
            tick={{ fill: "#777b73", fontSize: 11 }}
            axisLine={{ stroke: "#bfc2b8" }}
            tickLine={false}
          />
          <Tooltip
            cursor={{ stroke: "#a5a99f", strokeDasharray: "4 4" }}
            content={({ active, payload }) => {
              const point = payload?.[0]?.payload;
              if (!active || !point) return null;
              const vector = point.vector as Vector | undefined;
              return (
                <div className="chart-tooltip">
                  <strong>{vector?.label || "Search point"}</strong>
                  {vector && <span>{vector.group} · click to search</span>}
                  <code>
                    x {point.x.toFixed(2)} · y {point.y.toFixed(2)}
                  </code>
                </div>
              );
            }}
          />
          <Legend verticalAlign="top" align="right" height={34} />
          {groups.map((group) => (
            <Scatter
              key={group}
              name={group}
              data={vectors
                .filter((vector) => vector.group === group)
                .map((vector) => ({
                  x: vector.values[0],
                  y: vector.values[1],
                  vector,
                }))}
              fill={colors[group] || colors.Custom}
              onClick={(entry) => {
                const point = (entry as any)?.payload?.vector;
                if (point) onChooseQuery(point);
              }}
              cursor="pointer"
            />
          ))}
          <Scatter
            name="Search point"
            data={[{ x: query[0] ?? 0, y: query[1] ?? 0 }]}
            shape="diamond"
            fill="#252821"
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function App() {
  const [tab, setTab] = useState<"explorer" | "documents">("explorer"),
    [vectors, setVectors] = useState<Vector[]>([]),
    [stats, setStats] = useState<any>(),
    [query, setQuery] = useState([0, 0]),
    [metric, setMetric] = useState("cosine"),
    [k, setK] = useState(5),
    [search, setSearch] = useState<any>(),
    [benchmark, setBenchmark] = useState<any>(),
    [error, setError] = useState("");
  const [title, setTitle] = useState(""),
    [text, setText] = useState(""),
    [docs, setDocs] = useState<any[]>([]),
    [embedding, setEmbedding] = useState(""),
    [chat, setChat] = useState(""),
    [modelStatus, setModelStatus] = useState<any>(),
    [modelSettingsInitialized, setModelSettingsInitialized] = useState(false),
    [question, setQuestion] = useState(""),
    [answer, setAnswer] = useState(""),
    [contexts, setContexts] = useState<any[]>([]),
    [busy, setBusy] = useState(false);
  const refresh = async () => {
    try {
      const [v, s, d, status] = await Promise.all([
        api("/vectors"),
        api("/stats"),
        api("/documents"),
        api("/status"),
      ]);
      setVectors(v.vectors);
      setQuery((current) => {
        const dimensions = Math.max(2, v.dimensions);
        return Array.from(
          { length: dimensions },
          (_, index) => current[index] ?? 0,
        );
      });
      setStats(s);
      setDocs(d.documents);
      setModelStatus(status);
      if (!modelSettingsInitialized) {
        setEmbedding(status.embedding_default);
        setChat(status.chat_default);
        setModelSettingsInitialized(true);
      }
      setError("");
    } catch (e: any) {
      setError(e.message);
    }
  };
  useEffect(() => {
    refresh();
  }, []);
  const runSearch = async (searchQuery = query) => {
    try {
      setSearch(
        await api("/vectors/search", {
          method: "POST",
          body: JSON.stringify({
            query: searchQuery,
            k,
            metric,
            algorithms: ["brute_force", "kd_tree", "hnsw"],
          }),
        }),
      );
    } catch (e: any) {
      setError(e.message);
    }
  };
  const runBenchmark = async () => {
    try {
      setBenchmark(
        await api("/vectors/benchmark", {
          method: "POST",
          body: JSON.stringify({ queries: 150, k, metric }),
        }),
      );
    } catch (e: any) {
      setError(e.message);
    }
  };
  const ingest = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api("/documents", {
        method: "POST",
        body: JSON.stringify({ title, text, embedding_model: embedding }),
      });
      setTitle("");
      setText("");
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const ask = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setAnswer("");
    setContexts([]);
    try {
      const r = await fetch("/api/v1/documents/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          embedding_model: embedding,
          chat_model: chat,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).detail);
      const reader = r.body!.getReader(),
        decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          pending += decoder.decode();
          if (pending) {
            const p = JSON.parse(pending);
            if (p.type === "context") setContexts(p.items);
            if (p.type === "token") setAnswer((a) => a + p.text);
            if (p.type === "error") setError(p.message);
          }
          break;
        }
        pending += decoder.decode(value, { stream: true });
        const lines = pending.split("\n");
        pending = lines.pop() || "";
        for (const line of lines)
          if (line) {
            const p = JSON.parse(line);
            if (p.type === "context") setContexts(p.items);
            if (p.type === "token") setAnswer((a) => a + p.text);
            if (p.type === "error") setError(p.message);
          }
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <main>
      <header>
        <div className="brand">
          <span className="brand-mark">VA</span>
          <div>
            <h1>Vector Atlas</h1>
            <p>Local search laboratory</p>
          </div>
        </div>
        <nav>
          <button
            className={tab === "explorer" ? "active" : ""}
            onClick={() => setTab("explorer")}
          >
            Vector explorer
          </button>
          <button
            className={tab === "documents" ? "active" : ""}
            onClick={() => setTab("documents")}
          >
            Document desk
          </button>
        </nav>
        <div className="status">
          <i /> Local service
        </div>
      </header>
      {error && (
        <div className="notice">
          {error}
          <button onClick={() => setError("")}>Dismiss</button>
        </div>
      )}
      {tab === "explorer" ? (
        <section className="shell">
          <div className="intro">
            <div>
              <span className="eyebrow">SEARCH SYSTEMS, MADE LEGIBLE</span>
              <h2>
                Find the nearest point.
                <br />
                See how each path gets there.
              </h2>
            </div>
            <div className="stats">
              {stats && (
                <>
                  <div>
                    <b>{stats.vectors}</b>
                    <span>vectors</span>
                  </div>
                  <div>
                    <b>{stats.dimensions}D</b>
                    <span>space</span>
                  </div>
                  <div>
                    <b>{stats.engines.brute_force}</b>
                    <span>Brute force</span>
                  </div>
                  <div>
                    <b>{stats.engines.kd_tree}</b>
                    <span>KD-tree</span>
                  </div>
                  <div>
                    <b>{stats.engines.hnsw}</b>
                    <span>HNSW</span>
                  </div>
                </>
              )}
            </div>
          </div>
          <div className="workspace">
            <aside className="control">
              <h3>Search point</h3>
              <p>
                Adjust coordinates and compare identical queries through three
                retrieval strategies.
              </p>
              {query.slice(0, 2).map((coordinate, index) => (
                <label key={index}>
                  {index === 0 ? "X" : "Y"} coordinate
                  <input
                    type="number"
                    step=".1"
                    value={coordinate}
                    onChange={(event) =>
                      setQuery((current) =>
                        current.map((value, coordinateIndex) =>
                          coordinateIndex === index
                            ? +event.target.value
                            : value,
                        ),
                      )
                    }
                  />
                </label>
              ))}
              {query.length > 2 && (
                <details className="additional-coordinates">
                  <summary>Additional dimensions ({query.length - 2})</summary>
                  {query.slice(2).map((coordinate, index) => (
                    <label key={index + 2}>
                      Dimension {index + 3}
                      <input
                        type="number"
                        step=".1"
                        value={coordinate}
                        onChange={(event) =>
                          setQuery((current) =>
                            current.map((value, coordinateIndex) =>
                              coordinateIndex === index + 2
                                ? +event.target.value
                                : value,
                            ),
                          )
                        }
                      />
                    </label>
                  ))}
                </details>
              )}
              <label>
                Distance
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                >
                  <option value="cosine">Cosine</option>
                  <option value="euclidean">Euclidean</option>
                  <option value="manhattan">Manhattan</option>
                </select>
              </label>
              <label>
                Neighbours <output>{k}</output>
                <input
                  type="range"
                  min="1"
                  max="12"
                  value={k}
                  onChange={(e) => setK(+e.target.value)}
                />
              </label>
              <button className="primary" onClick={() => runSearch()}>
                Run comparison
              </button>
              <button className="quiet" onClick={runBenchmark}>
                Benchmark 150 queries
              </button>
            </aside>
            <div className="chart-panel">
              <div className="panel-title">
                <div>
                  <span className="eyebrow">COLLECTION VIEW</span>
                  <h3>Cluster map</h3>
                </div>
                <span className="query-key">● query [{query.join(", ")}]</span>
              </div>
              <VectorScatterPlot
                vectors={vectors}
                query={query}
                onChooseQuery={(point) => {
                  setQuery(point.values);
                  runSearch(point.values);
                }}
              />
              <div className="map-actions">
                <span>Hover a point to inspect it · select a point to search</span>
              </div>
              <p className="caption">
                Color identifies each cluster. The dark diamond marks the query;
                higher dimensions remain part of the search when present.
              </p>
            </div>
          </div>
          {search && (
            <section className="result-section">
              <div className="section-head">
                <div>
                  <span className="eyebrow">
                    SAME QUERY · {metric.toUpperCase()}
                  </span>
                  <h2>Results by index</h2>
                </div>
                <span>Lower distance is closer</span>
              </div>
              <div className="result-grid">
                {Object.entries(search.searches).map(([name, run]: any) => (
                  <article className="result-card" key={name}>
                    <div className="result-top">
                      <h3>{name.replace("_", " ")}</h3>
                      <span>{run.elapsed_ms} ms</span>
                    </div>
                    <small>{run.implementation}</small>
                    {run.results.map((r: Vector, i: number) => (
                      <div className="hit" key={r.id}>
                        <b>{i + 1}</b>
                        <span>
                          <strong>{r.label}</strong>
                          <em>{r.group}</em>
                        </span>
                        <code>{r.distance}</code>
                      </div>
                    ))}
                  </article>
                ))}
              </div>
            </section>
          )}
          {benchmark && (
            <section className="bench">
              <div>
                <span className="eyebrow">REPEATABLE MEASUREMENT</span>
                <h2>Benchmark run</h2>
                <p>
                  150 sampled queries · {benchmark.metric} distance · recall
                  measured against brute force.
                </p>
              </div>
              <div className="bench-list">
                {benchmark.results.map((r: any) => (
                  <div key={r.algorithm}>
                    <strong>{r.algorithm.replace("_", " ")}</strong>
                    <span>{r.mean_ms} ms avg</span>
                    <span>{Math.round(r.recall_at_k * 100)}% recall</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </section>
      ) : (
        <section className="shell documents">
          <div className="intro">
            <div>
              <span className="eyebrow">DOCUMENT RETRIEVAL WORKSPACE</span>
              <h2>
                Your documents,
                <br />
                ready to search.
              </h2>
              <p>
                Paste source material, embed it with the configured provider,
                and inspect what is retrieved before an answer is written.
              </p>
            </div>
            <div className="model-card">
              <label>
                {modelStatus?.provider === "gemini" ? "Gemini embedding model" : "Ollama embedding model"}
                <input
                  value={embedding}
                  onChange={(e) => setEmbedding(e.target.value)}
                />
              </label>
              <label>
                {modelStatus?.provider === "gemini" ? "Gemini chat model" : "Ollama chat model"}
                <input value={chat} onChange={(e) => setChat(e.target.value)} />
              </label>
              <p className={modelStatus?.available ? "model-ready" : "model-error"}>
                {modelStatus?.available
                  ? modelStatus.provider === "gemini"
                    ? "Gemini API key configured"
                    : `Ollama online · ${modelStatus.models.length} local models found`
                  : modelStatus?.detail || "AI provider is unavailable"}
              </p>
            </div>
          </div>
          <div className="doc-grid">
            <form className="paper" onSubmit={ingest}>
              <span className="eyebrow">ADD SOURCE</span>
              <h3>Document intake</h3>
              <label>
                Title
                <input
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Field notes, product brief…"
                />
              </label>
              <label>
                Text
                <textarea
                  required
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste text here. It will be split into overlapping passages before embedding."
                />
              </label>
              <button className="primary" disabled={busy}>
                {busy ? "Embedding…" : "Embed & store"}
              </button>
              <div className="doc-list">
                {docs.map((d) => (
                  <div key={d.id}>
                    <span>
                      <b>{d.title}</b>
                      <small>{d.chunk_count} chunks</small>
                    </span>
                    <button
                      type="button"
                      onClick={async () => {
                        await api("/documents/" + d.id, { method: "DELETE" });
                        refresh();
                      }}
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </form>
            <div className="answer-panel">
              <span className="eyebrow">ASK THE COLLECTION</span>
              <h3>Grounded answer</h3>
              <form className="question" onSubmit={ask}>
                <textarea
                  required
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question about the documents…"
                />
                <button className="primary" disabled={busy || !docs.length}>
                  {busy ? "Working…" : "Ask documents"}
                </button>
              </form>
              {contexts.length > 0 && (
                <div className="contexts">
                  <h4>Retrieved context</h4>
                  {contexts.map((c, i) => (
                    <details key={i}>
                      <summary>
                        {c.title} · chunk {c.chunk}
                        <span>distance {c.distance}</span>
                      </summary>
                      <p>{c.text}</p>
                    </details>
                  ))}
                </div>
              )}
              <div className={"answer " + (!answer ? "empty" : "")}>
                {answer || "Your grounded response will stream here."}
              </div>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
