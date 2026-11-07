import ollama
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse

from common import connect, embed_text

VLM_MODEL = "llava:7b"

# --- db connection (opened once) ---
# autocommit: otherwise every read leaves the connection idle in transaction,
# and one failed statement would abort it for all later requests
conn = connect()
conn.autocommit = True

app = FastAPI()

# HNSW returns at most hnsw.ef_search rows (default 40, max 1000), so it is raised
# to k for this query; otherwise k > 40 would be silently truncated
@app.get("/search")
def search(q: str, k: int = Query(12, ge=1, le=1000)):
    emb = embed_text(q)
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, k)),))
        rows = conn.execute(
            """SELECT id, path, category, embedding <=> %s::vector AS dist
               FROM images ORDER BY dist LIMIT %s""",
            (emb, k),
        ).fetchall()
    return [{"id": i, "path": p, "category": c, "dist": float(d)} for i, p, c, d in rows]

# runs the VLM on demand for a single image, caches the result in the db
@app.get("/caption")
def caption(id: int):
    row = conn.execute("SELECT path, caption FROM images WHERE id = %s", (id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="image not found")
    path, existing = row
    if existing:                      # already captioned before -> return cached
        return {"caption": existing, "cached": True}
    # captions are optional: if Ollama is down or the model is not pulled, say so instead of a 500
    try:
        response = ollama.chat(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": "Describe this image in one short sentence.",
                "images": [path],
            }],
        )
    except (ConnectionError, ollama.ResponseError) as e:
        raise HTTPException(status_code=503, detail=f"captioning unavailable: {e}")
    text = response["message"]["content"].strip()
    conn.execute("UPDATE images SET caption = %s WHERE id = %s", (text, id))
    return {"caption": text, "cached": False}

# serves only files registered in the db; taking a raw path would expose any file on disk
@app.get("/image")
def image(id: int):
    row = conn.execute("SELECT path FROM images WHERE id = %s", (id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(row[0])

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Search</title>
  <style>
    :root {
      --bg: #0b0d12; --bg2: #0f131b; --card: #151a23; --card2: #1b212c;
      --line: #232a36; --text: #eef1f6; --muted: #7d8595; --accent: #6ea0ff;
      --accent2: #a879ff; --radius: 16px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: radial-gradient(1200px 600px at 50% -10%, #16203a 0%, var(--bg) 55%);
      color: var(--text); min-height: 100vh;
    }
    .topbar {
      position: sticky; top: 0; z-index: 10;
      backdrop-filter: blur(12px);
      background: rgba(11,13,18,0.72);
      border-bottom: 1px solid var(--line);
      padding: 20px 24px;
    }
    .topbar-inner { max-width: 1180px; margin: 0 auto; }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
    .logo {
      width: 30px; height: 30px; border-radius: 9px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
    }
    .brand h1 { font-size: 18px; font-weight: 650; letter-spacing: -0.01em; }
    .brand .badge {
      margin-left: auto; font-size: 11px; color: var(--muted);
      border: 1px solid var(--line); padding: 4px 10px; border-radius: 20px;
    }
    #bar { display: flex; gap: 10px; }
    .input-wrap { position: relative; flex: 1; }
    .input-wrap svg { position: absolute; left: 15px; top: 50%; transform: translateY(-50%); opacity: .5; }
    input {
      width: 100%; padding: 15px 16px 15px 44px; font-size: 15px;
      background: var(--card); color: var(--text);
      border: 1px solid var(--line); border-radius: 13px; outline: none;
      transition: border-color .15s, box-shadow .15s;
    }
    input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(110,160,255,.15); }
    button {
      padding: 0 26px; font-size: 15px; font-weight: 600; border: 0; border-radius: 13px;
      background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; cursor: pointer;
      transition: opacity .15s, transform .1s;
    }
    button:hover { opacity: .92; }
    button:active { transform: scale(.98); }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 24px 60px; }
    .meta-row { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; height: 18px; }
    .count { font-size: 13px; color: var(--muted); }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip {
      font-size: 12px; color: var(--muted); background: var(--card);
      border: 1px solid var(--line); padding: 6px 12px; border-radius: 20px; cursor: pointer;
      transition: color .15s, border-color .15s;
    }
    .chip:hover { color: var(--text); border-color: var(--accent); }
    #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 18px; }
    .card {
      background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
      overflow: hidden; cursor: pointer;
      transition: transform .16s ease, border-color .16s, box-shadow .16s;
      animation: rise .35s ease both;
    }
    @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    .card:hover { transform: translateY(-4px); border-color: var(--accent);
                  box-shadow: 0 12px 30px -12px rgba(110,160,255,.4); }
    .thumb { position: relative; height: 190px; overflow: hidden; }
    .thumb img { width: 100%; height: 100%; object-fit: cover; display: block;
                 transition: transform .3s; }
    .card:hover .thumb img { transform: scale(1.05); }
    .overlay {
      position: absolute; inset: 0; display: flex; align-items: flex-end;
      background: linear-gradient(to top, rgba(0,0,0,.55), transparent 55%);
      opacity: 0; transition: opacity .16s;
    }
    .card:hover .overlay { opacity: 1; }
    .overlay span { font-size: 12px; font-weight: 600; color: #fff; padding: 12px; }
    .dist-pill {
      position: absolute; top: 10px; right: 10px;
      background: rgba(0,0,0,.5); backdrop-filter: blur(4px);
      font-size: 11px; color: #dfe6f5; padding: 3px 8px; border-radius: 20px;
    }
    .body { padding: 13px 14px 15px; }
    .tag { font-size: 11px; color: var(--accent); text-transform: uppercase;
           letter-spacing: .05em; font-weight: 600; }
    .cap { font-size: 13px; line-height: 1.5; margin-top: 9px; color: var(--text); min-height: 16px; }
    .cap.empty { color: var(--muted); font-style: italic; }
    .spinner {
      display: inline-block; width: 13px; height: 13px; vertical-align: -2px;
      border: 2px solid var(--line); border-top-color: var(--accent);
      border-radius: 50%; animation: spin .7s linear infinite; margin-right: 7px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .empty-state { text-align: center; color: var(--muted); padding: 80px 20px; }
    .empty-state .big { font-size: 40px; margin-bottom: 12px; opacity: .4; }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="logo"></div>
        <h1>Image Search</h1>
        <span class="badge">CLIP · pgvector · LLaVA</span>
      </div>
      <div id="bar">
        <div class="input-wrap">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2">
            <circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>
          </svg>
          <input id="q" placeholder="Search by meaning — a church, a dog, a chainsaw..." />
        </div>
        <button onclick="run()">Search</button>
      </div>
    </div>
  </div>

  <div class="wrap">
    <div class="meta-row">
      <span class="count" id="count"></span>
      <div class="chips" id="chips"></div>
    </div>
    <div id="grid"></div>
    <div class="empty-state" id="empty">
      <div class="big">◎</div>
      <div>Search your local image collection by meaning.</div>
    </div>
  </div>

  <script>
    const SUGGESTIONS = ['a church', 'a dog', 'a chainsaw', 'a fish', 'a cassette player'];

    function renderChips() {
      const c = document.getElementById('chips');
      c.innerHTML = '';
      SUGGESTIONS.forEach(s => {
        const el = document.createElement('span');
        el.className = 'chip'; el.textContent = s;
        el.onclick = () => { document.getElementById('q').value = s; run(); };
        c.appendChild(el);
      });
    }

    async function run() {
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      document.getElementById('empty').style.display = 'none';
      const res = await fetch('/search?q=' + encodeURIComponent(q));
      const items = await res.json();
      document.getElementById('count').textContent = items.length + ' results for "' + q + '"';
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      items.forEach((it, idx) => {
        const card = document.createElement('div');
        card.className = 'card';
        card.style.animationDelay = (idx * 30) + 'ms';
        card.onclick = () => describe(it.id, card);
        card.innerHTML =
          '<div class="thumb">' +
            '<img src="/image?id=' + it.id + '">' +
            '<div class="dist-pill">' + it.dist.toFixed(3) + '</div>' +
            '<div class="overlay"><span>Click to describe →</span></div>' +
          '</div>' +
          '<div class="body">' +
            '<div class="tag">' + it.category + '</div>' +
            '<div class="cap empty">not described yet</div>' +
          '</div>';
        grid.appendChild(card);
      });
    }

    async function describe(id, card) {
      const cap = card.querySelector('.cap');
      if (cap.dataset.done) return;
      cap.className = 'cap';
      cap.innerHTML = '<span class="spinner"></span>describing...';
      const res = await fetch('/caption?id=' + id);
      const data = await res.json();
      if (!res.ok) {                  // leave it retryable, e.g. after starting Ollama
        cap.className = 'cap empty';
        cap.textContent = data.detail || 'captioning failed';
        return;
      }
      cap.className = 'cap';
      cap.textContent = data.caption;
      cap.dataset.done = '1';
    }

    document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
    renderChips();
  </script>
</body>
</html>
"""