#!/usr/bin/env python
"""Chat with your PDF — upload a PDF and ask questions; answers are grounded in the
document with page citations (real RAG: chunk -> embed -> retrieve -> answer).

Embeddings: fastembed (ONNX, CPU-friendly, no torch). LLM: Anthropic Claude API if
ANTHROPIC_API_KEY is set, else a local Ollama model.

  pip install flask anthropic pypdf fastembed numpy
  python app.py            # http://127.0.0.1:8500

Project #3 of the "30 Projects in 15 Days" challenge — GritAI.
"""
import os, io, json, re, uuid
import numpy as np
from flask import Flask, request, Response, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "8500"))
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

MAX_PAGES = 120
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 160
TOP_K = 6
MAX_DOCS = 24

SAMPLE_DIR = os.path.join(HERE, "sample_pdfs")
SAMPLES = {
    "memo": ("simple_memo.pdf", "📋", "Office memo", "1-page internal memo"),
    "manual": ("product_manual.pdf", "📘", "Product manual", "Smart thermostat guide"),
    "report": ("research_report.pdf", "📊", "Market report", "Data, tables, findings"),
    "handbook": ("employee_handbook.pdf", "📗", "Employee handbook", "13 pages, many sections"),
}

# ---- embedding model (lazy singleton) ----------------------------------------
_EMB = None
def emb_model():
    global _EMB
    if _EMB is None:
        from fastembed import TextEmbedding
        _EMB = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _EMB

def embed_passages(texts):
    return np.array(list(emb_model().passage_embed(texts)), dtype=np.float32)

def embed_query(text):
    return np.array(list(emb_model().query_embed([text]))[0], dtype=np.float32)

def _norm(m):
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(n, 1e-8, None)

# ---- pdf -> chunks -----------------------------------------------------------
def extract_pages(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else "").strip()
    pages = []
    for i, pg in enumerate(reader.pages[:MAX_PAGES], start=1):
        txt = (pg.extract_text() or "").strip()
        if txt:
            pages.append((i, re.sub(r"[ \t]+", " ", txt)))
    return title, len(reader.pages), pages

def chunk_pages(pages):
    chunks = []
    for pageno, text in pages:
        i = 0
        while i < len(text):
            piece = text[i:i + CHUNK_CHARS].strip()
            if len(piece) > 40:
                chunks.append({"page": pageno, "text": piece})
            i += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks

DOCS = {}
DOC_ORDER = []

def index_pdf(data, filename):
    title, npages, pages = extract_pages(data)
    if not pages:
        raise ValueError("No extractable text found — this PDF may be scanned images (needs OCR).")
    chunks = chunk_pages(pages)
    emb = _norm(embed_passages([c["text"] for c in chunks]))
    doc_id = uuid.uuid4().hex[:12]
    disp = re.sub(r"\.(html?|pdf)$", "", (title or ""), flags=re.I).strip()
    if not disp:
        disp = re.sub(r"\.pdf$", "", filename or "document", flags=re.I)
    DOCS[doc_id] = {"title": disp, "pages": npages, "chunks": chunks, "emb": emb}
    DOC_ORDER.append(doc_id)
    while len(DOC_ORDER) > MAX_DOCS:
        DOCS.pop(DOC_ORDER.pop(0), None)
    return doc_id, DOCS[doc_id]

def retrieve(doc, question, k=TOP_K):
    q = _norm(embed_query(question).reshape(1, -1))[0]
    sims = doc["emb"] @ q
    idx = np.argsort(-sims)[:k]
    return [doc["chunks"][i] for i in idx]

# ---- answer (streaming) ------------------------------------------------------
def system_prompt():
    return (
        "You answer questions about a specific document using ONLY the excerpts provided.\n"
        "- Cite the page number(s) you used, in parentheses like (p. 4).\n"
        "- If the answer is not in the excerpts, say you couldn't find it in the document — do not guess.\n"
        "- Be concise and faithful. Write plain text; no markdown symbols like ** or #."
    )

def build_context(chunks):
    return "\n\n".join(f"[p. {c['page']}] {c['text']}" for c in chunks)

def answer_from(chunks, question):
    ctx = build_context(chunks)
    user = f"Document excerpts:\n\n{ctx}\n\n---\nQuestion: {question}"
    if ANTHROPIC_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        with client.messages.stream(model=ANTHROPIC_MODEL, max_tokens=700,
                                     system=system_prompt(),
                                     messages=[{"role": "user", "content": user}]) as s:
            for t in s.text_stream:
                yield t
        return
    import requests
    with requests.post(f"{OLLAMA_URL}/api/chat", stream=True, timeout=120, json={
            "model": OLLAMA_MODEL, "stream": True,
            "messages": [{"role": "system", "content": system_prompt()},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0.2}}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if d.get("message", {}).get("content"):
                yield d["message"]["content"]
            if d.get("done"):
                break

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")

@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("pdf")
    if not f:
        return jsonify(error="No file uploaded."), 400
    try:
        doc_id, doc = index_pdf(f.read(), f.filename or "document.pdf")
        return jsonify(doc_id=doc_id, title=doc["title"], pages=doc["pages"], chunks=len(doc["chunks"]))
    except Exception as e:
        return jsonify(error=str(e)[:200]), 400

@app.route("/api/sample", methods=["POST"])
def api_sample():
    name = request.get_json(force=True).get("name")
    meta = SAMPLES.get(name)
    if not meta:
        return jsonify(error="unknown sample"), 400
    path = os.path.join(SAMPLE_DIR, meta[0])
    if not os.path.exists(path):
        return jsonify(error="sample not available"), 404
    try:
        with open(path, "rb") as fh:
            doc_id, doc = index_pdf(fh.read(), meta[0])
        return jsonify(doc_id=doc_id, title=doc["title"], pages=doc["pages"], chunks=len(doc["chunks"]))
    except Exception as e:
        return jsonify(error=str(e)[:200]), 400

@app.route("/api/ask", methods=["POST"])
def api_ask():
    b = request.get_json(force=True)
    doc = DOCS.get(b.get("doc_id", ""))
    question = (b.get("question") or "").strip()
    def gen():
        import sys as _sys
        if not doc:
            yield "data: " + json.dumps({"t": "That document isn't loaded anymore — please re-upload it."}) + "\n\n"
        elif not question:
            yield "data: " + json.dumps({"t": "Please ask a question."}) + "\n\n"
        else:
            try:
                chunks = retrieve(doc, question)
                src = {}
                for c in chunks:
                    src.setdefault(str(c["page"]), c["text"][:420])
                yield "data: " + json.dumps({"sources": src}) + "\n\n"
                for chunk in answer_from(chunks, question):
                    yield "data: " + json.dumps({"t": chunk}) + "\n\n"
            except Exception as e:
                print("ask error:", repr(e), file=_sys.stderr, flush=True)
                yield "data: " + json.dumps({"t": "Sorry — I hit an error answering. Please try again."}) + "\n\n"
        yield "data: [DONE]\n\n"
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Chat with your PDF</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--teal:#0d9488;--teal2:#0f766e;--deep:#0b3b39;--ink:#132320;--bg:#eef4f3;--card:#fff;
 --muted:#6f807c;--line:#e0eae8;--cite:#0d9488;--cite-bg:#e6f5f2}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:var(--ink);
 background:var(--bg);display:flex;justify-content:center;overflow-x:hidden}
.amb{position:fixed;inset:0;z-index:0;pointer-events:none;
 background:radial-gradient(55vw 45vh at 82% -6%,rgba(13,148,136,.18),transparent 60%),
  radial-gradient(50vw 40vh at 6% 8%,rgba(79,70,229,.10),transparent 62%),
  radial-gradient(60vw 50vh at 50% 112%,rgba(13,148,136,.10),transparent 60%)}
.app{position:relative;z-index:1;width:100%;max-width:680px;min-height:100dvh;display:flex;flex-direction:column;
 background:linear-gradient(180deg,rgba(255,255,255,.6),var(--card) 12%);}
header{padding:16px 22px;color:#fff;background:linear-gradient(120deg,var(--teal),var(--deep));
 display:flex;align-items:center;gap:13px;position:relative;overflow:hidden}
header::after{content:"";position:absolute;right:-40px;top:-60px;width:200px;height:200px;border-radius:50%;
 background:radial-gradient(circle,rgba(255,255,255,.12),transparent 70%)}
.brand-av{width:44px;height:44px;border-radius:12px;display:grid;place-items:center;font-size:22px;
 background:rgba(255,255,255,.17);box-shadow:0 2px 10px rgba(0,0,0,.18);flex:none;z-index:1}
header h1{margin:0;font-size:19px;font-weight:700;letter-spacing:-.2px}
header .sub{margin:2px 0 0;font-size:12.5px;color:rgba(255,255,255,.85)}
.tag{margin-left:auto;z-index:1;font-size:10px;letter-spacing:.14em;text-transform:uppercase;
 background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.22);border-radius:20px;padding:5px 11px;font-weight:600}

/* ---- upload view ---- */
#uploadView{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:30px 22px;text-align:center}
.hero-badge{width:64px;height:64px;border-radius:18px;display:grid;place-items:center;font-size:32px;margin-bottom:18px;
 background:linear-gradient(135deg,#d7f2ee,#c3e9ff);box-shadow:0 10px 30px -10px rgba(13,148,136,.5)}
#uploadView h2{margin:0;font-size:30px;font-weight:800;letter-spacing:-.5px;
 background:linear-gradient(120deg,var(--deep),var(--teal));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
#uploadView .lead{margin:10px 0 20px;color:var(--muted);font-size:15px;max-width:46ch;line-height:1.5}
.feats{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-bottom:24px}
.feat{display:inline-flex;align-items:center;gap:7px;background:var(--card);border:1px solid var(--line);
 border-radius:22px;padding:7px 14px;font-size:12.5px;color:var(--ink);box-shadow:0 2px 8px rgba(19,35,32,.04)}
.feat b{font-weight:600}
#drop{width:100%;max-width:460px;border:2px dashed #bcd6d2;border-radius:20px;padding:36px 24px;cursor:pointer;
 transition:.18s;background:linear-gradient(180deg,#fbfefe,#f4faf9)}
#drop:hover,#drop.drag{border-color:var(--teal);background:#eafaf7;transform:translateY(-2px);box-shadow:0 16px 34px -20px rgba(13,148,136,.45)}
#drop .ic{font-size:38px}#drop .t{font-weight:700;margin:8px 0 3px;font-size:16px}#drop .h{color:var(--muted);font-size:13px}
.samples{margin-top:26px;width:100%;max-width:520px}
.samples .lbl{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:11px}
.sgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.samp{display:flex;align-items:center;gap:11px;text-align:left;background:var(--card);border:1px solid var(--line);
 border-radius:13px;padding:11px 13px;cursor:pointer;transition:.15s}
.samp:hover{border-color:var(--teal);transform:translateY(-2px);box-shadow:0 10px 22px -14px rgba(13,148,136,.4)}
.samp .e{font-size:22px;flex:none}.samp .tt{font-weight:600;font-size:13.5px}.samp .hh{color:var(--muted);font-size:11.5px}

/* ---- chat view ---- */
#chatView{flex:1;display:none;flex-direction:column;min-height:0}
.docbar{display:flex;align-items:center;gap:10px;padding:11px 18px;border-bottom:1px solid var(--line);
 background:linear-gradient(180deg,#f5fbfa,#fff)}
.docbar .dic{font-size:18px}.docbar .dt{font-weight:700;font-size:14px}
.docbar .dm{color:var(--muted);font-size:12px}
.docbar .new{margin-left:auto;color:var(--teal);cursor:pointer;font-size:12.5px;font-weight:600;
 border:1px solid var(--line);border-radius:18px;padding:6px 12px;transition:.15s}
.docbar .new:hover{background:var(--cite-bg);border-color:var(--teal)}
#log{flex:1;overflow-y:auto;padding:18px 16px;display:flex;flex-direction:column;gap:13px}
#log::-webkit-scrollbar{width:8px}#log::-webkit-scrollbar-thumb{background:#cfe2de;border-radius:8px}
.row{display:flex;gap:9px;align-items:flex-end;animation:rise .28s ease both}
.row.user{flex-direction:row-reverse}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.av{width:30px;height:30px;border-radius:50%;flex:none;display:grid;place-items:center;font-size:15px;
 background:linear-gradient(135deg,#d7f2ee,#c3e9ff)}
.msg{max-width:80%;padding:12px 15px;border-radius:16px;line-height:1.55;font-size:14.5px;white-space:pre-wrap;
 box-shadow:0 2px 6px rgba(19,35,32,.06)}
.msg.bot{background:var(--card);border:1px solid var(--line);border-bottom-left-radius:5px}
.msg.user{background:linear-gradient(135deg,var(--teal),var(--teal2));color:#fff;border-bottom-right-radius:5px}
.cite{position:relative;display:inline-block;background:var(--cite-bg);color:var(--cite);border-radius:6px;padding:0 6px;
 font-size:12px;font-weight:600;white-space:nowrap;cursor:help}
.cite .pop{display:none;position:absolute;bottom:135%;left:50%;transform:translateX(-50%);width:290px;max-width:74vw;
 background:#0f2a28;color:#eafaf7;border-radius:11px;padding:11px 13px;font-size:12px;font-weight:400;line-height:1.5;
 box-shadow:0 14px 34px rgba(0,0,0,.32);z-index:30;white-space:normal;text-align:left}
.cite .pop::after{content:"";position:absolute;top:100%;left:50%;transform:translateX(-50%);border:6px solid transparent;border-top-color:#0f2a28}
.cite:hover .pop{display:block}
.cite .ps{margin:5px 0}.cite .ps:first-child{margin-top:0}.cite .ps b{color:#7fe6d6;font-weight:700}
.starters{display:flex;flex-wrap:wrap;gap:8px;padding:2px 4px 4px 43px}
.starter{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:7px 12px;font-size:12.5px;
 cursor:pointer;color:var(--teal);transition:.15s}
.starter:hover{border-color:var(--teal);background:var(--cite-bg)}
form{display:flex;gap:9px;padding:12px 14px;border-top:1px solid var(--line);background:var(--card)}
input#q{flex:1;padding:13px 16px;border:1px solid var(--line);border-radius:24px;font-size:15px;background:#f8fcfb;color:var(--ink);outline:none;transition:.15s}
input#q:focus{border-color:var(--teal);background:#fff;box-shadow:0 0 0 3px rgba(13,148,136,.14)}
button.send{flex:none;width:46px;height:46px;border-radius:50%;border:0;cursor:pointer;display:grid;place-items:center;
 background:linear-gradient(135deg,var(--teal),var(--teal2));color:#fff;transition:.15s}
button.send:hover{filter:brightness(1.1)}button.send:disabled{opacity:.45}button.send svg{width:18px;height:18px}
.typing{display:inline-flex;gap:4px}.typing span{width:7px;height:7px;border-radius:50%;background:var(--teal);opacity:.5;animation:bd 1.2s infinite ease-in-out}
.typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
@keyframes bd{0%,60%,100%{transform:translateY(0);opacity:.35}30%{transform:translateY(-6px);opacity:1}}
.foot{text-align:center;font-size:10.5px;color:var(--muted);padding:8px}.foot b{color:#4a5d59}
</style></head>
<body><div class="amb"></div><div class="app">
<header>
 <div class="brand-av">📄</div>
 <div><h1>Chat with your PDF</h1><div class="sub">Upload a document — answers cite the exact page</div></div>
 <span class="tag">RAG</span>
</header>

<div id="uploadView">
 <div class="hero-badge">📄</div>
 <h2>Ask your documents anything</h2>
 <p class="lead">Drop in a PDF and get answers grounded in its contents — every answer cites the page it came from, and it won't make things up.</p>
 <div class="feats">
  <span class="feat">⚡ <b>Instant</b> answers</span>
  <span class="feat">📍 <b>Cites</b> the page</span>
  <span class="feat">🛡️ <b>Won't</b> guess</span>
 </div>
 <div id="drop">
  <div class="ic">⬆️</div><div class="t">Drop a PDF here or click to upload</div>
  <div class="h">Up to 20 MB · text-based PDFs</div>
  <input type="file" id="file" accept="application/pdf" style="display:none">
 </div>
 <div class="samples">
  <div class="lbl">— or try a sample —</div>
  <div class="sgrid" id="sgrid"></div>
 </div>
</div>

<div id="chatView">
 <div class="docbar"><span class="dic">📄</span><div><div class="dt" id="dt">document</div><div class="dm" id="dm"></div></div>
  <span class="new" onclick="reset()">↺ New PDF</span></div>
 <div id="log"></div>
 <div class="starters" id="starters"></div>
 <form id="form" onsubmit="ask();return false;">
  <input id="q" placeholder="Ask about the document…" autocomplete="off">
  <button class="send" id="btn" aria-label="Send">
   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
  </button>
 </form>
</div>
<div class="foot">Answers grounded in your uploaded PDF · <b>GritAI</b></div>
</div><script>
const SAMPLES=[["memo","📋","Office memo","1-page internal memo"],
 ["manual","📘","Product manual","Smart thermostat guide"],
 ["report","📊","Market report","Data, tables, findings"],
 ["handbook","📗","Employee handbook","13 pages, many sections"]];
const STARTERS=["Summarize this document","What are the key points?","What should I know first?"];
const $=id=>document.getElementById(id);
const drop=$('drop'),file=$('file'),log=$('log'),form=$('form'),q=$('q'),btn=$('btn'),
 uploadView=$('uploadView'),chatView=$('chatView'),dt=$('dt'),dm=$('dm');
let docId=null,busy=false;
// render sample buttons
$('sgrid').innerHTML=SAMPLES.map(s=>`<div class="samp" onclick="loadSample('${s[0]}')"><span class="e">${s[1]}</span><div><div class="tt">${s[2]}</div><div class="hh">${s[3]}</div></div></div>`).join('');
$('starters').innerHTML=STARTERS.map(s=>`<span class="starter" onclick="ask('${s.replace(/'/g,"\\\\'")}')">${s}</span>`).join('');
drop.onclick=()=>file.click();
file.onchange=e=>{if(e.target.files[0])upload(e.target.files[0]);};
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('drag');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('drag');}));
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f&&f.type==='application/pdf')upload(f);});
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const CITE=/\\(pp?\\.?\\s*\\d+(?:\\s*(?:,|;|&amp;|and)\\s*(?:pp?\\.?\\s*)?\\d+)*\\)/gi;
function popFor(m,sources){if(!sources)return'';const seen={};let out='';
 (m.match(/\\d+/g)||[]).forEach(p=>{if(sources[p]&&!seen[p]){seen[p]=1;out+='<span class="ps"><b>p. '+p+'</b> '+esc(sources[p])+'…</span>';}});
 return out?'<span class="pop">'+out+'</span>':'';}
function render(el,text,sources){el.innerHTML=esc(text).replace(CITE,m=>'<span class="cite">'+m+popFor(m,sources)+'</span>');}
function add(t,who){const row=document.createElement('div');row.className='row '+who;
 if(who==='bot'){const a=document.createElement('div');a.className='av';a.textContent='📄';row.appendChild(a);}
 const d=document.createElement('div');d.className='msg '+who;d.textContent=t;row.appendChild(d);
 log.appendChild(row);log.scrollTop=log.scrollHeight;return d;}
function showChat(d){docId=d.doc_id;dt.textContent=d.title;dm.textContent=d.pages+' pages · '+d.chunks+' chunks indexed';
 uploadView.style.display='none';chatView.style.display='flex';log.innerHTML='';
 add('Loaded "'+d.title+'". Ask me anything about it — I\\'ll cite the pages I use.','bot');q.focus();}
function reset(){docId=null;chatView.style.display='none';uploadView.style.display='flex';file.value='';
 drop.innerHTML='<div class="ic">⬆️</div><div class="t">Drop a PDF here or click to upload</div><div class="h">Up to 20 MB · text-based PDFs</div>';drop.appendChild(file);}
async function post(url,opts){const r=await fetch(url,opts);return r.json();}
async function upload(f){
 drop.innerHTML='<div class="ic">⏳</div><div class="t">Reading & indexing…</div><div class="h">'+esc(f.name)+'</div>';
 const fd=new FormData();fd.append('pdf',f);
 try{const d=await post('/api/upload',{method:'POST',body:fd});
  if(d.error){drop.innerHTML='<div class="ic">⚠️</div><div class="t">Couldn\\'t read that PDF</div><div class="h">'+esc(d.error)+'</div>';drop.appendChild(file);return;}
  showChat(d);}catch(e){drop.innerHTML='<div class="ic">⚠️</div><div class="t">Upload failed — click to retry</div>';drop.appendChild(file);}
}
async function loadSample(name){
 const box=[...document.querySelectorAll('.samp')].find(x=>x.getAttribute('onclick').includes("'"+name+"'"));
 if(box)box.style.opacity=.5;
 try{const d=await post('/api/sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  if(d.error){alert(d.error);}else showChat(d);}catch(e){}
 if(box)box.style.opacity=1;
}
async function ask(preset){if(busy||!docId)return;const question=(preset||q.value).trim();if(!question)return;
 q.value='';busy=true;btn.disabled=true;add(question,'user');
 const b=add('','bot');b.innerHTML='<span class="typing"><span></span><span></span><span></span></span>';
 let full='',srcMap={};
 try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({doc_id:docId,question})});
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
  while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
   let i;while((i=buf.indexOf('\\n\\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+2);
    if(!line.startsWith('data:'))continue;const p=line.slice(5).trim();if(p==='[DONE]')continue;
    try{const o=JSON.parse(p);if(o.sources){srcMap=o.sources;}else if(o.t){full+=o.t;b.textContent=full;log.scrollTop=log.scrollHeight;}}catch(e){}}}
  if(full)render(b,full,srcMap);else b.textContent='No answer produced — try rephrasing.';
 }catch(e){b.textContent='Sorry — something went wrong.';}
 busy=false;btn.disabled=false;q.focus();}
</script></body></html>"""

if __name__ == "__main__":
    backend = "Claude API" if ANTHROPIC_KEY else f"Ollama ({OLLAMA_MODEL} @ {OLLAMA_URL})"
    print(f"Chat with your PDF on http://127.0.0.1:{PORT}  [backend: {backend}]", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
