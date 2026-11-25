"""Local labelling page for the E1 human agreement check (Amendment 1, Section 6).

Serves run_code_data/e1/human_sample_30.csv at http://127.0.0.1:8765 and writes
each label back to that file as soon as it is chosen. Only the critique text is
shown; the agent's labels are never read, so the check stays blind.

Run from repo root:  python scripts/e1_label_ui.py
Stdlib only. Binds to 127.0.0.1.
"""

import csv
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "run_code_data", "e1", "human_sample_30.csv")
FIELDS = ["item_id", "critique_text", "specificity_label", "notes"]
LABELS = ("specific", "vague", "")
PORT = int(os.environ.get("E1_UI_PORT", "8765"))
_lock = threading.Lock()


def read_rows():
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    os.replace(tmp, CSV_PATH)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>E1 Labelling</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1f2328;--muted:#5f6670;--line:#e3e1dc;--spec:#1f6f43;--spec-bg:#e6f4ec;--vague:#8a4b0f;--vague-bg:#fbefe1;--accent:#2f5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#16181b;--card:#1f2226;--ink:#e8e6e3;--muted:#a0a6ad;--line:#2e3238;--spec:#7fd3a3;--spec-bg:#18331f;--vague:#f2b673;--vague-bg:#3a2a16;--accent:#8fb2f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:20px 16px 40px;display:grid;grid-template-columns:1fr 300px;gap:20px}
@media (max-width:860px){.wrap{grid-template-columns:1fr}}
header{grid-column:1/-1;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:20px;margin:0}.progress{flex:1;min-width:200px;height:8px;background:var(--line);border-radius:99px;overflow:hidden}
.progress>div{height:100%;background:var(--accent);width:0;transition:width .2s}.count{color:var(--muted);font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.meta{display:flex;justify-content:space-between;color:var(--muted);font-size:14px;margin-bottom:10px}
.critique{white-space:pre-wrap;font-size:17px;line-height:1.65;max-height:58vh;overflow:auto;padding-right:6px}
.buttons{display:flex;gap:12px;margin-top:18px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:10px;padding:10px 16px;cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.lab{flex:1;min-width:150px;font-weight:600;padding:14px 16px}
.lab.spec.on{background:var(--spec-bg);border-color:var(--spec);color:var(--spec)}
.lab.vague.on{background:var(--vague-bg);border-color:var(--vague);color:var(--vague)}
textarea{width:100%;margin-top:14px;font:inherit;font-size:14px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);min-height:44px}
.nav{display:flex;justify-content:space-between;margin-top:14px;gap:8px}.saved{color:var(--muted);font-size:13px;align-self:center}
aside h2{font-size:15px;margin:0 0 8px}aside p,aside li{font-size:14px;color:var(--muted)}aside ul{padding-left:18px;margin:6px 0 12px}
.tag{display:inline-block;font-weight:600;border-radius:6px;padding:0 6px}.tag.spec{background:var(--spec-bg);color:var(--spec)}.tag.vague{background:var(--vague-bg);color:var(--vague)}
.list{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.dot{width:30px;height:30px;border-radius:7px;border:1px solid var(--line);font-size:12px;padding:0;background:var(--card)}
.dot.spec{background:var(--spec-bg);border-color:var(--spec)}.dot.vague{background:var(--vague-bg);border-color:var(--vague)}.dot.cur{outline:2px solid var(--accent)}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px}
.done{margin-top:12px;padding:10px 12px;border-radius:8px;background:var(--spec-bg);color:var(--spec);display:none}
</style></head><body>
<div class="wrap">
<header><h1>E1 Labelling</h1><div class="progress"><div id="bar"></div></div><span class="count" id="count"></span></header>
<main class="card">
  <div class="meta"><span id="pos"></span><span>item <b id="iid"></b></span></div>
  <div class="critique" id="text"></div>
  <div class="buttons">
    <button class="lab spec" id="bs">Specific <kbd>S</kbd></button>
    <button class="lab vague" id="bv">Vague <kbd>V</kbd></button>
  </div>
  <textarea id="notes" placeholder="Notes (optional), saved when you move on"></textarea>
  <div class="nav"><button id="prev">&larr; Previous</button><span class="saved" id="saved"></span><button id="next">Next &rarr;</button></div>
  <div class="done" id="done">All 30 labelled. You can close this page; your labels are saved in human_sample_30.csv.</div>
</main>
<aside class="card">
  <h2>Rules</h2>
  <p><span class="tag spec">specific</span> names at least one of:</p>
  <ul><li>an exact failing input</li><li>a concrete logic edge case (e.g. "fails when N = 1")</li><li>a trace of what the code does</li></ul>
  <p><span class="tag vague">vague</span> is generic disapproval with none of these ("the logic is wrong", "too slow").</p>
  <p>Generic complaints plus one concrete point count as <b>specific</b>. Empty, or "wrong" with no detail, is <b>vague</b>. Judge only the text, not which model wrote it or whether the code was really wrong.</p>
  <p><kbd>S</kbd> / <kbd>V</kbd> label and go to the next item, <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> move between items.</p>
  <h2 style="margin-top:14px">Items</h2><div class="list" id="list"></div>
</aside>
</div>
<script>
let items=[], i=0;
const $=id=>document.getElementById(id);
async function load(){items=await (await fetch('/api/items')).json(); const f=items.findIndex(x=>!x.specificity_label); i=f<0?0:f; render();}
function render(){
  const it=items[i]; $('text').textContent=it.critique_text||'(empty critique)'; $('iid').textContent=it.item_id;
  $('pos').textContent=`${i+1} of ${items.length}`; $('notes').value=it.notes||'';
  $('bs').classList.toggle('on',it.specificity_label==='specific'); $('bv').classList.toggle('on',it.specificity_label==='vague');
  const n=items.filter(x=>x.specificity_label).length; $('count').textContent=`${n} / ${items.length} labelled`;
  $('bar').style.width=(100*n/items.length)+'%'; $('done').style.display=n===items.length?'block':'none';
  $('list').innerHTML=''; items.forEach((x,k)=>{const b=document.createElement('button');b.className='dot'+(x.specificity_label==='specific'?' spec':x.specificity_label==='vague'?' vague':'')+(k===i?' cur':'');b.textContent=k+1;b.title='item '+x.item_id;b.onclick=()=>go(k);$('list').appendChild(b);});
  $('text').scrollTop=0;
}
async function save(patch){
  const it=items[i]; Object.assign(it,patch);
  const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({item_id:it.item_id,specificity_label:it.specificity_label||'',notes:it.notes||''})});
  $('saved').textContent=r.ok?'Saved':'Save failed: is the server still running?';
}
async function go(k){ if(($('notes').value||'')!==(items[i].notes||'')) await save({notes:$('notes').value}); i=Math.max(0,Math.min(items.length-1,k)); render(); }
async function label(v){ await save({specificity_label:v,notes:$('notes').value}); render(); if(i<items.length-1) setTimeout(()=>go(i+1),180); }
$('bs').onclick=()=>label('specific'); $('bv').onclick=()=>label('vague');
$('prev').onclick=()=>go(i-1); $('next').onclick=()=>go(i+1);
document.addEventListener('keydown',e=>{ if(e.target.tagName==='TEXTAREA') return;
  if(e.key==='s'||e.key==='S') label('specific'); else if(e.key==='v'||e.key==='V') label('vague');
  else if(e.key==='ArrowRight') go(i+1); else if(e.key==='ArrowLeft') go(i-1); });
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/items":
            with _lock:
                self._send(200, json.dumps(read_rows(), ensure_ascii=False))
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path != "/api/label":
            return self._send(404, '{"error":"not found"}')
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        label = str(body.get("specificity_label", "")).strip().lower()
        if label not in LABELS:
            return self._send(400, '{"error":"label must be specific or vague"}')
        with _lock:
            rows = read_rows()
            hit = [r for r in rows if r["item_id"] == str(body.get("item_id"))]
            if not hit:
                return self._send(404, '{"error":"unknown item"}')
            hit[0]["specificity_label"] = label
            hit[0]["notes"] = str(body.get("notes", ""))
            write_rows(rows)
        self._send(200, '{"ok":true}')

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"E1 labelling page: http://127.0.0.1:{PORT}  (saving to {CSV_PATH})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
