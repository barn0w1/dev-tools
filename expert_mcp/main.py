"""
expert_mcp — MCP server that lets Claude Code ask a human expert questions.

Architecture:
  - FastMCP (stdio) handles MCP protocol with Claude Code
  - FastAPI (background thread) serves the web UI and answer API
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    question: str
    context: str
    already_tried: str
    event: threading.Event = field(default_factory=threading.Event)
    answer: Optional[str] = None


_questions: dict[str, Question] = {}
_questions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# FastAPI web server
# ---------------------------------------------------------------------------

web_app = FastAPI(title="Expert MCP UI")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Expert MCP</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem; }
  h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 1.5rem; color: #f8fafc; }
  #waiting { color: #94a3b8; font-style: italic; margin-top: 2rem; text-align: center; }
  .q-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem;
    margin-bottom: 1rem; cursor: pointer; transition: border-color 0.15s;
  }
  .q-card:hover { border-color: #6366f1; }
  .q-card.selected { border-color: #818cf8; background: #1e2a4a; }
  .q-header {
    padding: 1rem 1.25rem; display: flex; align-items: flex-start; gap: 0.75rem;
  }
  .q-num {
    background: #6366f1; color: #fff; font-size: 0.75rem; font-weight: 700;
    border-radius: 9999px; min-width: 1.5rem; height: 1.5rem; display: flex;
    align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px;
  }
  .q-text { font-size: 0.975rem; line-height: 1.5; }
  .q-body { padding: 0 1.25rem 1.25rem; display: none; }
  .q-card.selected .q-body { display: block; }
  .q-section { margin-bottom: 0.75rem; }
  .q-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 0.3rem; }
  .q-content { background: #0f172a; border-radius: 0.5rem; padding: 0.6rem 0.75rem;
               font-size: 0.875rem; white-space: pre-wrap; color: #cbd5e1; }
  textarea {
    width: 100%; background: #0f172a; border: 1px solid #334155; border-radius: 0.5rem;
    color: #e2e8f0; font-size: 0.9rem; padding: 0.6rem 0.75rem; resize: vertical;
    min-height: 6rem; font-family: inherit;
  }
  textarea:focus { outline: none; border-color: #6366f1; }
  .btn-row { display: flex; gap: 0.75rem; margin-top: 0.75rem; }
  button {
    padding: 0.5rem 1.25rem; border-radius: 0.5rem; font-size: 0.875rem;
    font-weight: 600; cursor: pointer; border: none; transition: opacity 0.15s;
  }
  button:hover { opacity: 0.85; }
  .btn-submit { background: #6366f1; color: #fff; }
  .btn-skip { background: #334155; color: #94a3b8; }
</style>
</head>
<body>
<h1>Expert MCP — Pending Questions</h1>
<div id="list"></div>
<div id="waiting" style="display:none">Waiting for questions from Claude Code...</div>

<script>
let selected = null;

async function poll() {
  try {
    const res = await fetch('/api/questions');
    const questions = await res.json();
    render(questions);
  } catch(e) { /* ignore */ }
}

function render(questions) {
  const list = document.getElementById('list');
  const waiting = document.getElementById('waiting');

  if (questions.length === 0) {
    list.innerHTML = '';
    waiting.style.display = 'block';
    selected = null;
    return;
  }
  waiting.style.display = 'none';

  // Keep existing selections stable
  const ids = new Set(questions.map(q => q.id));
  if (selected && !ids.has(selected)) selected = null;

  list.innerHTML = questions.map((q, i) => {
    const isSelected = q.id === selected;
    const hasCtx = q.context && q.context.trim();
    const hasTried = q.already_tried && q.already_tried.trim();
    return `
    <div class="q-card${isSelected ? ' selected' : ''}" onclick="toggle('${q.id}')">
      <div class="q-header">
        <div class="q-num">${i + 1}</div>
        <div class="q-text">${escHtml(q.question)}</div>
      </div>
      <div class="q-body">
        ${hasCtx ? `<div class="q-section"><div class="q-label">Context</div><div class="q-content">${escHtml(q.context)}</div></div>` : ''}
        ${hasTried ? `<div class="q-section"><div class="q-label">Already tried</div><div class="q-content">${escHtml(q.already_tried)}</div></div>` : ''}
        <div class="q-section">
          <div class="q-label">Your answer</div>
          <textarea id="ans-${q.id}" placeholder="Type your answer here..." onclick="event.stopPropagation()"></textarea>
        </div>
        <div class="btn-row">
          <button class="btn-submit" onclick="event.stopPropagation(); submit('${q.id}')">Submit</button>
          <button class="btn-skip" onclick="event.stopPropagation(); skip('${q.id}')">Skip</button>
        </div>
      </div>
    </div>`;
  }).join('');
}

function toggle(id) {
  selected = selected === id ? null : id;
  // Re-render without re-fetching to keep textarea content
  document.querySelectorAll('.q-card').forEach(card => {
    const match = card.querySelector('.q-body');
    if (card.onclick.toString().includes(`'${id}'`)) {
      card.classList.toggle('selected');
      match.style.display = card.classList.contains('selected') ? 'block' : 'none';
    }
  });
}

async function submit(id) {
  const ta = document.getElementById('ans-' + id);
  const answer = ta ? ta.value.trim() : '';
  if (!answer) { ta.style.borderColor = '#ef4444'; return; }
  await fetch('/api/answer/' + id, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({answer})
  });
  selected = null;
  poll();
}

async function skip(id) {
  await fetch('/api/skip/' + id, { method: 'POST' });
  selected = null;
  poll();
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

setInterval(poll, 1000);
poll();
</script>
</body>
</html>
"""


@web_app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@web_app.get("/api/questions")
async def get_questions():
    with _questions_lock:
        return [
            {
                "id": q.id,
                "question": q.question,
                "context": q.context,
                "already_tried": q.already_tried,
            }
            for q in _questions.values()
        ]


class AnswerBody(BaseModel):
    answer: str


@web_app.post("/api/answer/{question_id}")
async def post_answer(question_id: str, body: AnswerBody):
    with _questions_lock:
        q = _questions.get(question_id)
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")
    q.answer = body.answer.strip()
    q.event.set()
    return {"status": "ok"}


@web_app.post("/api/skip/{question_id}")
async def post_skip(question_id: str):
    with _questions_lock:
        q = _questions.get(question_id)
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")
    q.answer = None  # signals "skipped"
    q.event.set()
    return {"status": "skipped"}


def start_web_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the FastAPI server in a daemon background thread."""
    config = uvicorn.Config(web_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# FastMCP server (stdio)
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "expert_mcp",
    instructions=(
        "Use ask_expert ONLY when your training data is insufficient or you need "
        "project-specific decisions that cannot be inferred from available context. "
        "Ask ONE focused, self-contained question per call. "
        "Do NOT ask things answerable from your training data."
    ),
)


@mcp.tool(
    description=(
        "Ask a human expert a question and wait for their answer. "
        "Use ONLY when training data is insufficient or for project-specific decisions. "
        "Ask ONE focused question per call. "
        "Do NOT ask things answerable from training data (language syntax, well-known APIs, etc.)."
    )
)
def ask_expert(
    question: Annotated[str, Field(min_length=10, max_length=2000, description="The focused question to ask")],
    context: Annotated[str, Field(max_length=3000, description="Relevant code or background that helps the expert answer")] = "",
    already_tried: Annotated[str, Field(max_length=1000, description="What you already attempted, to avoid redundant suggestions")] = "",
) -> str:
    """Block until a human answers or skips the question via the web UI."""
    qid = str(uuid.uuid4())
    q = Question(
        id=qid,
        question=question,
        context=context,
        already_tried=already_tried,
    )
    with _questions_lock:
        _questions[qid] = q

    # Block until the human answers or skips
    q.event.wait()

    with _questions_lock:
        _questions.pop(qid, None)

    if q.answer:
        return q.answer
    return "No answer provided, proceed with best judgment."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_web_server()
    print("Expert MCP web UI running at http://127.0.0.1:8765", flush=True)
    mcp.run(transport="stdio")
