# main.py
# Entry point for the Agent 101 demo system.
# Starts a FastAPI server that:
#   - Serves the HTML UI at http://localhost:<PORT>
#   - Exposes /api/chat for the agent pipeline
#   - Exposes /api/scribe for cheat sheet generation
#   - Exposes /api/stats and /api/clear for session management
#
# Run: python main.py
# Then open: http://localhost:<PORT>

import asyncio
import os
import re
import sys
import webbrowser
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from agents.base import get_azure_openai_config

try:
    AZURE_OPENAI_CONFIG = get_azure_openai_config()
except RuntimeError as exc:
    print(f"\n❌  {exc}")
    print("    Update .env with your Azure OpenAI endpoint, deployment, and API key.\n")
    sys.exit(1)

from memory import session
from agents.orchestrator import OrchestratorAgent
from agents.guide import GuideAgent
from agents.concept import ConceptAgent
from agents.viz import VizAgent
from agents.code_agent import CodeAgent
from agents.best_practices import BestPracticesAgent
from agents.mcp_agent import McpAgent
from agents.deploy import DeployAgent
from agents.qa import QaAgent
from agents.scribe import ScribeAgent


def should_auto_open_browser() -> bool:
    value = os.getenv("AUTO_OPEN_BROWSER", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_server_port() -> int:
    value = os.getenv("PORT", "").strip() or "8000"
    try:
        return int(value)
    except ValueError:
        print(f"[Warning] Invalid PORT={value!r}; falling back to 8000.")
        return 8000


SERVER_PORT = get_server_port()

# Instantiate all agents once at startup
orchestrator = OrchestratorAgent()
AGENTS = {
    "guide":   GuideAgent(),
    "concept": ConceptAgent(),
    "viz":     VizAgent(),
    "code":    CodeAgent(),
    "bp":      BestPracticesAgent(),
    "mcp":     McpAgent(),
    "deploy":  DeployAgent(),
    "qa":      QaAgent(),
    "scribe":  ScribeAgent(),
}


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    routing: dict | None = None
    forced_agents: list[str] | None = None

class AgentResponse(BaseModel):
    agent: str
    content: str
    is_viz: bool = False

class ChatResponse(BaseModel):
    routing: dict          # orchestrator decision — shown in trace bar
    responses: list[AgentResponse]
    stats: dict

class RouteResponse(BaseModel):
    routing: dict


# ── Core pipeline ────────────────────────────────────────────────────────────

def normalize_agents(agents: list[str], enforce_pairs: bool = True) -> list[str]:
    """Keep routing predictable and optionally enforce concept+viz pairing."""
    normalized: list[str] = []
    for agent in agents or ["qa"]:
        if agent not in normalized:
            normalized.append(agent)

    if enforce_pairs and "concept" in normalized and "viz" not in normalized:
        concept_idx = normalized.index("concept")
        normalized.insert(concept_idx + 1, "viz")
    elif enforce_pairs and "viz" in normalized and "concept" not in normalized:
        viz_idx = normalized.index("viz")
        normalized.insert(viz_idx, "concept")

    return normalized or ["qa"]

def _plain_text(text: str) -> str:
    cleaned = re.sub(r"[*`#_>-]+", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_concept_line(line: str) -> str:
    line = re.sub(r"^\s*\d+[\).\s:-]*", "", line)
    line = re.sub(r"^\s*[-•]\s*", "", line)
    return _plain_text(line)


def _canonical_heading(line: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
    mapping = {
        "plain language definition": "Definition",
        "plain language": "Definition",
        "definition": "Definition",
        "real world analogy": "Analogy",
        "analogy": "Analogy",
        "how it works technically": "Mechanics",
        "how it works": "Mechanics",
        "technical explanation": "Mechanics",
        "mechanics": "Mechanics",
        "why it matters": "Why it matters",
        "importance": "Why it matters",
    }
    return mapping.get(normalized, "")


def _concept_visual_items(text: str, limit: int = 4) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    pending_label = ""

    for raw_line in text.splitlines():
        cleaned = _normalize_concept_line(raw_line)
        if len(cleaned) < 2:
            continue

        heading = _canonical_heading(cleaned)
        if heading:
            pending_label = heading
            continue

        if ":" in cleaned:
            maybe_label, remainder = cleaned.split(":", 1)
            heading = _canonical_heading(maybe_label)
            remainder = _plain_text(remainder)
            if heading and remainder:
                items.append((heading, remainder))
                pending_label = ""
                if len(items) >= limit:
                    break
                continue

        if pending_label:
            items.append((pending_label, cleaned))
            pending_label = ""
        else:
            items.append(("", cleaned))

        if len(items) >= limit:
            break

    return items


def _concept_sentences(text: str, limit: int = 4) -> list[str]:
    line_candidates = []
    for raw_line in text.splitlines():
        cleaned = _normalize_concept_line(raw_line)
        if len(cleaned) >= 6:
            line_candidates.append(cleaned)

    if line_candidates:
        return line_candidates[:limit]

    plain = _normalize_concept_line(text)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", plain) if len(part.strip()) >= 6]
    return parts[:limit] if parts else ([plain] if plain else [])


def _pick_viz_style(text: str) -> str:
    lower = _plain_text(text).lower()
    if any(term in lower for term in ("loop", "cycle", "iterate", "iteration", "feedback", "retry")):
        return "cycle"
    if any(term in lower for term in ("compare", "comparison", "versus", "vs ", "trade-off", "tradeoff", "difference")):
        return "comparison"
    if any(term in lower for term in ("layer", "stack", "tier", "foundation")):
        return "layers"
    if any(term in lower for term in ("hierarchy", "tree", "parent", "child", "top-down")):
        return "hierarchy"
    return "sequence"


def build_viz_messages(user_message: str, concept_content: str) -> list[dict]:
    """Give the Viz agent a focused brief instead of the whole conversation."""
    visual_items = _concept_visual_items(concept_content)
    sentences = _concept_sentences(concept_content)
    title = user_message.strip() or (sentences[0] if sentences else "Agent concept")
    style = "concept_grid" if any(label for label, _ in visual_items) else _pick_viz_style(concept_content)
    bullet_source = visual_items or [("", sentence) for sentence in sentences[:4]]
    bullet_points = "\n".join(
        f"- {label}: {body}" if label else f"- {body}"
        for label, body in bullet_source[:4]
        if body
    ) or "- Explain the core concept clearly"
    brief = (
        "Visualization brief:\n"
        f"Title: {title[:72]}\n"
        f"Diagram style: {style}\n"
        "Key points to visualize:\n"
        f"{bullet_points}"
    )
    return [
        {"role": "user", "content": user_message.strip() or "Explain this concept visually."},
        {"role": "assistant", "content": brief},
    ]


def build_manual_agent_messages(agent_key: str, user_message: str, history: list[dict]) -> list[dict]:
    prompt = user_message.strip()

    if agent_key == "concept":
        return [
            *history,
            {
                "role": "user",
                "content": (
                    f"Explain this as an agent design concept for beginners: {prompt}\n\n"
                    "Focus on how it could work as a single-agent system or a multi-agent system. "
                    "Do not provide code. Explain the architecture, flow, and responsibilities."
                ),
            },
        ]

    if agent_key == "code":
        return [
            *history,
            {
                "role": "user",
                "content": (
                    f"For this idea, provide pseudocode first: {prompt}\n\n"
                    "Keep code or pseudocode as the primary output. Prefer pseudocode over real code unless the user explicitly asks for implementation code."
                ),
            },
        ]

    if agent_key == "viz":
        return [
            {"role": "user", "content": prompt or "Explain this agent idea visually."},
            {
                "role": "assistant",
                "content": (
                    "Visualization brief:\n"
                    f"Title: {prompt[:72] or 'Agent workflow'}\n"
                    "Diagram style: workflow\n"
                    "Key points to visualize:\n"
                    f"- Show the workflow for: {prompt}\n"
                    "- Focus on user input, decision steps, tool use or reasoning steps, and final output\n"
                    "- Make it understandable to a beginner\n"
                    "- Prefer a flowchart-style explanation"
                ),
            },
        ]

    return [
        *history,
        {"role": "user", "content": prompt},
    ]


async def run_agent(agent_key: str, messages: list[dict]) -> AgentResponse:
    """Run a single agent asynchronously in a thread pool."""
    agent = AGENTS.get(agent_key)
    if not agent:
        return AgentResponse(agent=agent_key, content=f"Unknown agent: {agent_key}")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: agent.run(messages)
    )

    is_viz = agent_key == "viz"
    return AgentResponse(agent=agent_key, content=result, is_viz=is_viz)


def get_routing(user_message: str, forced_agents: list[str] | None = None) -> dict:
    if forced_agents:
        agents = normalize_agents(forced_agents, enforce_pairs=False)
        return {
            "agents": agents,
            "reason": "Manual agent selection override",
            "topic": "manual selection",
            "manual_selection": True,
        }

    history = session.get_recent_history(turns=6)
    print(f"\n[Orchestrator] Routing: '{user_message[:60]}...'")
    routing = orchestrator.route(user_message, history)
    agents_to_fire = normalize_agents(routing.get("agents", ["qa"]))
    routing["agents"] = agents_to_fire
    print(f"[Orchestrator] → {agents_to_fire} | Reason: {routing.get('reason','')}")
    return routing


async def run_pipeline(user_message: str, routing: dict | None = None) -> ChatResponse:
    """
    The full agent pipeline:
    1. Orchestrator routes the message
    2. concept runs first so viz can support the explanation
    3. viz runs from the concept output while other agents continue
    4. Results returned in a sensible display order
    """
    # Add user message to shared session memory
    session.add_user_message(user_message)
    history = session.get_recent_history(turns=6)

    # Step 1: Orchestrate
    routing = routing or get_routing(user_message)
    agents_to_fire = normalize_agents(
        routing.get("agents", ["qa"]),
        enforce_pairs=not routing.get("manual_selection", False),
    )
    topic = routing.get("topic", "")
    routing["agents"] = agents_to_fire
    session.add_topic(topic)
    session.record_agents_fired(len(agents_to_fire))

    concept_requested = "concept" in agents_to_fire
    viz_requested = "viz" in agents_to_fire
    other_keys = [a for a in agents_to_fire if a not in ("concept", "viz")]
    manual_selection = routing.get("manual_selection", False)

    responses_by_agent: dict[str, AgentResponse] = {}

    concept_task = None
    if concept_requested:
        print("[Pipeline] Firing concept first so viz can support it")
        concept_messages = build_manual_agent_messages("concept", user_message, history) if manual_selection else history
        concept_task = asyncio.create_task(run_agent("concept", concept_messages))

    other_tasks = []
    if other_keys:
        print(f"[Pipeline] Firing concurrently: {other_keys}")
        other_tasks = [
            asyncio.create_task(
                run_agent(
                    agent_key,
                    build_manual_agent_messages(agent_key, user_message, history) if manual_selection else history,
                )
            )
            for agent_key in other_keys
        ]

    concept_response: AgentResponse | None = None
    if concept_task is not None:
        concept_response = await concept_task
        responses_by_agent["concept"] = concept_response

    if viz_requested:
        viz_messages = [{"role": "user", "content": user_message}]
        if manual_selection:
            viz_messages = build_manual_agent_messages("viz", user_message, history)
        elif concept_response is not None and concept_response.content.strip():
            viz_messages = build_viz_messages(user_message, concept_response.content)
        print("[Pipeline] Firing viz from concept output")
        responses_by_agent["viz"] = await run_agent("viz", viz_messages)

    if other_tasks:
        other_results = await asyncio.gather(*other_tasks)
        for result in other_results:
            responses_by_agent[result.agent] = result

    responses: list[AgentResponse] = []
    for key in ["concept", "viz", *other_keys]:
        response = responses_by_agent.get(key)
        if response is not None:
            responses.append(response)

    # Save combined assistant response to memory
    combined = "\n\n".join(r.content for r in responses if not r.is_viz)
    if combined:
        session.add_assistant_message(combined)

    return ChatResponse(
        routing=routing,
        responses=[r.model_dump() for r in responses],
        stats=session.get_stats(),
    )


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    local_url = f"http://localhost:{SERVER_PORT}"
    print("\n🤖  Agent 101 Demo System")
    print("─" * 40)
    print(f"   Agents loaded: {len(AGENTS)}")
    print(f"   Azure deployment: {AZURE_OPENAI_CONFIG['deployment']}")
    print(f"   UI: {local_url}")
    print("   Browser auto-open: enabled" if should_auto_open_browser() else "   Browser auto-open: disabled")
    print("─" * 40)
    if should_auto_open_browser():
        await asyncio.sleep(1)
        try:
            if not webbrowser.open(local_url):
                print(f"   Could not open a browser automatically. Open {local_url} manually.")
        except Exception as exc:
            print(f"   Could not open a browser automatically: {exc}")
    yield

app = FastAPI(title="Agent 101 Demo", lifespan=lifespan)

# Serve static files (the HTML UI)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.post("/api/route")
async def route(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        routing = get_routing(request.message.strip(), forced_agents=request.forced_agents)
        return RouteResponse(routing=routing)
    except Exception as e:
        print(f"[Error] Routing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        result = await run_pipeline(request.message.strip(), routing=request.routing)
        return result
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scribe")
async def scribe():
    """Generate the cheat sheet from full session history."""
    history_text = session.get_full_history_text()
    if not history_text:
        history_text = "No conversation yet — generate a general Agent 101 cheat sheet."

    print("\n[Scribe] Generating cheat sheet from session history...")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: AGENTS["scribe"].generate(history_text)
    )
    session.record_agents_fired(1)
    return {"content": result, "stats": session.get_stats()}


@app.get("/api/stats")
async def stats():
    return session.get_stats()


@app.post("/api/clear")
async def clear():
    session.clear()
    print("\n[Session] Cleared.")
    return {"status": "cleared"}


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVER_PORT,
        reload=False,
        log_level="warning",  # keep terminal clean during demo
    )
