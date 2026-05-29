# main.py
# Entry point for the Agent 101 learning system.
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
# ---- Optional RAG / Visual retrieval imports ----
# Keep these global so startup/lifespan and pipeline code can reference them safely.
try:
    from rag import rag_store
except Exception as exc:
    rag_store = None
    print(f"[RAG] disabled: {exc}")

try:
    from visual_store import visual_store
except Exception as exc:
    visual_store = None
    print(f"[Visual] disabled: {exc}")
# -----------------------------------------------


from agents.base import get_azure_openai_config

try:
    AZURE_OPENAI_CONFIG = get_azure_openai_config()
except RuntimeError as exc:
    print(f"\n❌  {exc}")
    print("    Update .env with your Azure OpenAI endpoint, deployment, and API key.\n")
    sys.exit(1)

from memory import session
from agents.orchestrator import OrchestratorAgent
from agents.concept import ConceptAgent
from agents.viz import VizAgent
from agents.code_agent import CodeAgent
from agents.best_practices import BestPracticesAgent
from agents.mcp_agent import McpAgent
from agents.deploy import DeployAgent
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
    "concept": ConceptAgent(),
    "viz":     VizAgent(),
    "code":    CodeAgent(),
    "bp":      BestPracticesAgent(),
    "mcp":     McpAgent(),
    "deploy":  DeployAgent(),
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
    """Keep routing predictable without forcing concept/viz for targeted questions."""
    normalized: list[str] = []
    for agent in agents or ["concept"]:
        if agent not in normalized:
            normalized.append(agent)

    if enforce_pairs and "concept" in normalized and "viz" not in normalized:
        concept_idx = normalized.index("concept")
        normalized.insert(concept_idx + 1, "viz")

    return normalized or ["concept"]

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
        "real world example": "Example",
        "example": "Example",
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


def _prefers_workflow_visual(user_message: str) -> bool:
    lower = _plain_text(user_message).lower()
    workflow_terms = (
        "how would",
        "how does",
        "how do",
        "how it works",
        "how it would work",
        "workflow",
        "flow",
        "pipeline",
        "step by step",
        "steps",
        "process",
    )
    agent_terms = ("agent", "assistant", "bot", "calculator")
    return any(term in lower for term in workflow_terms) or (
        any(term in lower for term in agent_terms) and "work" in lower
    )


def _clean_visual_text(text: str) -> str:
    cleaned = _plain_text(text)
    cleaned = re.sub(
        r"^(definition|example|analogy|mechanics|why it matters)\s*[:\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" -:;,.")


def _visual_subject(user_message: str) -> str:
    subject = _plain_text(user_message)
    subject = re.sub(r"^(how would|how does|how do|explain|show me|what is|what's)\s+", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\b(may|can|could|would|should)\s+\b", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\b(work|works|working)\b.*$", "", subject, flags=re.IGNORECASE).strip(" ?.,")
    subject_match = re.search(
        r"\b((?:an?\s+)?[a-z0-9\s-]{0,40}?(?:agent|assistant|bot)|calculator)\b",
        subject,
        flags=re.IGNORECASE,
    )
    if subject_match:
        subject = subject_match.group(1).strip()
    return subject


def _visual_title(user_message: str, concept_content: str, style: str) -> str:
    sentences = _concept_sentences(concept_content, limit=1)
    if sentences:
        title = _clean_visual_text(sentences[0])[:72]
        if title:
            if style == "workflow":
                return f"{title} — Workflow"
            return title
    subject = _visual_subject(user_message)
    if style == "workflow":
        return f"{subject.title()} Workflow" if subject else "How It Works"
    return subject.title() if subject else "Agent Concept"


def _workflow_visual_points(user_message: str, concept_content: str) -> list[tuple[str, str]]:
    lower = _plain_text(user_message).lower()
    if any(term in lower for term in ("agent", "assistant", "bot", "calculator")):
        subject = _visual_subject(user_message) or "the agent"
        return [
            ("", f"The user gives {subject} a goal or problem to solve."),
            ("", f"It interprets the request and extracts the important inputs or constraints."),
            ("", f"It runs the right tool or logic step to do the work reliably."),
            ("", f"It returns the result clearly and can explain or verify the answer."),
        ]

    sentences = _concept_sentences(concept_content, limit=4)
    return [("", sentence) for sentence in sentences[:4]]


def build_viz_messages(user_message: str, concept_content: str) -> list[dict]:
    """Give the Visual agent a focused brief instead of the whole conversation."""
    visual_items = _concept_visual_items(concept_content)
    sentences = _concept_sentences(concept_content)
    base_style = _pick_viz_style(f"{user_message}\n{concept_content}")
    if _prefers_workflow_visual(user_message):
        style = "workflow"
    elif any(label for label, _ in visual_items) and base_style in {"sequence", "workflow"}:
        style = "concept_grid"
    else:
        style = base_style
    title = _visual_title(user_message, concept_content, style)
    if style == "workflow":
        bullet_source = _workflow_visual_points(user_message, concept_content)
    else:
        bullet_source = visual_items or [("", sentence) for sentence in sentences]
    bullet_points = "\n".join(
        f"- {label}: {_clean_visual_text(body)}" if label else f"- {_clean_visual_text(body)}"
        for label, body in bullet_source
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


def build_manual_viz_messages(user_message: str) -> list[dict]:
    prompt = user_message.strip() or "Explain this agent idea visually."
    lower = _plain_text(prompt).lower()
    style = _pick_viz_style(prompt)
    if _prefers_workflow_visual(prompt) or any(term in lower for term in ("agent", "assistant", "bot", "calculator")):
        style = "workflow"
    title = _visual_title(prompt, prompt, style)

    if style == "workflow":
        bullet_source = _workflow_visual_points(prompt, prompt)
    else:
        bullet_source = [("", _clean_visual_text(sentence)) for sentence in _concept_sentences(prompt, limit=8)]

    bullet_points = "\n".join(
        f"- {label}: {_clean_visual_text(body)}" if label else f"- {_clean_visual_text(body)}"
        for label, body in bullet_source
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
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": brief},
    ]


AGENT_PERSPECTIVE_HINTS = {
    "concept": (
        "Answer this question as a concept explainer for people learning about AI agents. "
        "Start with a clear 1-2 sentence definition or direct answer, then explain the idea — "
        "how it works, why it matters, and a concrete example. Do not provide code. "
        "Max 300 words. Always finish every sentence and section you begin — "
        "if running long, write one closing sentence and stop rather than starting a new section."
    ),
    "code": (
        "Answer this question from a code and implementation perspective. "
        "Start with a 1-2 sentence overview, then show pseudocode or working code as the main output. "
        "Keep code under 40 lines and the surrounding text under 150 words. "
        "Always finish every sentence and every code block you begin — never cut off mid-function or mid-comment."
    ),
    "bp": (
        "Answer this question from a best practices perspective. "
        "Give 3-4 DO/DON'T pairs followed by one Golden Rule. Max 300 words. "
        "Always complete every pair and the Golden Rule fully — never leave a line half-written."
    ),
    "mcp": (
        "Answer this question from a tools, APIs, and integrations perspective. "
        "Start with a 1-2 sentence overview, then focus on tool calling, MCP, function schemas, or external APIs. "
        "Max 300 words. Always finish every sentence and section you begin — "
        "if running long, write one closing sentence and stop rather than starting a new heading."
    ),
    "deploy": (
        "Answer this question from a production and operations perspective. "
        "Start with a 1-2 sentence framing, then cover production concerns: "
        "evals, observability, hosting, reliability, cost, safety, and scaling. "
        "Max 300 words. Always finish every sentence and section — "
        "if running long, write one closing sentence and stop rather than starting a new checklist."
    ),
    "scribe": (
        "Summarise this topic clearly and concisely as a structured reference. "
        "Use headers, bullet points, and short explanations. Focus on what a learner would want to remember. "
        "Always complete every section you begin — never leave a heading without content."
    ),
}


def build_manual_agent_messages(agent_key: str, user_message: str, history: list[dict]) -> list[dict]:
    prompt = user_message.strip()

    if agent_key == "viz":
        return build_manual_viz_messages(prompt)

    hint = AGENT_PERSPECTIVE_HINTS.get(agent_key, "")
    if hint:
        framed_content = f"{hint}\n\nQuestion: {prompt}"
    else:
        framed_content = prompt

    return [
        *history,
        {"role": "user", "content": framed_content},
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




def _insert_rag_context(messages: list[dict], rag_context: str) -> list[dict]:
    """Insert retrieved context immediately before the latest user message."""
    if not rag_context:
        return messages
    context_message = {
        "role": "user",
        "content": (
            "Retrieved knowledge-base context for the next answer. "
            "Use it first, cite sources inline using [1], [2], etc., and include a short "
            "Sources Used section if you rely on it.\n\n"
            f"{rag_context}"
        ),
    }
    if messages and messages[-1].get("role") == "user":
        return [*messages[:-1], context_message, messages[-1]]
    return [*messages, context_message]


def build_concept_messages(user_message: str, history: list[dict], manual_selection: bool = False) -> list[dict]:
    """Build Concept messages with RAG context when available."""
    messages = build_manual_agent_messages("concept", user_message, history) if manual_selection else list(history)
    rag_context = ""
    if rag_store is not None:
        try:
            rag_context = rag_store.build_context_message(user_message)
        except Exception as exc:
            print(f"[RAG] Retrieval skipped: {exc}")
    else:
        print("[RAG] Retrieval skipped: rag_store unavailable")
    if rag_context:
        return _insert_rag_context(messages, rag_context)
    print("[RAG] Concept agent running without retrieved context")
    return messages



def get_last_rag_sources() -> list[dict]:
    """Return sources from the latest RAG retrieval, if the current rag.py exposes them."""
    if rag_store is None:
        return []
    try:
        if hasattr(rag_store, "get_last_sources"):
            sources = rag_store.get_last_sources()
        else:
            sources = getattr(rag_store, "last_sources", [])
        return list(sources or [])
    except Exception as exc:
        print(f"[RAG] Could not read last sources: {exc}")
        return []


def append_sources_used_if_missing(content: str, sources: list[dict]) -> str:
    """Deterministically append sources so attribution appears even if the model forgets."""
    if not content or not sources:
        return content
    if "sources used" in content.lower():
        return content

    lines = []
    seen = set()
    for source in sources:
        number = source.get("number") or len(lines) + 1
        name = str(source.get("source") or "Unknown source").strip()
        content_type = str(source.get("content_type") or "document").strip()
        category = str(source.get("category") or "general").strip()
        key = (number, name)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{number}] {name} — {content_type}, {category}")

    if not lines:
        return content
    return content.rstrip() + "\n\n**Sources Used**\n" + "\n".join(f"- {line}" for line in lines)

def get_visual_response(user_message: str, routing: dict, concept_response: AgentResponse | None, manual_selection: bool, history: list[dict]) -> AgentResponse | None:
    """Prefer trusted library visuals; fall back to VizAgent template generation."""
    if visual_store is not None:
        try:
            # asset = visual_store.search(
            #     query=user_message,
            #     topic=str(routing.get("topic", "")),
            #     difficulty=str(routing.get("difficulty", "")),
            # )
            asset = visual_store.search(user_message)
            if asset:
                print(
                    f"[Visual] Using library visual: "
                    f"id={asset.id} score={asset.score:.3f} file={asset.file}"
                )

                return AgentResponse(
                    agent="viz",
                    content=asset.content,
                    is_viz=True,
                )
            print("[Visual] No library visual matched; falling back to Viz agent")
        except Exception as exc:
            print(f"[Visual] Library lookup failed: {exc}; falling back to Viz agent")
    return None

def _routing_steps(raw_steps: object) -> list[str]:
    if isinstance(raw_steps, str):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        raw_steps = []

    steps: list[str] = []
    for step in raw_steps:
        cleaned = _plain_text(str(step))
        if cleaned and cleaned not in steps:
            steps.append(cleaned[:160])

    return steps[:4]


def _routing_thinking(raw_thinking: object, reason: str, topic: str, agents: list[str]) -> str:
    if isinstance(raw_thinking, list):
        parts = [_plain_text(str(part)) for part in raw_thinking if _plain_text(str(part))]
        thinking = "\n\n".join(parts)
    else:
        thinking = str(raw_thinking or "")
    thinking = thinking.strip()
    if thinking:
        return thinking[:1600]
    return (
        f"The request appears to be about {topic or 'agent fundamentals'}, so the orchestrator is leaning on "
        f"{', '.join(agents)}. {reason}"
    )


def finalize_routing(routing: dict | None, manual_selection: bool = False) -> dict:
    routing = dict(routing or {})

    # Short-circuit for off-topic: don't normalize agents or build a routing message
    if routing.get("off_topic"):
        routing["agents"] = []
        routing.setdefault("reason", "Question is not related to AI agents or agentic workflows")
        routing.setdefault("topic", "")
        routing["thinking"] = ""
        routing["decision_steps"] = []
        return routing

    routing["agents"] = normalize_agents(
        routing.get("agents", ["concept"]),
        enforce_pairs=not manual_selection,
    )
    routing["reason"] = _plain_text(str(routing.get("reason", "")))
    routing["topic"] = _plain_text(str(routing.get("topic", "")))

    if not routing["reason"]:
        routing["reason"] = (
            "Manual agent selection override"
            if manual_selection
            else "Routing fallback — starting with the concept explanation"
        )
    if not routing["topic"]:
        routing["topic"] = "manual selection" if manual_selection else "agent fundamentals"
    routing["thinking"] = _routing_thinking(
        routing.get("thinking") or routing.get("rationale") or routing.get("analysis"),
        routing["reason"],
        routing["topic"],
        routing["agents"],
    )

    steps = _routing_steps(
        routing.get("decision_steps")
        or routing.get("routing_steps")
        or routing.get("chain_of_thought")
    )
    if not steps:
        if manual_selection:
            steps = [
                "Manual agent selection is active in the UI.",
                f"Running the chosen agents directly: {', '.join(routing['agents'])}.",
            ]
        else:
            steps = [
                f"Detected topic: {routing['topic']}.",
                f"Selected agents: {', '.join(routing['agents'])}.",
                routing["reason"],
            ]

    routing["decision_steps"] = steps
    if manual_selection:
        routing["manual_selection"] = True
    return routing


def get_routing(user_message: str, forced_agents: list[str] | None = None) -> dict:
    if forced_agents:
        return finalize_routing({
            "agents": forced_agents,
            "reason": "Manual agent selection override",
            "topic": "manual selection",
        }, manual_selection=True)

    history = session.get_recent_history(turns=6)
    print(f"\n[Orchestrator] Routing: '{user_message[:60]}...'")
    routing = finalize_routing(orchestrator.route(user_message, history))
    print(f"[Orchestrator] → {routing['agents']} | Reason: {routing.get('reason','')}")
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

    # Guardrail: always check topic relevance first — covers both auto and manual-override paths
    loop = asyncio.get_event_loop()
    if await loop.run_in_executor(None, lambda: orchestrator.is_off_topic(user_message)):
        off_topic_routing = {
            "agents": [],
            "reason": "Question is not related to AI agents or agentic workflows",
            "topic": "",
            "off_topic": True,
        }
        off_topic_msg = (
            "That question falls outside what I can help with here.\n\n"
            "**Agent 101 Learning Studio** is focused exclusively on **AI agents and agentic workflows** — "
            "topics like how agents are built, how they reason and plan, tool calling, memory, "
            "orchestration patterns, RAG, multi-agent systems, and deploying agents to production.\n\n"
            "Try asking something like:\n"
            "- *What is an AI agent?*\n"
            "- *How does tool calling work?*\n"
            "- *What's the difference between RAG and fine-tuning?*\n"
            "- *How do I make an agent reliable in production?*"
        )
        return ChatResponse(
            routing=off_topic_routing,
            responses=[AgentResponse(agent="system", content=off_topic_msg)],
            stats={"off_topic": True},
        )

    # Step 1: Orchestrate
    routing = routing or get_routing(user_message)
    routing = finalize_routing(routing, manual_selection=routing.get("manual_selection", False))

    # Secondary guardrail: catch off_topic flag from orchestrator routing (auto-routing path)
    if routing.get("off_topic"):
        off_topic_msg = (
            "That question falls outside what I can help with here.\n\n"
            "**Agent 101 Learning Studio** is focused exclusively on **AI agents and agentic workflows** — "
            "topics like how agents are built, how they reason and plan, tool calling, memory, "
            "orchestration patterns, RAG, multi-agent systems, and deploying agents to production.\n\n"
            "Try asking something like:\n"
            "- *What is an AI agent?*\n"
            "- *How does tool calling work?*\n"
            "- *What's the difference between RAG and fine-tuning?*\n"
            "- *How do I make an agent reliable in production?*"
        )
        return ChatResponse(
            routing=routing,
            responses=[AgentResponse(agent="system", content=off_topic_msg)],
            stats={"off_topic": True},
        )

    agents_to_fire = routing["agents"]
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
        concept_messages = build_concept_messages(user_message, history, manual_selection)
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
        rag_sources = get_last_rag_sources()
        if rag_sources:
            concept_response.content = append_sources_used_if_missing(concept_response.content, rag_sources)
        responses_by_agent["concept"] = concept_response

    if viz_requested:
        library_visual = get_visual_response(user_message, routing, concept_response, manual_selection, history)
        if library_visual is not None:
            responses_by_agent["viz"] = library_visual
        else:
            viz_messages = [{"role": "user", "content": user_message}]
            if manual_selection:
                viz_messages = build_manual_viz_messages(user_message, history)
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
    print("\n🤖  Agent 101 Learning Studio")
    print("─" * 40)
    print(f"   Agents loaded: {len(AGENTS)}")
    print(f"   Azure deployment: {AZURE_OPENAI_CONFIG['deployment']}")
    print(f"   UI: {local_url}")
    if rag_store is not None:
        try:
            rag_store.ensure_ready()
            if getattr(rag_store, "ready", False):
                print(f"   RAG: ready ({len(getattr(rag_store, 'chunks', []))} chunks)")
            else:
                print("   RAG: enabled but index not ready")
        except Exception as exc:
            print(f"   RAG index unavailable: {exc}")
    else:
        print("   RAG: disabled")

    if visual_store is not None:
        try:
            count = visual_store.count()
            print(f"   Visual library: ready ({count} visuals)")
        except Exception as exc:
            print(f"   Visual library unavailable: {exc}")
    else:
        print("   Visual library: disabled")

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

app = FastAPI(title="Agent 101 Learning Studio", lifespan=lifespan)

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
        log_level="warning",  # keep terminal output focused
    )
