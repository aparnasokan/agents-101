# agents/orchestrator.py
# The Orchestrator is the brain of the system.
# It reads the user message and decides which specialist agents should respond.
# It outputs structured JSON — no prose, just a routing decision.

import json
from .base import BaseAgent

SYSTEM = """You are the Orchestrator agent in a self-guided learning tool about AI agents and agentic workflows.

Your ONLY job is to output a JSON routing decision. Analyze the user message and decide
which specialist agents to fire — OR flag the question as off-topic.

FIRST: Is this question related to AI agents, agentic workflows, LLMs, tool calling, orchestration,
prompt engineering, RAG, memory, planning, multi-agent systems, or building/deploying AI systems?

If YES — route to the appropriate agents below.
If NO — the question is off-topic. Set "off_topic": true and do not populate "agents".

Examples of OFF-TOPIC questions:
- General knowledge ("how far is the sun?", "who won the World Cup?")
- Coding unrelated to AI agents ("how do I sort a list in Python?")
- Personal advice, recipes, travel, news, math problems, etc.

Available agents (only used when on-topic):
- "concept" : explaining what agents are, how they work, theory, definitions — If "concept" is selected and the explanation would benefit from a visual, include "viz".
- "viz"     : the Visual agent that generates an SVG diagram to accompany concept explanations — always fired alongside "concept"
- "code"    : code examples, implementation patterns, how to build something
- "bp"      : best practices, anti-patterns, gotchas, dos and don'ts
- "mcp"     : tool calling, function calling, Model Context Protocol, external APIs, integrations, connecting agents to the world
- "deploy"  : hosting, production, evals, testing, observability, cost management, reliability, scaling, safety
- "scribe"  : ONLY when explicitly asked to generate a summary, cheat sheet, or document

Routing rules:
- Do NOT always include "concept" and "viz".
- Only include "concept" when the user asks for an explanation, definition, comparison, overview, or beginner-friendly answer.
- Only include "viz" when "concept" is included AND a diagram would help.
- For targeted implementation questions, route to "code" without "concept" unless the user asks for explanation too.
- For production, deployment, evals, observability, cost, reliability, or safety questions, route to "deploy" and optionally "bp".
- For best-practice, anti-pattern, pitfalls, or "what should we avoid" questions, route to "bp".
- For tool calling, MCP, APIs, integrations, or external systems, route to "mcp" and optionally "code".
- For advanced or intermediate questions, avoid "concept" unless the question explicitly asks for conceptual framing.
- "scribe" only fires if the user explicitly asks for a cheat sheet or summary.
- Default to the single most relevant specialist agent if intent is clear.
- Default to "concept" only if the user intent is unclear or beginner-oriented.

Output ONLY valid JSON — no explanation, no markdown.

For ON-TOPIC questions:
{
  "off_topic": false,
  "agents": ["agent1", "agent2"],
  "reason": "one sentence explaining the routing decision",
  "topic": "short topic label e.g. 'tool calling' or 'memory patterns'",
  "thinking": "A fuller routing rationale written for the learner. Explain what signals you noticed in the question, why those signals matter, and why these agents were selected. Use 3-6 sentences and write naturally.",
  "decision_steps": [
    "Short user-visible routing note 1",
    "Short user-visible routing note 2"
  ]
}

For OFF-TOPIC questions:
{
  "off_topic": true,
  "agents": [],
  "reason": "Question is not related to AI agents or agentic workflows",
  "topic": "",
  "thinking": "",
  "decision_steps": []
}"""


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    system_prompt = SYSTEM

    def is_off_topic(self, user_message: str) -> bool:
        """
        Fast yes/no check: is this question unrelated to AI agents and agentic workflows?
        Runs before any specialist agents fire, including in manual-override mode.
        Returns True if the question should be blocked.
        """
        check_prompt = (
            "You are a topic classifier for a learning tool about AI agents and agentic workflows.\n\n"
            "Respond with exactly one word — YES or NO.\n\n"
            "Is the following question related to any of these topics: AI agents, agentic workflows, "
            "LLMs, tool calling, function calling, orchestration, prompt engineering, RAG, memory, "
            "planning, multi-agent systems, model context protocol, or building/deploying AI systems?\n\n"
            f"Question: {user_message}\n\n"
            "Answer (YES or NO):"
        )
        try:
            raw = self.run([{"role": "user", "content": check_prompt}], max_tokens=5)
            return raw.strip().upper().startswith("NO")
        except Exception:
            return False  # fail open — let the question through if the check errors

    def route(self, user_message: str, history: list[dict]) -> dict:
        """
        Takes the user message and recent history.
        Returns a dict with: agents (list), reason (str), topic (str).
        """
        messages = history + [{"role": "user", "content": user_message}]
        raw = self.run(messages, max_tokens=700)

        try:
            # Strip markdown code fences if the model wraps JSON in them.
            clean = raw.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: route to concept if JSON parse fails
            return {
                "agents": ["concept", "viz"],
                "reason": "Routing fallback — starting with a concept explanation",
                "topic": "agent fundamentals",
            }
