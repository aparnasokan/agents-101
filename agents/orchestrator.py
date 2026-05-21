# agents/orchestrator.py
# The Orchestrator is the brain of the system.
# It reads the user message and decides which specialist agents should respond.
# It outputs structured JSON — no prose, just a routing decision.

import json
from .base import BaseAgent

SYSTEM = """You are the Orchestrator agent in a self-guided learning tool about AI agents and agentic workflows.

Your ONLY job is to output a JSON routing decision. Analyze the user message and decide
which specialist agents to fire.

Available agents:
- "concept" : explaining what agents are, how they work, theory, definitions — ALWAYS pair with "viz"
- "viz"     : the Visual agent that generates an SVG diagram to accompany concept explanations — always fired alongside "concept"
- "code"    : code examples, implementation patterns, how to build something
- "bp"      : best practices, anti-patterns, gotchas, dos and don'ts
- "mcp"     : Model Context Protocol explanations, MCP architecture, tool connections
- "deploy"  : hosting, production, monitoring, scaling, cost, deployment patterns
- "scribe"  : ONLY when explicitly asked to generate a summary, cheat sheet, or document

Routing rules:
- "concept" and "viz" MUST always be fired together
- You may fire multiple agents e.g. ["concept", "viz", "code"]
- "scribe" only fires if the user explicitly asks for a cheat sheet or summary
- Default to "concept" if the intent is unclear

Output ONLY valid JSON — no explanation, no markdown:
{
  "agents": ["agent1", "agent2"],
  "reason": "one sentence explaining the routing decision",
  "topic": "short topic label e.g. 'tool calling' or 'memory patterns'",
  "decision_steps": [
    "Short user-visible routing note 1",
    "Short user-visible routing note 2"
  ]
}"""


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    system_prompt = SYSTEM

    def route(self, user_message: str, history: list[dict]) -> dict:
        """
        Takes the user message and recent history.
        Returns a dict with: agents (list), reason (str), topic (str).
        """
        messages = history + [{"role": "user", "content": user_message}]
        raw = self.run(messages, max_tokens=256)

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
