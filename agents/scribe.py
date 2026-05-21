# agents/scribe.py
# The Scribe reads the full session history and generates a take-home cheat sheet.
# Fired on demand when the user wants a summary.

import re

from .base import BaseAgent

SYSTEM = """You are the Scribe agent. Generate a clean cheat sheet summarising the Agent 101 session.

Format as a structured markdown document with only the sections that were actually covered.

Required goals:
- Summarise the session clearly
- Pull out the strongest best practices
- Keep it scannable and worth keeping

Rules:
- Do NOT include sections for topics that were not covered
- Do NOT write placeholder lines like "not covered in this session"
- Prefer short sections over exhaustive ones
- Include best practices whenever the conversation supports them
- If code patterns were discussed, include a short section for them
- Use bullets and short snippets where helpful

Suggested section shapes when relevant:
# Agent 101 — Cheat Sheet
## Session Summary
## Core Concepts Covered
## Code Patterns
## Best Practices
## Key Takeaways"""


class ScribeAgent(BaseAgent):
    name = "scribe"
    system_prompt = SYSTEM

    def _clean_sections(self, text: str) -> str:
        sections = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
        if not sections:
            return text

        cleaned: list[str] = []
        for section in sections:
            lower = section.lower()
            if "not covered" in lower or "was not covered" in lower:
                continue
            cleaned.append(section.strip())

        return "\n\n".join(part for part in cleaned if part).strip()

    def generate(self, full_history_text: str) -> str:
        """
        Special method for Scribe — takes the full session text rather than
        the standard message history format.
        """
        prompt = (
            f"Here is the full Agent 101 learning session transcript:\n\n"
            f"{full_history_text}\n\n"
            f"Generate the cheat sheet now, covering only what was actually discussed."
        )
        result = self.run(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        return self._clean_sections(result)
