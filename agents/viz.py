# agents/viz.py
# Safe fallback visual agent. The main pipeline should prefer visual_store assets first.
# If no trusted visual matches, this returns a readable, wrapped SVG template.

from __future__ import annotations

import html
import re
from textwrap import wrap

from .base import BaseAgent

SYSTEM = """You are the Visual agent. Prefer simple, accurate diagrams over creative diagrams.
Only visualize facts present in the supplied visualization brief. Output only raw SVG."""

BG = "#031114"
PANEL = "rgba(29,244,244,.08)"
PRIMARY = "#1df4f4"
SECONDARY = "rgba(162,243,243,.55)"
TEXT = "#ffffff"
MUTED = "rgba(255,255,255,.55)"
FONT = "Montserrat, Arial, sans-serif"


class VizAgent(BaseAgent):
    name = "viz"
    system_prompt = SYSTEM

    def run(self, messages: list[dict], max_tokens: int = 1024) -> str:
        source = self._source_text(messages)
        brief = self._extract_brief(source)
        return self._safe_svg(brief)

    def _source_text(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            content = str(message.get("content", "")).strip()
            if content.startswith("Visualization brief:"):
                return content
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", "")).strip()
        return "Agent concept"

    def _plain(self, text: str) -> str:
        text = re.sub(r"[*`#_>-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" -:;,.\n\t")

    def _extract_brief(self, source_text: str) -> dict:
        title_match = re.search(r"^Title:\s*(.+)$", source_text, re.MULTILINE)
        style_match = re.search(r"^Diagram style:\s*(.+)$", source_text, re.MULTILINE)
        point_matches = re.findall(r"^-\s*(.+)$", source_text, re.MULTILINE)

        if not point_matches:
            point_matches = re.split(r"(?<=[.!?])\s+", self._plain(source_text))[:4]

        cleaned_points = []
        for p in point_matches:
            cleaned = self._plain(p)
            cleaned = re.sub(r"^(definition|example|analogy|mechanics|why it matters)\s*[:\-]\s*", "", cleaned, flags=re.I)
            if cleaned and cleaned.lower() not in {"plain language definition", "workflow"}:
                cleaned_points.append(cleaned)

        return {
            "title": self._plain(title_match.group(1))[:72] if title_match else "Agent concept",
            "style": self._plain(style_match.group(1)).lower() if style_match else "sequence",
            "points": cleaned_points[:5] or ["User request", "Agent reasoning", "Tool or action", "Result"],
        }

    def _label(self, text: str, fallback: str) -> tuple[str, str]:
        cleaned = self._plain(text)
        if ":" in cleaned:
            left, right = cleaned.split(":", 1)
            if 2 <= len(left.strip()) <= 28 and len(right.strip()) > 4:
                return left.strip(), right.strip()
        words = cleaned.split()
        title = " ".join(words[:4]).strip() or fallback
        body = " ".join(words[4:]).strip()
        return title[:34], body or cleaned

    def _text(self, x: int, y: int, text: str, width: int, size: int = 10, color: str = TEXT, anchor: str = "start", max_lines: int = 3) -> str:
        # Approximate chars-per-line from pixel width and font size.
        chars = max(8, int(width / (size * 0.58)))
        lines = wrap(self._plain(text), width=chars)[:max_lines]
        if not lines:
            return ""
        if len(wrap(self._plain(text), width=chars)) > max_lines:
            lines[-1] = lines[-1].rstrip(".,;: ") + "…"
        tspans = []
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else size + 3
            tspans.append(f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>')
        return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-family="{FONT}" text-anchor="{anchor}">{"".join(tspans)}</text>'

    def _sequence_svg(self, title: str, points: list[str]) -> str:
        points = points[:4]
        while len(points) < 3:
            points.append(["User request", "Agent reasoning", "Result"][len(points)])

        card_w = 96
        card_h = 92
        xs = [28, 142, 256, 370][:len(points)]
        y = 96
        parts = [
            '<svg viewBox="0 0 480 300" xmlns="http://www.w3.org/2000/svg">',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="rgba(162,243,243,.65)"/></marker></defs>',
            f'<rect width="480" height="300" fill="{BG}"/>',
            self._text(24, 34, title, 430, size=15, color=TEXT, max_lines=2),
        ]

        for i, point in enumerate(points):
            x = xs[i]
            label, body = self._label(point, f"Step {i+1}")
            parts.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" fill="{PANEL}" stroke="{PRIMARY}" stroke-width="1.5"/>')
            parts.append(self._text(x + 10, y + 24, label, card_w - 20, size=10, color=PRIMARY, max_lines=2))
            parts.append(self._text(x + 10, y + 52, body, card_w - 20, size=8, color=TEXT, max_lines=3))
            if i < len(points) - 1:
                parts.append(f'<line x1="{x + card_w + 6}" y1="{y + card_h/2}" x2="{xs[i+1] - 8}" y2="{y + card_h/2}" stroke="{SECONDARY}" stroke-width="1.2" marker-end="url(#arrow)"/>')

        parts.append(self._text(24, 270, "Fallback visual generated from retrieved/agent brief only.", 430, size=9, color=MUTED, max_lines=1))
        parts.append('</svg>')
        return "".join(parts)

    def _safe_svg(self, brief: dict) -> str:
        title = brief.get("title") or "Agent concept"
        points = [str(p) for p in brief.get("points") or []]
        return self._sequence_svg(title, points)
