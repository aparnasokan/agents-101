# agents/viz.py
# Generates SVG diagrams to accompany concept explanations.
# Runs after the Concept agent so it can support that explanation.
# Outputs raw SVG that the UI renders inline.

import html
import re

from .base import BaseAgent

SYSTEM = """You are the Viz agent. Generate a clean SVG diagram that visually supports the Concept agent's explanation.

The conversation will include the original user question and may include an assistant message starting with:
"Visualization brief:"
Use that brief as the primary source for the diagram, especially its title, diagram style, and key points.

Output ONLY a raw SVG element — no explanation, no markdown, no prose around it. Just the SVG.

Requirements:
- viewBox="0 0 480 280"
- Dark background: start with <rect width="480" height="280" fill="#031114"/>
- Colors: primary=#1df4f4, secondary=#a2f3f3, text=#ffffff, panel=#053057
- Node boxes: rx=8, stroke-width=1, use fill with low opacity + matching stroke
- Text: fill=#ffffff, font-size=11, font-family="Montserrat, Arial, sans-serif"
- Arrows: use <line> or <path> with marker-end for direction, stroke=rgba(162,243,243,0.3)
- Add arrowhead marker in <defs>
- Keep it simple: max 7 nodes, clear left-to-right or top-to-bottom flow
- Make the visual explain the concept itself, not the multi-agent system, unless the concept is explicitly about orchestration or routing
- Prefer visual structures like stages, loops, components, inputs/outputs, or before/after states based on the concept explanation
- If the brief suggests `concept_grid`, `workflow`, `sequence`, `cycle`, `comparison`, `layers`, or `hierarchy`, reflect that structure directly in the SVG
- Add a subtle title: font-size=10, fill=rgba(255,255,255,0.3), top-left corner"""

BG = "#031114"
PANEL = "#053057"
PRIMARY = "#1df4f4"
SECONDARY = "#a2f3f3"
TEXT = "#ffffff"
TEXT_SOFT = "rgba(162,243,243,0.68)"
ARROW = "rgba(162,243,243,0.3)"
ARROW_HEAD = "rgba(162,243,243,0.45)"
PANEL_FILL = "rgba(5,48,87,0.32)"
PRIMARY_FILL = "rgba(29,244,244,0.08)"
SECONDARY_FILL = "rgba(162,243,243,0.08)"
TEXT_FILL = "rgba(255,255,255,0.06)"
FONT = "Montserrat, Arial, sans-serif"


class VizAgent(BaseAgent):
    name = "viz"
    system_prompt = SYSTEM

    def run(self, messages: list[dict], max_tokens: int = 1024) -> str:
        return self._fallback_svg(self._source_text(messages))

    def _source_text(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            content = str(message.get("content", "")).strip()
            if content.startswith("Visualization brief:"):
                return content
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", "")).strip()
        return "Agent concept"

    def _extract_brief(self, source_text: str) -> dict[str, object]:
        if not source_text.startswith("Visualization brief:"):
            plain = self._plain_text(source_text) or "Agent concept"
            parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", plain) if part.strip()]
            return {
                "title": (parts[0] if parts else plain)[:42],
                "style": "sequence",
                "points": parts[:4] if parts else [plain],
            }

        title_match = re.search(r"^Title:\s*(.+)$", source_text, re.MULTILINE)
        style_match = re.search(r"^Diagram style:\s*(.+)$", source_text, re.MULTILINE)
        point_matches = re.findall(r"^- (.+)$", source_text, re.MULTILINE)
        title = self._plain_text(title_match.group(1)) if title_match else "Agent concept"
        style = self._plain_text(style_match.group(1)).lower() if style_match else "sequence"
        points = [self._plain_text(point) for point in point_matches if self._plain_text(point)]
        if not points:
            points = [title]
        return {
            "title": title[:42],
            "style": style,
            "points": points[:4],
        }

    def _plain_text(self, text: str) -> str:
        plain = re.sub(r"[*`#_>-]+", " ", text)
        return re.sub(r"\s+", " ", plain).strip()

    def _clean_point(self, text: str) -> str:
        text = re.sub(r"^\s*\d+[\).\s:-]*", "", text)
        text = re.sub(r"^\s*[-•]\s*", "", text)
        return self._plain_text(text)

    def _short_label(self, text: str, fallback: str) -> str:
        cleaned = self._clean_point(text)
        if not cleaned:
            return fallback
        if ":" in cleaned:
            maybe_label, remainder = cleaned.split(":", 1)
            if len(maybe_label.strip()) <= 24 and len(remainder.strip()) >= 8:
                return maybe_label.strip()[:20]
        words = cleaned.split()[:3]
        label = " ".join(words).rstrip(".,:;")
        return label[:20] or fallback

    def _split_point(self, text: str, fallback_label: str) -> tuple[str, str]:
        cleaned = self._clean_point(text)
        if ":" in cleaned:
            maybe_label, remainder = cleaned.split(":", 1)
            maybe_label = maybe_label.strip()
            remainder = remainder.strip()
            if maybe_label and remainder:
                return maybe_label[:24], remainder
        return fallback_label, cleaned

    def _wrap_text(self, text: str, width: int = 24, max_lines: int = 3) -> list[str]:
        text = self._clean_point(text)
        words = text.split()
        if not words:
            return []

        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
                continue
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break

        if len(lines) < max_lines and current:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        if words and len(lines) == max_lines and " ".join(lines) != " ".join(words):
            lines[-1] = lines[-1][: max(0, width - 1)].rstrip() + "…"
        return lines

    def _svg_text_block(self, x: int, y: int, lines: list[str], color: str = TEXT) -> str:
        escaped = [html.escape(line) for line in lines if line]
        if not escaped:
            return ""
        tspans = "".join(
            f'<tspan x="{x}" dy="{0 if idx == 0 else 10}">{line}</tspan>'
            for idx, line in enumerate(escaped)
        )
        return (
            f'<text x="{x}" y="{y}" fill="{color}" font-size="8" '
            f'font-family="{FONT}">{tspans}</text>'
        )

    def _concept_grid_svg(self, title: str, points: list[str]) -> str:
        cells = [
            (32, 64, PRIMARY, "Definition"),
            (248, 64, SECONDARY, "Analogy"),
            (32, 164, SECONDARY, "Mechanics"),
            (248, 164, PRIMARY, "Why it matters"),
        ]
        points = (points + ["", "", "", ""])[:4]
        parts = []
        for idx, (x, y, color, fallback_label) in enumerate(cells):
            label, body = self._split_point(points[idx], fallback_label)
            fill = PRIMARY_FILL if color == PRIMARY else SECONDARY_FILL
            parts.append(f'<rect x="{x}" y="{y}" width="200" height="84" rx="10" fill="{fill}" stroke="{color}" stroke-width="1"/>')
            parts.append(f'<text x="{x + 12}" y="{y + 15}" fill="{color}" font-size="8" font-family="{FONT}">{html.escape(label)}</text>')
            parts.append(self._svg_text_block(x + 12, y + 30, self._wrap_text(body, width=31, max_lines=4)))
        return "".join(parts)

    def _sequence_svg(self, title: str, points: list[str]) -> str:
        labels = [self._short_label(point, f"Step {idx + 1}") for idx, point in enumerate((points + ["", "", ""])[:3])]
        colors = [PRIMARY, SECONDARY, PRIMARY]
        fills = [PRIMARY_FILL, SECONDARY_FILL, PRIMARY_FILL]
        xs = [24, 178, 332]
        cards = []
        points = (points + ["", "", ""])[:3]
        for idx, point in enumerate(points):
            cards.append(f'<rect x="{xs[idx]}" y="66" width="124" height="112" rx="10" fill="{fills[idx]}" stroke="{colors[idx]}" stroke-width="1"/>')
            cards.append(f'<text x="{xs[idx] + 14}" y="88" fill="{colors[idx]}" font-size="8" font-family="{FONT}">{labels[idx]}</text>')
            cards.append(self._svg_text_block(xs[idx] + 14, 108, self._wrap_text(point or labels[idx], width=19, max_lines=5)))
        return (
            f'<line x1="148" y1="122" x2="178" y2="122" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>'
            f'<line x1="302" y1="122" x2="332" y2="122" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>'
            + "".join(cards)
        )

    def _comparison_svg(self, title: str, points: list[str]) -> str:
        left = points[0] if points else "Option A"
        right = points[1] if len(points) > 1 else "Option B"
        shared = points[2] if len(points) > 2 else ""
        left_label = self._short_label(left, "Side A")
        right_label = self._short_label(right, "Side B")
        return (
            f'<rect x="32" y="66" width="168" height="112" rx="10" fill="{PRIMARY_FILL}" stroke="{PRIMARY}" stroke-width="1"/>'
            f'<rect x="280" y="66" width="168" height="112" rx="10" fill="{SECONDARY_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
            f'<line x1="220" y1="84" x2="260" y2="84" stroke="{ARROW}" stroke-width="1.5"/>'
            f'<text x="46" y="88" fill="{PRIMARY}" font-size="8" font-family="{FONT}">{html.escape(left_label)}</text>'
            f'<text x="294" y="88" fill="{SECONDARY}" font-size="8" font-family="{FONT}">{html.escape(right_label)}</text>'
            + self._svg_text_block(46, 108, self._wrap_text(left, width=26, max_lines=6))
            + self._svg_text_block(294, 108, self._wrap_text(right, width=26, max_lines=6))
            + (
                f'<rect x="126" y="186" width="228" height="22" rx="8" fill="{PANEL_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
                f'<text x="240" y="200" text-anchor="middle" fill="{TEXT}" font-size="8" font-family="{FONT}">{html.escape(self._clean_point(shared)[:64])}</text>'
                if shared else ""
            )
        )

    def _cycle_svg(self, title: str, points: list[str]) -> str:
        points = (points + ["", "", ""])[:3]
        nodes = [
            (210, 26, PRIMARY, self._short_label(points[0], "Phase A"), points[0]),
            (58, 128, SECONDARY, self._short_label(points[1], "Phase B"), points[1]),
            (314, 128, PRIMARY, self._short_label(points[2], "Phase C"), points[2]),
        ]
        parts = []
        for x, y, color, label, text in nodes:
            fill = PRIMARY_FILL if color == PRIMARY else SECONDARY_FILL
            parts.append(f'<rect x="{x}" y="{y}" width="112" height="60" rx="10" fill="{fill}" stroke="{color}" stroke-width="1"/>')
            parts.append(f'<text x="{x + 14}" y="{y + 16}" fill="{color}" font-size="8" font-family="{FONT}">{html.escape(label)}</text>')
            parts.append(self._svg_text_block(x + 14, y + 30, self._wrap_text(text or label, width=18, max_lines=3)))
        parts.append(f'<path d="M210 74 C168 82, 132 104, 114 128" fill="none" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        parts.append(f'<path d="M170 158 C212 186, 268 186, 314 158" fill="none" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        parts.append(f'<path d="M370 128 C352 104, 318 82, 266 74" fill="none" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        return "".join(parts)

    def _layers_svg(self, title: str, points: list[str]) -> str:
        points = (points + ["", "", ""])[:3]
        ys = [60, 102, 144]
        colors = [PRIMARY, SECONDARY, PRIMARY]
        labels = [self._short_label(point, f"Layer {idx + 1}") for idx, point in enumerate(points)]
        parts = []
        for idx, point in enumerate(points):
            fill = PRIMARY_FILL if colors[idx] == PRIMARY else SECONDARY_FILL
            parts.append(f'<rect x="72" y="{ys[idx]}" width="336" height="32" rx="10" fill="{fill}" stroke="{colors[idx]}" stroke-width="1"/>')
            parts.append(f'<text x="92" y="{ys[idx] + 18}" fill="{colors[idx]}" font-size="8" font-family="{FONT}">{html.escape(labels[idx])}</text>')
            parts.append(f'<text x="170" y="{ys[idx] + 18}" fill="{TEXT}" font-size="8" font-family="{FONT}">{html.escape(self._clean_point(point)[:56])}</text>')
        return "".join(parts)

    def _hierarchy_svg(self, title: str, points: list[str]) -> str:
        root = points[0] if points else title
        left = points[1] if len(points) > 1 else "Branch A"
        right = points[2] if len(points) > 2 else "Branch B"
        return (
            f'<rect x="168" y="38" width="144" height="42" rx="10" fill="{PANEL_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
            + self._svg_text_block(186, 62, self._wrap_text(root, width=20, max_lines=2))
            + f'<line x1="240" y1="80" x2="240" y2="98" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<line x1="120" y1="118" x2="360" y2="118" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<line x1="120" y1="118" x2="120" y2="136" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<line x1="360" y1="118" x2="360" y2="136" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<rect x="44" y="136" width="152" height="56" rx="10" fill="{PRIMARY_FILL}" stroke="{PRIMARY}" stroke-width="1"/>'
            + f'<rect x="284" y="136" width="152" height="56" rx="10" fill="{SECONDARY_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
            + self._svg_text_block(60, 158, self._wrap_text(left, width=22, max_lines=2))
            + self._svg_text_block(300, 158, self._wrap_text(right, width=22, max_lines=2))
        )

    def _fallback_svg(self, source_text: str) -> str:
        brief = self._extract_brief(source_text)
        title = html.escape(str(brief["title"]))
        style = str(brief["style"]).lower()
        points = [str(point) for point in brief["points"]]

        if style == "comparison":
            body = self._comparison_svg(title, points)
        elif style == "cycle":
            body = self._cycle_svg(title, points)
        elif style == "layers":
            body = self._layers_svg(title, points)
        elif style == "hierarchy":
            body = self._hierarchy_svg(title, points)
        elif style == "workflow":
            body = self._sequence_svg(title, points)
        elif style == "concept_grid":
            body = self._concept_grid_svg(title, points)
        else:
            body = self._sequence_svg(title, points)

        return f"""<svg viewBox="0 0 480 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Fallback diagram for {title}">
<rect width="480" height="280" fill="{BG}"/>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_HEAD}"/>
  </marker>
  </defs>
<text x="18" y="20" fill="{TEXT_SOFT}" font-size="9" font-family="{FONT}">concept support fallback</text>
<text x="24" y="42" fill="{SECONDARY}" font-size="11" font-family="{FONT}">{title}</text>
{body}
</svg>"""
