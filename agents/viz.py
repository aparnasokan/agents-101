# agents/viz.py
# Generates SVG diagrams to accompany concept explanations.
# Runs after the Concept agent so it can support that explanation.
# Outputs raw SVG that the UI renders inline.

import html
import re

from .base import BaseAgent

SYSTEM = """You are the Visual agent. Generate a clean SVG diagram that visually supports the Concept agent's explanation.

The conversation will include the original user question and may include an assistant message starting with:
"Visualization brief:"
Use that brief as the primary source for the diagram, especially its title, diagram style, and key points.

Output ONLY a raw SVG element — no explanation, no markdown, no prose around it. Just the SVG.

Requirements:
- Use a 480-wide viewBox and expand the height when needed so all text remains visible
- Dark background: start with <rect width="480" height="280" fill="#031114"/>
- Colors: primary=#1df4f4, secondary=#a2f3f3, text=#ffffff, panel=#053057
- Node boxes: rx=8, stroke-width=1, use fill with low opacity + matching stroke
- Text: fill=#ffffff, font-size=10, font-family="Montserrat, Arial, sans-serif"
- Arrows: use <line> or <path> with marker-end for direction, stroke=rgba(162,243,243,0.3)
- Add arrowhead marker in <defs>
- Keep it simple: max 7 nodes, clear left-to-right or top-to-bottom flow
- Make the visual explain the concept itself, not the multi-agent system, unless the concept is explicitly about orchestration or routing
- Prefer visual structures like stages, loops, components, inputs/outputs, or before/after states based on the concept explanation
- If the brief suggests `concept_grid`, `workflow`, `sequence`, `cycle`, `comparison`, `layers`, or `hierarchy`, reflect that structure directly in the SVG"""

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

    def _strip_heading_prefix(self, text: str, label: str = "") -> str:
        cleaned = self._plain_text(text)
        prefixes = [
            label,
            "Definition",
            "Analogy",
            "Mechanics",
            "Why it matters",
            "How it works",
            "Technical explanation",
        ]
        for prefix in prefixes:
            if not prefix:
                continue
            updated = re.sub(
                rf"^{re.escape(prefix)}\s*[:\-]?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            if updated != cleaned:
                cleaned = updated.strip()
        return cleaned

    def _clean_point(self, text: str) -> str:
        text = re.sub(r"^\s*\d+[\).\s:-]*", "", text)
        text = re.sub(r"^\s*[-•]\s*", "", text)
        return self._plain_text(text)

    def _compact_text(self, text: str, limit: int = 72) -> str:
        cleaned = self._strip_heading_prefix(text)
        cleaned = cleaned.strip(" -:;,.")
        return cleaned

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
                label = maybe_label[:24]
                return label, self._strip_heading_prefix(remainder, label)
        return fallback_label, self._strip_heading_prefix(cleaned, fallback_label)

    def _wrap_text(self, text: str, width: int = 24, max_lines: int | None = None) -> list[str]:
        text = self._compact_text(self._clean_point(text), limit=width)
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
            if max_lines is not None and len(lines) == max_lines:
                break

        if (max_lines is None or len(lines) < max_lines) and current:
            lines.append(current)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
        if max_lines is not None and words and len(lines) == max_lines and " ".join(lines) != " ".join(words):
            lines[-1] = lines[-1][: max(0, width - 1)].rstrip() + "…"
        return lines

    def _text_block_height(self, lines: list[str], line_height: int = 9) -> int:
        return max(0, len(lines) * line_height)

    def _svg_text_block(self, x: int, y: int, lines: list[str], color: str = TEXT) -> str:
        escaped = [html.escape(line) for line in lines if line]
        if not escaped:
            return ""
        tspans = "".join(
            f'<tspan x="{x}" dy="{0 if idx == 0 else 9}">{line}</tspan>'
            for idx, line in enumerate(escaped)
        )
        return (
            f'<text x="{x}" y="{y}" fill="{color}" font-size="7" '
            f'font-family="{FONT}">{tspans}</text>'
        )

    def _concept_grid_svg(self, title: str, points: list[str]) -> tuple[str, int]:
        labels_and_colors = [
            ("Definition", PRIMARY),
            ("Analogy", SECONDARY),
            ("Mechanics", SECONDARY),
            ("Why it matters", PRIMARY),
        ]
        points = (points + ["", "", "", ""])[:4]
        y = 36
        parts = []
        bottoms: list[int] = []
        for idx, (fallback_label, color) in enumerate(labels_and_colors):
            label, body = self._split_point(points[idx], fallback_label)
            lines = self._wrap_text(body, width=62)
            height = max(54, 28 + self._text_block_height(lines) + 16)
            fill = PRIMARY_FILL if color == PRIMARY else SECONDARY_FILL
            parts.append(f'<rect x="32" y="{y}" width="416" height="{height}" rx="10" fill="{fill}" stroke="{color}" stroke-width="1"/>')
            parts.append(f'<text x="46" y="{y + 15}" fill="{color}" font-size="7" font-family="{FONT}">{html.escape(label)}</text>')
            parts.append(self._svg_text_block(46, y + 29, lines))
            bottoms.append(y + height)
            y += height + 12
        return "".join(parts), max(bottoms, default=280) + 16

    def _sequence_svg(self, title: str, points: list[str]) -> tuple[str, int]:
        labels = [self._short_label(point, f"Step {idx + 1}") for idx, point in enumerate((points + ["", "", ""])[:3])]
        colors = [PRIMARY, SECONDARY, PRIMARY]
        fills = [PRIMARY_FILL, SECONDARY_FILL, PRIMARY_FILL]
        xs = [24, 178, 332]
        cards = []
        bottoms: list[int] = []
        points = (points + ["", "", ""])[:3]
        for idx, point in enumerate(points):
            lines = self._wrap_text(point or labels[idx], width=19)
            height = max(112, 44 + self._text_block_height(lines) + 16)
            cards.append(f'<rect x="{xs[idx]}" y="66" width="124" height="{height}" rx="10" fill="{fills[idx]}" stroke="{colors[idx]}" stroke-width="1"/>')
            cards.append(f'<text x="{xs[idx] + 14}" y="88" fill="{colors[idx]}" font-size="7" font-family="{FONT}">{labels[idx]}</text>')
            cards.append(self._svg_text_block(xs[idx] + 14, 108, lines))
            bottoms.append(66 + height)
        body = (
            f'<line x1="148" y1="122" x2="178" y2="122" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>'
            f'<line x1="302" y1="122" x2="332" y2="122" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>'
            + "".join(cards)
        )
        return body, max(bottoms, default=280) + 16

    def _workflow_svg(self, title: str, points: list[str]) -> tuple[str, int]:
        points = (points + ["", "", "", ""])[:4]
        cards = [
            (32, 76, 118, 60, PRIMARY, "Ask", points[0] or "The user describes the problem to solve."),
            (182, 42, 116, 60, SECONDARY, "Interpret", points[1] or "The agent extracts intent, numbers, and constraints."),
            (182, 158, 116, 60, PRIMARY, "Compute", points[2] or "A tool or internal logic performs the calculation."),
            (332, 76, 116, 60, SECONDARY, "Respond", points[3] or "The agent returns the answer with a quick explanation."),
        ]

        parts = [
            f'<rect x="176" y="112" width="128" height="34" rx="17" fill="{PANEL_FILL}" stroke="{SECONDARY}" stroke-width="1"/>',
            f'<text x="240" y="132" text-anchor="middle" fill="{TEXT}" font-size="8" font-family="{FONT}">decision + tool use</text>',
        ]
        bottoms: list[int] = [146]

        for x, y, w, h, color, label, body in cards:
            lines = self._wrap_text(body, width=20)
            height = max(h, 44 + self._text_block_height(lines) + 16)
            fill = PRIMARY_FILL if color == PRIMARY else SECONDARY_FILL
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{height}" rx="10" fill="{fill}" stroke="{color}" stroke-width="1"/>')
            parts.append(f'<text x="{x + 12}" y="{y + 16}" fill="{color}" font-size="7" font-family="{FONT}">{html.escape(label)}</text>')
            parts.append(self._svg_text_block(x + 12, y + 30, lines))
            bottoms.append(y + height)

        parts.append(f'<line x1="150" y1="106" x2="182" y2="86" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        parts.append(f'<line x1="150" y1="106" x2="182" y2="188" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        parts.append(f'<line x1="298" y1="86" x2="332" y2="106" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        parts.append(f'<line x1="298" y1="188" x2="332" y2="106" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        return "".join(parts), max(bottoms, default=280) + 16

    def _comparison_svg(self, title: str, points: list[str]) -> tuple[str, int]:
        left = points[0] if points else "Option A"
        right = points[1] if len(points) > 1 else "Option B"
        shared = points[2] if len(points) > 2 else ""
        left_label = self._short_label(left, "Side A")
        right_label = self._short_label(right, "Side B")
        left_lines = self._wrap_text(left, width=26)
        right_lines = self._wrap_text(right, width=26)
        left_height = max(112, 44 + self._text_block_height(left_lines) + 16)
        right_height = max(112, 44 + self._text_block_height(right_lines) + 16)
        shared_y = max(left_height, right_height) + 84
        body = (
            f'<rect x="32" y="66" width="168" height="{left_height}" rx="10" fill="{PRIMARY_FILL}" stroke="{PRIMARY}" stroke-width="1"/>'
            f'<rect x="280" y="66" width="168" height="{right_height}" rx="10" fill="{SECONDARY_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
            f'<line x1="220" y1="84" x2="260" y2="84" stroke="{ARROW}" stroke-width="1.5"/>'
            f'<text x="46" y="88" fill="{PRIMARY}" font-size="7" font-family="{FONT}">{html.escape(left_label)}</text>'
            f'<text x="294" y="88" fill="{SECONDARY}" font-size="7" font-family="{FONT}">{html.escape(right_label)}</text>'
            + self._svg_text_block(46, 108, left_lines)
            + self._svg_text_block(294, 108, right_lines)
            + (
                f'<rect x="126" y="{shared_y}" width="228" height="22" rx="8" fill="{PANEL_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
                f'<text x="240" y="{shared_y + 14}" text-anchor="middle" fill="{TEXT}" font-size="7" font-family="{FONT}">{html.escape(self._clean_point(shared))}</text>'
                if shared else ""
            )
        )
        overall = max(66 + left_height, 66 + right_height, shared_y + (38 if shared else 0))
        return body, overall + 16

    def _cycle_svg(self, title: str, points: list[str]) -> tuple[str, int]:
        points = (points + ["", "", ""])[:3]
        nodes = [
            (210, 26, PRIMARY, self._short_label(points[0], "Phase A"), points[0]),
            (58, 128, SECONDARY, self._short_label(points[1], "Phase B"), points[1]),
            (314, 128, PRIMARY, self._short_label(points[2], "Phase C"), points[2]),
        ]
        parts = []
        bottoms: list[int] = []
        for x, y, color, label, text in nodes:
            lines = self._wrap_text(text or label, width=18)
            height = max(60, 34 + self._text_block_height(lines) + 16)
            fill = PRIMARY_FILL if color == PRIMARY else SECONDARY_FILL
            parts.append(f'<rect x="{x}" y="{y}" width="112" height="{height}" rx="10" fill="{fill}" stroke="{color}" stroke-width="1"/>')
            parts.append(f'<text x="{x + 14}" y="{y + 16}" fill="{color}" font-size="7" font-family="{FONT}">{html.escape(label)}</text>')
            parts.append(self._svg_text_block(x + 14, y + 30, lines))
            bottoms.append(y + height)
        parts.append(f'<path d="M210 74 C168 82, 132 104, 114 128" fill="none" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        parts.append(f'<path d="M170 158 C212 186, 268 186, 314 158" fill="none" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        parts.append(f'<path d="M370 128 C352 104, 318 82, 266 74" fill="none" stroke="{ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        return "".join(parts), max(bottoms, default=280) + 16

    def _layers_svg(self, title: str, points: list[str]) -> tuple[str, int]:
        points = (points + ["", "", ""])[:3]
        current_y = 60
        colors = [PRIMARY, SECONDARY, PRIMARY]
        labels = [self._short_label(point, f"Layer {idx + 1}") for idx, point in enumerate(points)]
        parts = []
        bottoms: list[int] = []
        for idx, point in enumerate(points):
            lines = self._wrap_text(self._clean_point(point), width=28)
            height = max(32, 18 + self._text_block_height(lines) + 10)
            y = current_y
            fill = PRIMARY_FILL if colors[idx] == PRIMARY else SECONDARY_FILL
            parts.append(f'<rect x="72" y="{y}" width="336" height="{height}" rx="10" fill="{fill}" stroke="{colors[idx]}" stroke-width="1"/>')
            parts.append(f'<text x="92" y="{y + 18}" fill="{colors[idx]}" font-size="7" font-family="{FONT}">{html.escape(labels[idx])}</text>')
            parts.append(self._svg_text_block(170, y + 18, lines))
            bottoms.append(y + height)
            current_y += height + 10
        return "".join(parts), max(bottoms, default=280) + 16

    def _hierarchy_svg(self, title: str, points: list[str]) -> tuple[str, int]:
        root = points[0] if points else title
        left = points[1] if len(points) > 1 else "Branch A"
        right = points[2] if len(points) > 2 else "Branch B"
        root_lines = self._wrap_text(root, width=20)
        left_lines = self._wrap_text(left, width=22)
        right_lines = self._wrap_text(right, width=22)
        root_height = max(42, 20 + self._text_block_height(root_lines) + 12)
        child_height = max(56, 18 + max(self._text_block_height(left_lines), self._text_block_height(right_lines)) + 16)
        branch_y = 38 + root_height + 20
        child_y = branch_y + 18
        body = (
            f'<rect x="168" y="38" width="144" height="{root_height}" rx="10" fill="{PANEL_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
            + self._svg_text_block(186, 56, root_lines)
            + f'<line x1="240" y1="{38 + root_height}" x2="240" y2="{branch_y}" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<line x1="120" y1="{branch_y}" x2="360" y2="{branch_y}" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<line x1="120" y1="{branch_y}" x2="120" y2="{child_y}" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<line x1="360" y1="{branch_y}" x2="360" y2="{child_y}" stroke="{ARROW}" stroke-width="1.5"/>'
            + f'<rect x="44" y="{child_y}" width="152" height="{child_height}" rx="10" fill="{PRIMARY_FILL}" stroke="{PRIMARY}" stroke-width="1"/>'
            + f'<rect x="284" y="{child_y}" width="152" height="{child_height}" rx="10" fill="{SECONDARY_FILL}" stroke="{SECONDARY}" stroke-width="1"/>'
            + self._svg_text_block(60, child_y + 22, left_lines)
            + self._svg_text_block(300, child_y + 22, right_lines)
        )
        return body, child_y + child_height + 16

    def _fallback_svg(self, source_text: str) -> str:
        brief = self._extract_brief(source_text)
        title = html.escape(str(brief["title"]))
        style = str(brief["style"]).lower()
        points = [str(point) for point in brief["points"]]

        if style == "comparison":
            body, content_bottom = self._comparison_svg(title, points)
        elif style == "cycle":
            body, content_bottom = self._cycle_svg(title, points)
        elif style == "layers":
            body, content_bottom = self._layers_svg(title, points)
        elif style == "hierarchy":
            body, content_bottom = self._hierarchy_svg(title, points)
        elif style == "workflow":
            body, content_bottom = self._workflow_svg(title, points)
        elif style == "concept_grid":
            body, content_bottom = self._concept_grid_svg(title, points)
        else:
            body, content_bottom = self._sequence_svg(title, points)

        view_height = max(280, content_bottom + 20)
        return f"""<svg viewBox="0 0 480 {view_height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diagram for {title or 'agent concept'}">
<rect width="480" height="{view_height}" fill="{BG}"/>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_HEAD}"/>
  </marker>
  </defs>
{body}
</svg>"""
