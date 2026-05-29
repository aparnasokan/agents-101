
# visual_store.py
# Retrieval-first visual selection for Agent 101.
# Prefer trusted SVGs from visuals/manifest.json over free-form generated diagrams.

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass
class VisualMatch:
    id: str
    title: str
    file: str
    score: float
    svg: str
    reason: str
    metadata: dict[str, Any]


class VisualAssetStore:
    def __init__(self, manifest_path: str | None = None) -> None:
        self.enabled = _truthy(os.getenv("VISUAL_STORE_ENABLED"), default=True)
        self.debug = _truthy(os.getenv("VISUAL_DEBUG"), default=False)
        raw_manifest = manifest_path or os.getenv("VISUAL_MANIFEST", "visuals/manifest.json")
        self.manifest_path = Path(raw_manifest)
        self.assets: list[dict[str, Any]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.enabled:
            return
        if not self.manifest_path.exists():
            if self.debug:
                print(f"[VisualStore] manifest not found: {self.manifest_path}")
            return
        try:
            self.assets = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            print(f"[VisualStore] Loaded {len(self.assets)} visual assets from {self.manifest_path}")
        except Exception as exc:
            print(f"[VisualStore] Failed loading manifest: {exc}")
            self.assets = []

    def _infer_intent(self, query: str) -> str:
        q = _plain(query)
        if any(x in q for x in ["deploy", "production", "eval", "monitor", "guardrail", "latency", "cost"]):
            return "production"
        if any(x in q for x in ["code", "implement", "python", "api", "build"]):
            return "implementation"
        return "concept"

    def _infer_difficulty(self, query: str) -> str:
        q = _plain(query)
        if any(x in q for x in ["advanced", "architecture", "production", "deploy", "eval", "observability", "scale"]):
            return "advanced"
        if any(x in q for x in ["implement", "code", "python", "api", "architecture"]):
            return "intermediate"
        return "beginner"

    def search(self, query: str, rag_context: str = "", concept_content: str = "") -> VisualMatch | None:
        if not self.enabled or not self.assets:
            return None
        haystack = _plain("\n".join([query, rag_context[:3000], concept_content[:2000]]))
        intent = self._infer_intent(query)
        difficulty = self._infer_difficulty(query)

        best: tuple[float, dict[str, Any], str] | None = None
        for asset in self.assets:
            score = 0.0
            reasons: list[str] = []
            for topic in asset.get("topics", []):
                t = _plain(str(topic))
                if t and t in haystack:
                    score += 3.0 if len(t) > 5 else 1.5
                    reasons.append(f"topic={topic}")
            if intent in asset.get("intent", []):
                score += 1.0
                reasons.append(f"intent={intent}")
            if asset.get("difficulty") == difficulty:
                score += 0.6
                reasons.append(f"difficulty={difficulty}")
            title = _plain(asset.get("title", ""))
            if title and title in haystack:
                score += 2.0
                reasons.append("title match")
            if best is None or score > best[0]:
                best = (score, asset, ", ".join(reasons) or "best available")

        if best is None or best[0] < float(os.getenv("VISUAL_MIN_SCORE", "1.2")):
            if self.debug:
                print("[VisualStore] No reliable visual match; falling back to VizAgent")
            return None

        score, asset, reason = best
        svg_path = Path(asset["file"])
        if not svg_path.is_absolute():
            # Resolve relative to current working directory first, then manifest directory.
            cwd_path = Path.cwd() / svg_path
            manifest_path = self.manifest_path.parent / svg_path.name if svg_path.parts and svg_path.parts[0] == "visuals" else self.manifest_path.parent / svg_path
            svg_path = cwd_path if cwd_path.exists() else manifest_path
        if not svg_path.exists():
            print(f"[VisualStore] Matched visual missing on disk: {svg_path}")
            return None
        svg = svg_path.read_text(encoding="utf-8")
        print(f"[VisualStore] Selected visual: {asset.get('id')} score={score:.2f} reason={reason}")
        return VisualMatch(
            id=str(asset.get("id")),
            title=str(asset.get("title")),
            file=str(svg_path),
            score=score,
            svg=svg,
            reason=reason,
            metadata=asset,
        )


visual_store = VisualAssetStore()
