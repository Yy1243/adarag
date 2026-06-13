# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, List


def _doc_full_text(d: Any) -> str:
    if d is None:
        return ""

    if isinstance(d, str):
        return d

    if isinstance(d, dict):
        return str(
            d.get("text")
            or d.get("contents")
            or d.get("passage")
            or d.get("document")
            or ""
        ).strip()

    for attr in ("text", "contents", "passage", "document"):
        if hasattr(d, attr):
            v = getattr(d, attr)
            if v is not None:
                return str(v).strip()

    return str(d).strip()


def format_doc_for_prompt(d: Any, max_doc_chars: int = 1600) -> str:
    text = _doc_full_text(d).strip()

    if max_doc_chars and max_doc_chars > 0 and len(text) > max_doc_chars:
        text = text[:max_doc_chars]

    return text


def build_prompt(question: str, docs: List[Any], max_doc_chars: int = 1600) -> str:
    question = str(question or "").strip()

    lines = []
    lines.append("You are an extractive QA system.")
    lines.append("Answer questions using ONLY the Context.")
    lines.append("If the context is empty or does not contain the answer, answer using your own knowledge.")
    lines.append("Return ONLY the short answer (1-10 words) in ONE line. No explanation. No extra words")
    lines.append("")
    lines.append("Context:")

    for d in docs or []:
        doc_block = format_doc_for_prompt(d, max_doc_chars=max_doc_chars)
        if not doc_block:
            continue
        lines.append(doc_block)
        lines.append("")

    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)