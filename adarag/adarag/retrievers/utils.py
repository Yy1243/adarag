# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any

def doc_to_str(d: Any) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        title = d.get("title") or ""
        text = d.get("text") or d.get("contents") or d.get("passage") or d.get("document") or ""
        if title and text:
            return f"{title}\n{text}"
        return title or text or ""
    # object-like
    title = getattr(d, "title", "") or ""
    text = (
        getattr(d, "text", None)
        or getattr(d, "contents", None)
        or getattr(d, "passage", None)
        or getattr(d, "document", None)
        or ""
    )
    if title and text:
        return f"{title}\n{text}"
    return title or str(text) or str(d)
