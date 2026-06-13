# -*- coding: utf-8 -*-
"""
A minimal "LLM" for fully-offline execution.

Given (question, retrieved_docs), we return a short extractive answer:
- Find the sentence in top documents with maximum overlap with query words
- Return that sentence truncated

This is NOT meant to reproduce absolute accuracy numbers of the paper,
but it makes the whole AdaRAG pipeline executable from scratch.

Swap with a real LLM in `adarag/llm/hf_llm.py` or your vLLM deployment.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

from adarag.retrievers.light_tfidf import RetrievedDoc


def _sentences(text: str) -> List[str]:
    # Simple sentence split; for demo only
    s = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [x.strip() for x in s if x.strip()]


def _tokset(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return set(toks)


class ExtractiveLLM:
    def __init__(self, *, max_answer_chars: int = 120) -> None:
        self.max_answer_chars = max_answer_chars

    def generate(self, question: str, docs: List[RetrievedDoc]) -> str:
        """
        Return an answer string.
        """
        qset = _tokset(question)
        best = ""
        best_score = -1
        for d in docs[:5]:
            for s in _sentences(d.text):
                sset = _tokset(s)
                score = len(qset & sset)
                if score > best_score:
                    best_score = score
                    best = s
        if not best:
            best = docs[0].text if docs else ""
        best = best.strip().replace("\n", " ")
        return best[: self.max_answer_chars].strip()
