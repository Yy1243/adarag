from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any
import json


@dataclass
class Doc:
    doc_id: str
    text: str
    title: str = ""

    @property
    def full_text(self) -> str:
        title = str(self.title or "").strip()
        text = str(self.text or "").strip()

        if title and text:
            return f"{title}\n{text}"
        if title:
            return title
        return text


@dataclass
class QAItem:
    q: str
    a: List[str]
    qid: Optional[str] = None

    @property
    def question(self) -> str:
        return self.q

    @property
    def answer(self) -> List[str]:
        return self.a


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(t) for t in x]


def load_corpus_jsonl(path: str) -> List[Doc]:
    docs: List[Doc] = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            o = json.loads(line)

            doc_id = o.get("doc_id", o.get("id", o.get("pid", i)))
            title = str(o.get("title", "") or "").strip()
            text = str(o.get("text", o.get("contents", "")) or "").strip()

            docs.append(
                Doc(
                    doc_id=str(doc_id),
                    title=title,
                    text=text,
                )
            )

    return docs


def load_qa_jsonl(path: str) -> List[QAItem]:
    items: List[QAItem] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            o = json.loads(line)

            qid = o.get("qid", None)
            if "q" in o:
                q = o.get("q", "")
                a = _as_list(o.get("a", []))
            else:
                q = o.get("question", "")
                a = _as_list(o.get("answer", []))

            items.append(
                QAItem(
                    q=str(q),
                    a=a,
                    qid=str(qid) if qid is not None else None,
                )
            )

    return items