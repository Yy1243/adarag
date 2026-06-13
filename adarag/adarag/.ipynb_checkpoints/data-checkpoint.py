from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Any
import json


@dataclass
class Doc:
    doc_id: str
    text: str


@dataclass       #装饰器:用于自动生成类的常用方法。
class QAItem:
    q: str
    a: List[str]
    qid: Optional[str] = None   #可选有默认值，所以即使没有传入qid也不会报错的.Optional[str]参数注解，可以是str格式也可以是None，默认值为None，如果传入了qid参数，那么它必须是一个字符串；如果没有传入qid参数，那么它的值就是None，这样就实现了qid参数的可选性。

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

            # robust field mapping
            doc_id = o.get("doc_id", o.get("id", o.get("pid", i)))
            title = o.get("title", "") or ""
            text = o.get("text", o.get("contents", "")) or ""
            full_text = f"{title}\n{text}".strip() if title else str(text).strip()

            docs.append(Doc(doc_id=str(doc_id), text=full_text))
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

            items.append(QAItem(q=str(q), a=a, qid=str(qid) if qid is not None else None))
    return items
