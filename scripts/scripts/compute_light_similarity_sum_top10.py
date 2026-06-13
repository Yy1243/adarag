from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from adarag.data import QAItem
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever
from adarag.utils import load_yaml


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_field(x: Any, names: List[str], default=None):
    if x is None:
        return default
    if isinstance(x, dict):
        for n in names:
            if n in x and x[n] is not None:
                return x[n]
        return default
    for n in names:
        if hasattr(x, n):
            v = getattr(x, n)
            if v is not None:
                return v
    return default


def _as_answers(a: Any) -> List[str]:
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    if isinstance(a, (list, tuple)):
        return [str(z) for z in a if z is not None]
    return [str(a)]


def _make_qa(x: Any, fallback_qid: str) -> QAItem:
    q = _get_field(x, ["q", "question", "query"], default="") or ""
    a = _get_field(x, ["a", "answers", "answer", "gold", "ground_truth"], default=None)
    qid = _get_field(x, ["qid", "id", "example_id"], default="") or fallback_qid
    return QAItem(q=str(q), a=_as_answers(a), qid=str(qid))


def load_questions(cfg: dict, max_questions: Optional[int] = None) -> List[QAItem]:
    path = cfg["dataset"]["local_path"]
    items: List[QAItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            items.append(_make_qa(json.loads(line), fallback_qid=f"q{i}"))
            if max_questions is not None and len(items) >= max_questions:
                break
    if not items:
        raise RuntimeError("dataset is empty")
    return items


def build_light_retriever(cfg: dict, top_k: int):
    light_cfg = cfg.get("light_retriever", {}) or {}
    retriever = FaissHNSWRetriever(
        corpus_path=light_cfg["corpus_path"],
        index_path=light_cfg["index_path"],
        embedding_model=light_cfg.get("embedding_model", "jinaai/jina-embeddings-v2-base-en"),
        top_n=top_k,
        device=light_cfg.get("device", "cpu"),
    )
    retriever.load_or_build(
        rebuild=bool(light_cfg.get("rebuild", False)),
        max_passages=light_cfg.get("max_passages", None),
    )
    return retriever


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-question light-retrieval top-k similarity sum.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--output-csv", type=str, required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-questions", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    questions = load_questions(cfg, max_questions=args.max_questions)
    retriever = build_light_retriever(cfg, top_k=int(args.top_k))

    rows: List[Dict[str, Any]] = []
    for qa in tqdm(questions, desc="Light similarity sum"):
        _, scores = retriever.retrieve_k(qa.q, int(args.top_k))
        rows.append(
            {
                "qid": qa.qid,
                f"light_score_sum_top{int(args.top_k)}": float(scores.sum()) if len(scores) else 0.0,
            }
        )

    _ensure_dir(os.path.dirname(args.output_csv) or ".")
    pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
