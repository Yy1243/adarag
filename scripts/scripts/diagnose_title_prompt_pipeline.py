# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from adarag.utils import load_yaml
from adarag.pipeline.prompt_builder import build_prompt
from scripts.eval_light_heavy_retrieval_generation import build_light_retriever, build_heavy_retriever


STOPWORDS = {
    "what", "who", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "the", "a", "an",
    "of", "to", "in", "on", "for", "at", "by", "from", "with", "about", "into", "over", "under",
    "and", "or", "but", "if", "then", "than", "as",
    "which", "that", "this", "these", "those",
    "it", "its", "their", "his", "her",
    "name", "named", "called",
    "year", "date", "time", "place", "located", "location",
}


def simplify_query(q: str) -> str:
    q = (q or "").lower()
    toks = re.findall(r"[a-z0-9]+", q)
    toks = [t for t in toks if len(t) >= 2 and t not in STOPWORDS]
    return " ".join(toks) if toks else (q or "").strip()


def load_qid(dataset_path: str, qid: int) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            cur_qid = obj.get("qid", obj.get("id", i))
            if int(cur_qid) == int(qid):
                ans = obj.get("answer", obj.get("answers", []))
                if isinstance(ans, str):
                    ans = [ans]
                return {
                    "qid": cur_qid,
                    "question": obj.get("question", obj.get("q", "")),
                    "answers": ans,
                    "raw": obj,
                }
    raise ValueError(f"qid={qid} not found in {dataset_path}")


def get_doc_text(d: Any) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        return str(d.get("text") or d.get("contents") or d.get("passage") or d.get("document") or "")
    for attr in ("text", "contents", "passage", "document"):
        if hasattr(d, attr):
            return str(getattr(d, attr) or "")
    return str(d)

def get_doc_title(d: Any) -> str:
    if d is None:
        return ""
    if isinstance(d, dict):
        return str(d.get("title") or "")
    return str(getattr(d, "title", "") or "")


def get_doc_full_text(d: Any) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        title = str(d.get("title") or "").strip()
        text = str(d.get("text") or d.get("contents") or d.get("passage") or d.get("document") or "").strip()
        if title and text:
            return f"{title}\n{text}"
        return title or text
    if hasattr(d, "full_text"):
        return str(d.full_text or "")
    return get_doc_text(d)


def preview(s: str, n: int = 600) -> str:
    s = str(s or "")
    s = s.replace("\r", "\\r")
    # 保留换行，方便看 title/body 分层
    if len(s) > n:
        return s[:n] + f"\n...[TRUNCATED chars={len(s)}]"
    return s


def first_lines(s: str, n: int = 5) -> str:
    lines = str(s or "").splitlines()
    return "\n".join(f"{i+1:02d}: {x}" for i, x in enumerate(lines[:n]))


def starts_with_title(text: str, title: str) -> bool:
    text = str(text or "").strip()
    title = str(title or "").strip()
    return bool(title) and text.lower().startswith(title.lower())


def es_raw_search(
    *,
    es_url: str,
    index_name: str,
    query: str,
    size: int,
    query_mode: str,
    timeout: int,
) -> List[Dict[str, Any]]:
    q_raw = query or ""
    q_simple = simplify_query(q_raw)

    if query_mode == "cross_fields":
        query_body = {
            "multi_match": {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "cross_fields",
                "operator": "or",
            }
        }
    else:
        query_body = {
            "multi_match": {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "best_fields",
            }
        }

    body = {
        "size": size,
        "_source": ["doc_id", "title", "text"],
        "query": query_body,
        "track_total_hits": False,
    }

    url = f"{es_url.rstrip('/')}/{index_name}/_search"
    r = requests.post(url, json=body, timeout=(3.0, timeout))
    r.raise_for_status()
    return r.json().get("hits", {}).get("hits", []) or []


def print_es_layer(hits: List[Dict[str, Any]], topn: int = 5) -> None:
    print("\n" + "=" * 100)
    print("[A] ES RAW RETURN LAYER")
    print("=" * 100)

    for idx, h in enumerate(hits[:topn], 1):
        src = h.get("_source", {}) or {}
        title = str(src.get("title", "") or "")
        text = str(src.get("text", "") or "")
        score = h.get("_score", None)

        print(f"\n--- ES RAW #{idx} score={score} ---")
        print(f"title: {title!r}")
        print(f"text_starts_with_title: {starts_with_title(text, title)}")
        print("[text first lines]")
        print(first_lines(text, 5))
        print("[text preview]")
        print(preview(text, 500))


def print_docs_layer(name: str, docs: List[Any], topn: int = 5) -> None:
    print("\n" + "=" * 100)
    print(f"[B] RETRIEVER DOC LAYER: {name}")
    print("=" * 100)

    for idx, d in enumerate(docs[:topn], 1):
        title = get_doc_title(d)
        text = get_doc_text(d)
        full_text = get_doc_full_text(d)

        print(f"\n--- {name} DOC #{idx} ---")
        print(f"doc_type: {type(d)}")
        print(f"title: {title!r}")
        print(f"text_starts_with_title: {starts_with_title(text, title)}")
        print("[Doc.text first lines]")
        print(first_lines(text, 6))
        print("[Doc.full_text first lines]")
        print(first_lines(full_text, 6))
        print("[Doc.text preview]")
        print(preview(text, 600))


def print_prompt_layer(question: str, docs: List[Any], max_doc_chars: int, name: str, out_dir: Path) -> None:
    print("\n" + "=" * 100)
    print(f"[C] FINAL PROMPT LAYER: {name}")
    print("=" * 100)

    prompt = build_prompt(question, docs, max_doc_chars=max_doc_chars)
    out_path = out_dir / f"{name}_prompt.txt"
    out_path.write_text(prompt, encoding="utf-8")

    print(f"Saved prompt to: {out_path}")
    print(f"prompt_chars: {len(prompt)}")
    print("[prompt first 4000 chars]")
    print(prompt[:4000])
    if len(prompt) > 4000:
        print(f"\n...[TRUNCATED PRINT chars={len(prompt)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. scripts/config_eval_light_heavy_7b_j14b.yaml")
    ap.add_argument("--qid", type=int, default=6)
    ap.add_argument("--es-size", type=int, default=5)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--out-dir", default="outputs_title_prompt_diagnose")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cfg["dataset"]["local_path"] if isinstance(cfg.get("dataset"), dict) else cfg["dataset"]
    item = load_qid(dataset_path, args.qid)
    q = item["question"]

    print("=" * 100)
    print("[QUESTION]")
    print("=" * 100)
    print(f"qid: {item['qid']}")
    print(f"question: {q}")
    print(f"answers: {item['answers']}")

    heavy_cfg = cfg.get("heavy_retriever", {}) or {}
    es_url = heavy_cfg.get("es_url", "http://127.0.0.1:9200")
    index_name = heavy_cfg.get("index_name", "adarag_kb_v2_1m")
    timeout = int(heavy_cfg.get("request_timeout", 120))
    query_mode = str(heavy_cfg.get("query_mode", "cross_fields"))

    # A. ES 原始返回层
    hits = es_raw_search(
        es_url=es_url,
        index_name=index_name,
        query=q,
        size=args.es_size,
        query_mode=query_mode,
        timeout=timeout,
    )
    print_es_layer(hits, topn=args.es_size)

    # B1. Dense retriever 文档层
    print("\n[init] loading light retriever...")
    light = build_light_retriever(cfg, topn=max(args.topn, 10))
    dense_docs, _ = light.retrieve(q)
    dense_docs = dense_docs[:args.topn]
    print_docs_layer("dense", dense_docs, topn=min(args.topn, 5))

    # B2. Heavy retriever 文档层
    try:
        print("\n[init] loading heavy retriever...")
        heavy = build_heavy_retriever(cfg, topn=args.topn, light=light, llm_rewriter=None)
        heavy_docs, _ = heavy.retrieve(q)
        heavy_docs = heavy_docs[:args.topn]
        print_docs_layer("heavy", heavy_docs, topn=min(args.topn, 5))
    except Exception as e:
        print("\n[WARN] heavy retriever failed, skip heavy layer.")
        print(repr(e))
        heavy_docs = []

    max_doc_chars = int((cfg.get("system", {}) or {}).get("prompt_max_doc_chars", 1600))

    # C. 最终 prompt 层
    print_prompt_layer(q, dense_docs[:args.topn], max_doc_chars, "dense", out_dir)
    if heavy_docs:
        print_prompt_layer(q, heavy_docs[:args.topn], max_doc_chars, "heavy", out_dir)

    print("\nDONE.")


if __name__ == "__main__":
    main()