# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from adarag.data import QAItem
from adarag.data_hf import load_nq_open_stream
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever


STOP = {
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


def _get_field(x: Any, names: List[str], default=None):
    if isinstance(x, dict):
        for n in names:
            if n in x and x[n] is not None:
                return x[n]
    else:
        for n in names:
            if hasattr(x, n):
                v = getattr(x, n)
                if v is not None:
                    return v
    return default


def _make_qa(x: Any, idx: int) -> QAItem:
    q = _get_field(x, ["q", "question", "query"], "")
    a = _get_field(x, ["a", "answers", "answer", "gold", "ground_truth"], [])
    if isinstance(a, str):
        a = [a]
    elif not isinstance(a, list):
        a = [str(a)]
    qid = _get_field(x, ["qid", "id", "example_id"], str(idx))
    return QAItem(q=str(q), a=[str(z) for z in a if z is not None], qid=str(qid))


def load_questions(dataset_path: str, max_examples: int, max_questions: int, seed: int):
    it = load_nq_open_stream(
        split="validation",
        seed=seed,
        local_path=dataset_path,
        max_examples=max_examples,
    )
    out = []
    for i, x in enumerate(it):
        out.append(_make_qa(x, i))
        if len(out) >= max_questions:
            break
    return out


def norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def contains_any(text: str, answers: List[str]) -> bool:
    t = norm(text)
    for a in answers:
        aa = norm(a)
        if aa and aa in t:
            return True
    return False


def hit_docs(docs: List[str], answers: List[str]) -> bool:
    return any(contains_any(d, answers) for d in docs)


def simplify_query(q: str) -> str:
    toks = re.findall(r"[a-z0-9]+", (q or "").lower())
    toks = [t for t in toks if len(t) >= 2 and t not in STOP]
    return " ".join(toks) if toks else (q or "")


def es_search(es_url: str, index_name: str, body: Dict[str, Any], timeout: float = 120.0):
    url = f"{es_url.rstrip('/')}/{index_name}/_search"
    r = requests.post(url, json=body, timeout=(3.0, timeout))
    if r.status_code >= 400:
        print("ES ERROR", r.status_code)
        print(json.dumps(body, ensure_ascii=False)[:3000])
        print(r.text[:3000])
        r.raise_for_status()
    return r.json().get("hits", {}).get("hits", []) or []


def make_query(q: str, variant: str, topk: int) -> Dict[str, Any]:
    q_simple = q 

    if variant == "standard":
        query = {
            "multi_match": {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "cross_fields",
                "operator": "or",
            }
        }

    elif variant == "english":
        query = {
            "multi_match": {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "cross_fields",
                "operator": "or",
            }
        }

    elif variant == "dual":
        query = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": q_simple,
                            "fields": ["title^3", "text"],
                            "type": "cross_fields",
                            "operator": "or",
                            "boost": 1.0,
                        }
                    },
                    {
                        "multi_match": {
                            "query": q_simple,
                            "fields": ["title.en^2", "text.en"],
                            "type": "cross_fields",
                            "operator": "or",
                            "boost": 0.9,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
    else:
        raise ValueError(f"unknown variant={variant}")

    return {
        "size": topk,
        "track_total_hits": False,
        "_source": ["title", "text"],
        "query": query,
    }


def hit_text_from_es_hit(h: Dict[str, Any]) -> str:
    src = h.get("_source", {}) or {}
    return str(src.get("title", "") or "") + "\n" + str(src.get("text", "") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-questions", type=int, default=1000)
    ap.add_argument("--dataset-max-examples", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--light-corpus", required=True)
    ap.add_argument("--light-index", required=True)
    ap.add_argument("--embedding-model", required=True)
    ap.add_argument("--light-device", default="cpu")

    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", default="outputs_probe_200k_analyzer_compare.csv")
    args = ap.parse_args()

    questions = load_questions(
        dataset_path=args.dataset,
        max_examples=args.dataset_max_examples,
        max_questions=args.max_questions,
        seed=args.seed,
    )

    light = FaissHNSWRetriever(
        corpus_path=args.light_corpus,
        index_path=args.light_index,
        embedding_model=args.embedding_model,
        top_n=50,
        device=args.light_device,
    )
    light.load_or_build(rebuild=False)

    variants = [
        ("standard", "adarag_kb_200k_standard"),
        ("english", "adarag_kb_200k_english"),
        ("dual", "adarag_kb_200k_dual"),
    ]

    rows = []

    for i, qa in enumerate(questions, 1):
        dense_docs_obj, _ = light.retrieve_k(qa.q, 50)
        dense_docs = [getattr(d, "text", "") or "" for d in dense_docs_obj]

        dense_hit10 = hit_docs(dense_docs[:10], qa.a)
        dense_hit50 = hit_docs(dense_docs[:50], qa.a)

        for variant, index_name in variants:
            t0 = time.time()
            body = make_query(qa.q, variant, args.topk)
            hits = es_search(args.es_url, index_name, body, timeout=120.0)
            rt = time.time() - t0

            bm25_docs = [hit_text_from_es_hit(h) for h in hits]

            row = {
                "i": i,
                "qid": qa.qid,
                "variant": variant,
                "index_name": index_name,
                "retrieve_time_s": rt,
                "dense_hit10": int(dense_hit10),
                "dense_hit50": int(dense_hit50),
                "bm25_hit10": int(hit_docs(bm25_docs[:10], qa.a)),
                "bm25_hit50": int(hit_docs(bm25_docs[:50], qa.a)),
                "bm25_hit100": int(hit_docs(bm25_docs[:100], qa.a)),
                "union_dense10_bm2510": int(hit_docs(dense_docs[:10] + bm25_docs[:10], qa.a)),
                "union_dense10_bm2550": int(hit_docs(dense_docs[:10] + bm25_docs[:50], qa.a)),
                "union_dense10_bm25100": int(hit_docs(dense_docs[:10] + bm25_docs[:100], qa.a)),
                "bm25_only_100": int((not dense_hit10) and hit_docs(bm25_docs[:100], qa.a)),
            }
            rows.append(row)

        if i % 100 == 0:
            print(f"processed {i}/{len(questions)}")

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("variant")
        .agg(
            n=("i", "count"),
            mean_retrieve_time_s=("retrieve_time_s", "mean"),
            dense_hit10=("dense_hit10", "mean"),
            dense_hit50=("dense_hit50", "mean"),
            bm25_hit10=("bm25_hit10", "mean"),
            bm25_hit50=("bm25_hit50", "mean"),
            bm25_hit100=("bm25_hit100", "mean"),
            union_dense10_bm2510=("union_dense10_bm2510", "mean"),
            union_dense10_bm2550=("union_dense10_bm2550", "mean"),
            union_dense10_bm25100=("union_dense10_bm25100", "mean"),
            bm25_only_100=("bm25_only_100", "mean"),
        )
        .reset_index()
    )

    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary)
    print("\nSaved detail :", args.out)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
