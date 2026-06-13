# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import pandas as pd

from adarag.utils import load_yaml
from adarag.data import QAItem
from adarag.data_hf import load_nq_open_stream
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever
from adarag.retrievers.heavy_es_bm25_rerank import ElasticBM25RerankRetriever


def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def _doc_text(d) -> str:
    return str(getattr(d, "text", "") or "")


def _hit(docs, answers) -> bool:
    texts = [_norm(_doc_text(d)) for d in docs]
    for a in answers:
        aa = _norm(a)
        if aa and any(aa in t for t in texts):
            return True
    return False


def _load_questions(dataset_path: str, max_examples: int, seed: int, max_questions: int):
    items = load_nq_open_stream(
        split="validation",
        local_path=dataset_path,
        max_examples=max_examples,
        seed=seed,
    )
    out = []
    for x in items:
        q = x.get("question") if isinstance(x, dict) else getattr(x, "question", "")
        a = x.get("answer") if isinstance(x, dict) else getattr(x, "answer", [])
        if isinstance(a, str):
            a = [a]
        out.append(QAItem(q=str(q), a=[str(z) for z in a], qid=str(len(out))))
        if len(out) >= max_questions:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-questions", type=int, default=1000)
    ap.add_argument("--dense-topk", type=int, default=10)
    ap.add_argument("--bm25-topk", type=int, default=100)
    ap.add_argument("--out", default="outputs_dense_bm25_complement.csv")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    light_cfg = cfg["light_retriever"]
    ds_cfg = cfg["dataset"]
    heavy_cfg = cfg["heavy_retriever"]

    questions = _load_questions(
        dataset_path=ds_cfg["local_path"],
        max_examples=int(ds_cfg.get("max_examples", args.max_questions)),
        seed=int(cfg.get("seed", 42)),
        max_questions=args.max_questions,
    )

    light = FaissHNSWRetriever(
        corpus_path=light_cfg["corpus_path"],
        index_path=light_cfg["index_path"],
        embedding_model=light_cfg["embedding_model"],
        top_n=max(args.dense_topk, 50),
        device=light_cfg.get("device", "cpu"),
    )
    light.load_or_build(rebuild=bool(light_cfg.get("rebuild", False)))

    bm25 = ElasticBM25RerankRetriever(
        es_url=heavy_cfg.get("es_url", "http://127.0.0.1:9200"),
        index_name=heavy_cfg["index_name"],
        top_n=args.bm25_topk,
        bm25_k=int(heavy_cfg.get("bm25_k", 300)),
        reranker_model=None,
        device="cpu",
        max_per_title=int(heavy_cfg.get("bm25_max_per_title", 5)),
        request_timeout=float(heavy_cfg.get("request_timeout", 120)),
        rerank_k=0,
        profile=False,
    )

    rows = []
    for i, qa in enumerate(questions, 1):
        d_docs, _ = light.retrieve_k(qa.q, max(args.dense_topk, 50))
        b_docs, _ = bm25.retrieve(qa.q)

        dense10 = d_docs[:10]
        dense50 = d_docs[:50]
        bm25_10 = b_docs[:10]
        bm25_50 = b_docs[:50]
        bm25_100 = b_docs[:100]

        row = {
            "i": i,
            "question": qa.q,
            "dense_hit10": int(_hit(dense10, qa.a)),
            "dense_hit50": int(_hit(dense50, qa.a)),
            "bm25_hit10": int(_hit(bm25_10, qa.a)),
            "bm25_hit50": int(_hit(bm25_50, qa.a)),
            "bm25_hit100": int(_hit(bm25_100, qa.a)),
            "union_dense10_bm2510": int(_hit(dense10 + bm25_10, qa.a)),
            "union_dense10_bm2550": int(_hit(dense10 + bm25_50, qa.a)),
            "union_dense10_bm25100": int(_hit(dense10 + bm25_100, qa.a)),
            "bm25_only_100": int((not _hit(dense10, qa.a)) and _hit(bm25_100, qa.a)),
        }
        rows.append(row)

        if i % 100 == 0:
            print(f"processed {i}/{len(questions)}")

    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "dense_hit10": df["dense_hit10"].mean(),
        "dense_hit50": df["dense_hit50"].mean(),
        "bm25_hit10": df["bm25_hit10"].mean(),
        "bm25_hit50": df["bm25_hit50"].mean(),
        "bm25_hit100": df["bm25_hit100"].mean(),
        "union_dense10_bm2510": df["union_dense10_bm2510"].mean(),
        "union_dense10_bm2550": df["union_dense10_bm2550"].mean(),
        "union_dense10_bm25100": df["union_dense10_bm25100"].mean(),
        "bm25_only_100": df["bm25_only_100"].mean(),
    }

    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(args.out.replace(".csv", "_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(pd.DataFrame([summary]))
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()