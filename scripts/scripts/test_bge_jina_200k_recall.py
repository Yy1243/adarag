# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever


def norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def hit_docs(docs, answers) -> bool:
    texts = [norm(getattr(d, "text", "") or "") for d in docs]
    for a in answers:
        aa = norm(a)
        if aa and any(aa in t for t in texts):
            return True
    return False


def load_questions(path: str, max_questions: int):
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            ans = obj.get("answer", obj.get("answers", []))
            if isinstance(ans, str):
                ans = [ans]
            out.append((q, [str(x) for x in ans]))
            if len(out) >= max_questions:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-questions", type=int, default=1000)
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs_eval_light_embed_200k_compare.csv")
    args = ap.parse_args()

    models = [
        {
            "name": "bge_200k",
            "index": "/T20050013/adarag_repro/adarag_assets/data/kb/faiss_hnsw_200k_bge.index",
            "model": "/T20050013/adarag_repro/adarag_assets/bge-base-en-v1.5/BAAI/bge-base-en-v1.5",
        },
        {
            "name": "jina_200k",
            "index": "/T20050013/adarag_repro/adarag_assets/data/kb/faiss_hnsw_200k_jina.index",
            "model": "/T20050013/adarag_repro/adarag_assets/model-cache/models/jinaai__jina-embeddings-v2-base-en",
        },
    ]

    questions = load_questions(args.dataset, args.max_questions)
    rows = []

    for mcfg in models:
        name = mcfg["name"]
        print(f"\n=== evaluating {name} ===", flush=True)

        r = FaissHNSWRetriever(
            corpus_path=args.corpus,
            index_path=mcfg["index"],
            embedding_model=mcfg["model"],
            top_n=args.topn,
            device=args.device,
        )
        r.load_or_build(rebuild=False)

        total_t = 0.0
        h10 = h50 = h200 = 0

        for i, (q, ans) in enumerate(questions, 1):
            t0 = time.time()
            docs, _ = r.retrieve_k(q, args.topn)
            total_t += time.time() - t0

            h10 += int(hit_docs(docs[:10], ans))
            h50 += int(hit_docs(docs[:50], ans))
            h200 += int(hit_docs(docs[:200], ans))

            if i % 100 == 0:
                print(f"{name}: {i}/{len(questions)}", flush=True)

        n = len(questions)
        rows.append({
            "model": name,
            "n_questions": n,
            "mean_retrieve_time_s": total_t / max(n, 1),
            "recall_at_10": h10 / max(n, 1),
            "recall_at_50": h50 / max(n, 1),
            "recall_at_200": h200 / max(n, 1),
            "passages": len(r._passages),
            "index_ntotal": r._index.ntotal,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(df)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
