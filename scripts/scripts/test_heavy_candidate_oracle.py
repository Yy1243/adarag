# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from adarag.utils import load_yaml
from scripts.eval_light_heavy_retrieval_generation import build_light_retriever


def norm_basic(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_loose(s: str) -> str:
    s = norm_basic(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def doc_has_answer(text: str, answers: List[str]) -> bool:
    db = norm_basic(text)
    dl = norm_loose(text)

    for ans in answers:
        ab = norm_basic(ans)
        al = norm_loose(ans)

        if ab and ab in db:
            return True

        if al:
            if " " in al:
                if al in dl:
                    return True
            else:
                if re.search(rf"\b{re.escape(al)}\b", dl) is not None:
                    return True

    return False


def docs_hit(docs: List[Any], answers: List[str], k: int) -> bool:
    for d in docs[:k]:
        text = getattr(d, "text", "") if not isinstance(d, dict) else d.get("text", "")
        if doc_has_answer(text or "", answers):
            return True
    return False


def load_questions(path: str, max_questions: int) -> List[Dict[str, Any]]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question", obj.get("q", ""))
            ans = obj.get("answer", obj.get("answers", []))
            if isinstance(ans, str):
                ans = [ans]
            ans = [str(x) for x in ans if str(x).strip()]
            out.append({
                "qid": obj.get("qid", obj.get("id", i)),
                "question": str(q),
                "answers": ans,
            })
            if len(out) >= max_questions:
                break
    return out


STOPWORDS = {
    "what", "who", "when", "where", "which", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "the", "a", "an", "of", "in", "on",
    "to", "for", "with", "by", "from", "at", "as", "and",
    "or", "that", "this", "these", "those", "it", "its",
    "after", "before", "into", "about", "there",
}


def simplify_query(q: str) -> str:
    qn = norm_loose(q)
    toks = [t for t in qn.split() if t not in STOPWORDS]
    return " ".join(toks) if toks else q


def es_question_body(q: str, size: int) -> Dict[str, Any]:
    qs = simplify_query(q)

    should = [
        {
            "multi_match": {
                "query": q,
                "type": "cross_fields",
                "fields": ["title^3", "text"],
                "operator": "or",
                "minimum_should_match": "60%",
                "boost": 2.0,
            }
        },
        {
            "multi_match": {
                "query": qs,
                "type": "cross_fields",
                "fields": ["title^4", "text"],
                "operator": "or",
                "minimum_should_match": "50%",
                "boost": 2.5,
            }
        },
        {
            "match_phrase": {
                "title": {
                    "query": qs,
                    "slop": 2,
                    "boost": 5.0,
                }
            }
        },
        {
            "match_phrase": {
                "text": {
                    "query": qs,
                    "slop": 3,
                    "boost": 1.5,
                }
            }
        },
    ]

    return {
        "size": size,
        "_source": ["doc_id", "title", "text"],
        "query": {
            "bool": {
                "should": should,
                "minimum_should_match": 1,
            }
        },
    }


def es_search_question(
    es_url: str,
    index_name: str,
    q: str,
    size: int,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], float]:
    url = f"{es_url.rstrip('/')}/{index_name}/_search"
    body = es_question_body(q, size=size)

    t0 = time.time()
    r = requests.post(url, json=body, timeout=timeout)
    elapsed = time.time() - t0
    r.raise_for_status()

    hits = r.json().get("hits", {}).get("hits", [])

    docs = []
    for h in hits:
        src = h.get("_source", {}) or {}
        title = str(src.get("title", "") or "")
        text = str(src.get("text", "") or "")
        doc_id = str(src.get("doc_id", h.get("_id", "")))
        full = (title + "\n" + text).strip() if title else text
        docs.append({
            "doc_id": doc_id,
            "title": title,
            "text": full,
            "score": h.get("_score", 0.0),
        })

    return docs, elapsed


def doc_key(d: Any) -> str:
    if isinstance(d, dict):
        title = d.get("title", "")
        text = d.get("text", "")
    else:
        text = getattr(d, "text", "") or ""
        if "\n" in text:
            title = text.split("\n", 1)[0]
        else:
            title = ""
    return norm_loose(str(title) + "\n" + str(text[:500]))


def union_docs(dense_docs: List[Any], bm25_docs: List[Dict[str, Any]]) -> List[Any]:
    out = []
    seen = set()

    for d in dense_docs + bm25_docs:
        k = doc_key(d)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(d)

    return out


def retrieve_dense(light, q: str, topn: int) -> List[Any]:
    if hasattr(light, "retrieve_k"):
        docs, _ = light.retrieve_k(q, topn)
        return docs

    old_topn = getattr(light, "top_n", None)
    try:
        if hasattr(light, "top_n"):
            light.top_n = topn
        docs, _ = light.retrieve(q)
        return docs[:topn]
    finally:
        if old_topn is not None and hasattr(light, "top_n"):
            light.top_n = old_topn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--index-name", default="adarag_kb_v2_1m")
    ap.add_argument("--max-questions", type=int, default=1000)
    ap.add_argument("--dense-topn", type=int, default=200)
    ap.add_argument("--bm25-topn", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="outputs_heavy_candidate_oracle.csv")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    print("[init] loading dense/light retriever...", flush=True)
    light = build_light_retriever(cfg, topn=args.dense_topn)
    print("[init] dense/light retriever loaded.", flush=True)

    print("[init] loading questions...", flush=True)
    data = load_questions(args.dataset, args.max_questions)
    print(f"[init] questions loaded: {len(data)}", flush=True)

    rows = []

    acc = {
        "dense10": 0,
        "dense50": 0,
        "dense100": 0,
        "dense200": 0,
        "bm25_10": 0,
        "bm25_50": 0,
        "bm25_100": 0,
        "bm25_300": 0,
        "bm25_500": 0,
        "union_d10_b500": 0,
        "union_d50_b500": 0,
        "union_d100_b500": 0,
        "union_d200_b500": 0,
        "bm25_rescue_over_dense10": 0,
        "bm25_rescue_over_dense50": 0,
    }

    total_dense_t = 0.0
    total_bm25_t = 0.0

    t_all = time.time()

    for i, item in enumerate(data, 1):
        qid = item["qid"]
        q = item["question"]
        answers = item["answers"]

        t0 = time.time()
        dense_docs = retrieve_dense(light, q, args.dense_topn)
        dense_t = time.time() - t0

        bm25_docs, bm25_t = es_search_question(
            es_url=args.es_url,
            index_name=args.index_name,
            q=q,
            size=args.bm25_topn,
            timeout=args.timeout,
        )

        total_dense_t += dense_t
        total_bm25_t += bm25_t

        dense10 = docs_hit(dense_docs, answers, 10)
        dense50 = docs_hit(dense_docs, answers, 50)
        dense100 = docs_hit(dense_docs, answers, 100)
        dense200 = docs_hit(dense_docs, answers, min(200, args.dense_topn))

        bm25_10 = docs_hit(bm25_docs, answers, 10)
        bm25_50 = docs_hit(bm25_docs, answers, 50)
        bm25_100 = docs_hit(bm25_docs, answers, 100)
        bm25_300 = docs_hit(bm25_docs, answers, min(300, args.bm25_topn))
        bm25_500 = docs_hit(bm25_docs, answers, min(500, args.bm25_topn))

        union_d10_b500 = docs_hit(
            union_docs(dense_docs[:10], bm25_docs[:args.bm25_topn]),
            answers,
            10 + args.bm25_topn,
        )

        union_d50_b500 = docs_hit(
            union_docs(dense_docs[:50], bm25_docs[:args.bm25_topn]),
            answers,
            50 + args.bm25_topn,
        )

        union_d100_b500 = docs_hit(
            union_docs(dense_docs[:100], bm25_docs[:args.bm25_topn]),
            answers,
            100 + args.bm25_topn,
        )

        union_d200_b500 = docs_hit(
            union_docs(dense_docs[:args.dense_topn], bm25_docs[:args.bm25_topn]),
            answers,
            args.dense_topn + args.bm25_topn,
        )

        rescue10 = (not dense10) and bm25_500
        rescue50 = (not dense50) and bm25_500

        for key, val in [
            ("dense10", dense10),
            ("dense50", dense50),
            ("dense100", dense100),
            ("dense200", dense200),
            ("bm25_10", bm25_10),
            ("bm25_50", bm25_50),
            ("bm25_100", bm25_100),
            ("bm25_300", bm25_300),
            ("bm25_500", bm25_500),
            ("union_d10_b500", union_d10_b500),
            ("union_d50_b500", union_d50_b500),
            ("union_d100_b500", union_d100_b500),
            ("union_d200_b500", union_d200_b500),
            ("bm25_rescue_over_dense10", rescue10),
            ("bm25_rescue_over_dense50", rescue50),
        ]:
            acc[key] += int(val)

        rows.append({
            "qid": qid,
            "question": q,
            "answers_json": json.dumps(answers, ensure_ascii=False),
            "dense10": int(dense10),
            "dense50": int(dense50),
            "dense100": int(dense100),
            "dense200": int(dense200),
            "bm25_10": int(bm25_10),
            "bm25_50": int(bm25_50),
            "bm25_100": int(bm25_100),
            "bm25_300": int(bm25_300),
            "bm25_500": int(bm25_500),
            "union_d10_b500": int(union_d10_b500),
            "union_d50_b500": int(union_d50_b500),
            "union_d100_b500": int(union_d100_b500),
            "union_d200_b500": int(union_d200_b500),
            "bm25_rescue_over_dense10": int(rescue10),
            "bm25_rescue_over_dense50": int(rescue50),
            "dense_time_s": dense_t,
            "bm25_time_s": bm25_t,
        })

        if i % 50 == 0:
            n = i
            print(
                f"[progress] {i}/{len(data)} "
                f"d10={acc['dense10']/n:.4f} d50={acc['dense50']/n:.4f} d200={acc['dense200']/n:.4f} "
                f"b100={acc['bm25_100']/n:.4f} b500={acc['bm25_500']/n:.4f} "
                f"u10+500={acc['union_d10_b500']/n:.4f} "
                f"u200+500={acc['union_d200_b500']/n:.4f} "
                f"r10={acc['bm25_rescue_over_dense10']/n:.4f} "
                f"elapsed={time.time()-t_all:.1f}s",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(data)
    summary = {
        "n_questions": n,
        "dense_topn": args.dense_topn,
        "bm25_topn": args.bm25_topn,
        "mean_dense_time_s": total_dense_t / max(n, 1),
        "mean_bm25_time_s": total_bm25_t / max(n, 1),
        **{k: v / max(n, 1) for k, v in acc.items()},
        "out_csv": str(out_path),
        "elapsed_s": time.time() - t_all,
    }

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved CSV:", out_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
