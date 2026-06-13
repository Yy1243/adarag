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

import numpy as np
import requests

from adarag.utils import load_yaml
from scripts.eval_light_heavy_retrieval_generation import build_light_retriever


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


def simplify_query(q: str) -> str:
    q = (q or "").lower()
    toks = re.findall(r"[a-z0-9]+", q)
    toks = [t for t in toks if len(t) >= 2 and t not in STOPWORDS]
    return " ".join(toks) if toks else ((q or "").strip())


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
        text = get_doc_text(d)
        if doc_has_answer(text or "", answers):
            return True
    return False


def get_doc_text(d: Any) -> str:
    if isinstance(d, dict):
        return str(d.get("text", "") or "")
    return str(getattr(d, "text", "") or "")


def get_doc_title(d: Any) -> str:
    if isinstance(d, dict):
        return str(d.get("title", "") or "")
    text = get_doc_text(d)
    if "\n" in text:
        return text.split("\n", 1)[0].strip()
    return ""


def doc_key(d: Any) -> str:
    title = get_doc_title(d)
    text = get_doc_text(d)
    return norm_loose(title + "\n" + text[:500])


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


def retrieve_dense(light, q: str, topn: int) -> Tuple[List[Any], float]:
    t0 = time.time()

    if hasattr(light, "retrieve_k"):
        docs, _ = light.retrieve_k(q, topn)
        return docs[:topn], time.time() - t0

    old_topn = getattr(light, "top_n", None)
    try:
        if hasattr(light, "top_n"):
            light.top_n = topn
        docs, _ = light.retrieve(q)
        return docs[:topn], time.time() - t0
    finally:
        if old_topn is not None and hasattr(light, "top_n"):
            light.top_n = old_topn


def es_query_body(q: str, size: int, query_mode: str) -> Dict[str, Any]:
    q_raw = q or ""
    q_simple = simplify_query(q_raw)

    if query_mode == "cross_fields":
        return {
            "size": size,
            "_source": ["doc_id", "title", "text"],
            "query": {
                "multi_match": {
                    "query": q_simple,
                    "fields": ["title^3", "text"],
                    "type": "cross_fields",
                    "operator": "or",
                }
            },
            "track_total_hits": False,
        }

    if query_mode == "bool_phrase_boost":
        return {
            "size": size,
            "_source": ["doc_id", "title", "text"],
            "query": {
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
                            "match_phrase": {
                                "title": {
                                    "query": q_simple,
                                    "slop": 2,
                                    "boost": 4.0,
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "text": {
                                    "query": q_simple,
                                    "slop": 4,
                                    "boost": 2.0,
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": q_raw,
                                "fields": ["title^2", "text"],
                                "type": "best_fields",
                                "boost": 0.3,
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
            "track_total_hits": False,
        }

    raise ValueError(f"unsupported query_mode={query_mode}")


def diversify_by_title(docs: List[Dict[str, Any]], max_per_title: int) -> List[Dict[str, Any]]:
    if max_per_title is None or max_per_title <= 0:
        return docs

    out = []
    cnt: Dict[str, int] = {}
    for d in docs:
        title = str(d.get("title", "") or "").strip()
        if cnt.get(title, 0) >= max_per_title:
            continue
        cnt[title] = cnt.get(title, 0) + 1
        out.append(d)
    return out


def retrieve_bm25(
    sess: requests.Session,
    es_url: str,
    index_name: str,
    q: str,
    topn: int,
    query_mode: str,
    timeout: int,
    max_per_title: int,
) -> Tuple[List[Dict[str, Any]], float]:
    url = f"{es_url.rstrip('/')}/{index_name}/_search"
    body = es_query_body(q, size=topn, query_mode=query_mode)

    t0 = time.time()
    r = sess.post(url, json=body, timeout=(3.0, timeout))
    elapsed = time.time() - t0
    r.raise_for_status()

    hits = r.json().get("hits", {}).get("hits", []) or []
    docs = []
    for h in hits:
        src = h.get("_source", {}) or {}
        title = str(src.get("title", "") or "")
        text = str(src.get("text", "") or "")
        full = (title + "\n" + text).strip() if title else text
        docs.append({
            "doc_id": str(src.get("doc_id", h.get("_id", ""))),
            "title": title,
            "text": full,
            "bm25_score": float(h.get("_score", 0.0) or 0.0),
        })

    docs = diversify_by_title(docs, max_per_title=max_per_title)
    return docs[:topn], elapsed


def weighted_rrf(
    dense_docs: List[Any],
    bm25_docs: List[Dict[str, Any]],
    alpha: float,
    beta: float,
    k: float,
) -> List[Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}

    for rank, d in enumerate(dense_docs, 1):
        key = doc_key(d)
        if not key:
            continue
        if key not in pool:
            pool[key] = {
                "key": key,
                "text": get_doc_text(d),
                "title": get_doc_title(d),
                "dense_rank": None,
                "bm25_rank": None,
                "score": 0.0,
            }
        pool[key]["dense_rank"] = rank
        pool[key]["score"] += alpha * (1.0 / (rank + k))

    for rank, d in enumerate(bm25_docs, 1):
        key = doc_key(d)
        if not key:
            continue
        if key not in pool:
            pool[key] = {
                "key": key,
                "text": get_doc_text(d),
                "title": get_doc_title(d),
                "dense_rank": None,
                "bm25_rank": None,
                "score": 0.0,
            }
        pool[key]["bm25_rank"] = rank
        pool[key]["score"] += beta * (1.0 / (rank + k))

    def sort_key(x: Dict[str, Any]):
        dense_rank = x["dense_rank"] if x["dense_rank"] is not None else 10**9
        bm25_rank = x["bm25_rank"] if x["bm25_rank"] is not None else 10**9
        return (-x["score"], dense_rank, bm25_rank)

    return sorted(pool.values(), key=sort_key)


def union_docs(a: List[Any], b: List[Any]) -> List[Any]:
    out = []
    seen = set()
    for d in a + b:
        key = doc_key(d)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf-config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.rrf_config)
    base_cfg = load_yaml(cfg["base_config"])

    dataset = cfg["dataset"]
    max_questions = int(cfg.get("max_questions", 1000))

    es_url = cfg["es"]["url"]
    index_name = cfg["es"]["index_name"]
    timeout = int(cfg["es"].get("timeout", 120))
    query_mode = cfg["es"].get("query_mode", "cross_fields")
    max_per_title = int(cfg["es"].get("max_per_title", 5))

    dense_topn = int(cfg["retrieval"].get("dense_topn", 200))
    bm25_topn = int(cfg["retrieval"].get("bm25_topn", 300))

    alpha = float(cfg["rrf"].get("alpha", 1.0))
    beta = float(cfg["rrf"].get("beta", 0.4))
    rrf_k = float(cfg["rrf"].get("k", 60))
    final_topk = int(cfg["rrf"].get("final_topk", 10))

    out_csv = cfg["output"]["out_csv"]

    print("[init] loading dense/light retriever...", flush=True)
    light = build_light_retriever(base_cfg, topn=dense_topn)
    print("[init] dense/light retriever loaded.", flush=True)

    print("[init] loading questions...", flush=True)
    questions = load_questions(dataset, max_questions)
    print(f"[init] questions loaded: {len(questions)}", flush=True)

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})

    acc = {
        "dense10": 0,
        "dense50": 0,
        "dense200": 0,
        "bm25_10": 0,
        "bm25_50": 0,
        "bm25_100": 0,
        "bm25_300": 0,
        "rrf_top10": 0,
        "rrf_top20": 0,
        "union_d10_b300": 0,
        "union_d200_b300": 0,
        "bm25_rescue_over_dense10": 0,
        "rrf_rescue_over_dense10": 0,
        "rrf_lost_dense10": 0,
    }

    total_dense_t = 0.0
    total_bm25_t = 0.0
    rows = []
    t_all = time.time()

    for i, item in enumerate(questions, 1):
        qid = item["qid"]
        q = item["question"]
        answers = item["answers"]

        dense_docs, dense_t = retrieve_dense(light, q, dense_topn)
        bm25_docs, bm25_t = retrieve_bm25(
            sess=sess,
            es_url=es_url,
            index_name=index_name,
            q=q,
            topn=bm25_topn,
            query_mode=query_mode,
            timeout=timeout,
            max_per_title=max_per_title,
        )

        total_dense_t += dense_t
        total_bm25_t += bm25_t

        rrf_docs = weighted_rrf(
            dense_docs=dense_docs[:dense_topn],
            bm25_docs=bm25_docs[:bm25_topn],
            alpha=alpha,
            beta=beta,
            k=rrf_k,
        )

        dense10 = docs_hit(dense_docs, answers, 10)
        dense50 = docs_hit(dense_docs, answers, 50)
        dense200 = docs_hit(dense_docs, answers, min(200, dense_topn))

        bm25_10 = docs_hit(bm25_docs, answers, 10)
        bm25_50 = docs_hit(bm25_docs, answers, 50)
        bm25_100 = docs_hit(bm25_docs, answers, min(100, bm25_topn))
        bm25_300 = docs_hit(bm25_docs, answers, min(300, bm25_topn))

        rrf_top10 = docs_hit(rrf_docs, answers, final_topk)
        rrf_top20 = docs_hit(rrf_docs, answers, 20)

        union_d10_b300 = docs_hit(
            union_docs(dense_docs[:10], bm25_docs[:bm25_topn]),
            answers,
            10 + bm25_topn,
        )
        union_d200_b300 = docs_hit(
            union_docs(dense_docs[:dense_topn], bm25_docs[:bm25_topn]),
            answers,
            dense_topn + bm25_topn,
        )

        bm25_rescue = (not dense10) and bm25_300
        rrf_rescue = (not dense10) and rrf_top10
        rrf_lost = dense10 and (not rrf_top10)

        vals = {
            "dense10": dense10,
            "dense50": dense50,
            "dense200": dense200,
            "bm25_10": bm25_10,
            "bm25_50": bm25_50,
            "bm25_100": bm25_100,
            "bm25_300": bm25_300,
            "rrf_top10": rrf_top10,
            "rrf_top20": rrf_top20,
            "union_d10_b300": union_d10_b300,
            "union_d200_b300": union_d200_b300,
            "bm25_rescue_over_dense10": bm25_rescue,
            "rrf_rescue_over_dense10": rrf_rescue,
            "rrf_lost_dense10": rrf_lost,
        }

        for k2, v in vals.items():
            acc[k2] += int(v)

        rows.append({
            "qid": qid,
            "question": q,
            "answers_json": json.dumps(answers, ensure_ascii=False),
            **{k2: int(v) for k2, v in vals.items()},
            "dense_time_s": dense_t,
            "bm25_time_s": bm25_t,
            "rrf_top10_titles": json.dumps([x.get("title", "") for x in rrf_docs[:final_topk]], ensure_ascii=False),
        })

        if i % 50 == 0:
            n = i
            print(
                f"[progress] {i}/{len(questions)} "
                f"d10={acc['dense10']/n:.4f} "
                f"b300={acc['bm25_300']/n:.4f} "
                f"rrf10={acc['rrf_top10']/n:.4f} "
                f"u10+b300={acc['union_d10_b300']/n:.4f} "
                f"rrf_rescue={acc['rrf_rescue_over_dense10']/n:.4f} "
                f"rrf_lost={acc['rrf_lost_dense10']/n:.4f} "
                f"elapsed={time.time()-t_all:.1f}s",
                flush=True,
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(questions)
    summary = {
        "n_questions": n,
        "dataset": dataset,
        "query_mode": query_mode,
        "dense_topn": dense_topn,
        "bm25_topn": bm25_topn,
        "alpha": alpha,
        "beta": beta,
        "rrf_k": rrf_k,
        "final_topk": final_topk,
        "max_per_title": max_per_title,
        "mean_dense_time_s": total_dense_t / max(n, 1),
        "mean_bm25_time_s": total_bm25_t / max(n, 1),
        **{k2: v / max(n, 1) for k2, v in acc.items()},
        "rrf_delta_vs_dense10": (acc["rrf_top10"] - acc["dense10"]) / max(n, 1),
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
