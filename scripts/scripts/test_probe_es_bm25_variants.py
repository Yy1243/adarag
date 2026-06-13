# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

from adarag.data_hf import load_nq_open_stream


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
}


def norm_text(x: str) -> str:
    x = (x or "").lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def contains_any(text: str, answers: List[str]) -> bool:
    t = norm_text(text)
    for a in answers or []:
        a = norm_text(str(a))
        if a and a in t:
            return True
    return False


def simplify_query(q: str) -> str:
    toks = re.findall(r"[a-z0-9]+", (q or "").lower())
    toks = [t for t in toks if t not in STOP and len(t) >= 2]
    return " ".join(toks) if toks else (q or "")


def get_src_text(src: Dict[str, Any]) -> str:
    return str(
        src.get("text")
        or src.get("contents")
        or src.get("passage")
        or src.get("document")
        or ""
    )


def strip_prompt_title_prefix(text: str, title: str) -> str:
    """
    仅用于诊断/后处理，不影响 ES 打分。
    只删除 text 开头重复的一次 title。
    """
    text = str(text or "").strip()
    title = str(title or "").strip()

    if not text or not title:
        return text

    if not text.lower().startswith(title.lower()):
        return text

    after = text[len(title):]

    if after and re.match(r"^[A-Za-z0-9]", after):
        return text

    return after.lstrip(" \n\t\r:：-—.,;；|").strip()


def text_starts_with_title(text: str, title: str) -> bool:
    text = str(text or "").strip()
    title = str(title or "").strip()
    if not text or not title:
        return False
    return text.lower().startswith(title.lower())


def hit_to_doc(hit: Dict[str, Any]) -> Dict[str, Any]:
    src = hit.get("_source", {}) or {}

    title = str(src.get("title", "") or "").strip()
    text = get_src_text(src).strip()
    text_for_prompt = strip_prompt_title_prefix(text, title)

    # full_text 用于 oracle recall：保留 title + 原始 text。
    # 因为最终 prompt 也会显式给 Title。
    full_text = (title + "\n" + text).strip() if title else text

    return {
        "doc_id": str(src.get("doc_id", hit.get("_id", ""))),
        "title": title,
        "text": text,
        "text_for_prompt": text_for_prompt,
        "full_text": full_text,
        "score": float(hit.get("_score", 0.0) or 0.0),
        "starts_with_title": int(text_starts_with_title(text, title)),
    }


def es_search(es_url: str, index_name: str, body: Dict[str, Any], timeout: float = 120.0) -> List[Dict[str, Any]]:
    url = f"{es_url.rstrip('/')}/{index_name}/_search"
    r = requests.post(url, json=body, timeout=(3.0, timeout))
    if r.status_code >= 400:
        print("ES ERROR", r.status_code)
        print(json.dumps(body, ensure_ascii=False)[:3000])
        print(r.text[:3000])
        r.raise_for_status()
    return (r.json().get("hits", {}) or {}).get("hits", []) or []


def build_query(q: str, mode: str, size: int) -> Dict[str, Any]:
    q_raw = q or ""
    q_simple = simplify_query(q_raw)

    if mode == "title_text_best_fields":
        query = {
            "multi_match": {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "best_fields",
            }
        }

    elif mode == "title_text_cross_fields":
        query = {
            "multi_match": {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "cross_fields",
                "operator": "or",
            }
        }

    elif mode == "title_text_strict_msm":
        query = {
            "multi_match": {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "best_fields",
                "minimum_should_match": "2<75%",
            }
        }

    elif mode == "text_only_simple":
        query = {
            "match": {
                "text": {
                    "query": q_simple,
                    "operator": "or",
                }
            }
        }

    elif mode == "text_only_raw":
        query = {
            "match": {
                "text": {
                    "query": q_raw,
                    "operator": "or",
                }
            }
        }

    elif mode == "text_only_strict_msm":
        query = {
            "match": {
                "text": {
                    "query": q_simple,
                    "operator": "or",
                    "minimum_should_match": "2<75%",
                }
            }
        }

    elif mode == "text_only_demote_title_prefix":
        # ES 先按 text_only_simple 召回，后面 Python 再轻微降权。
        query = {
            "match": {
                "text": {
                    "query": q_simple,
                    "operator": "or",
                }
            }
        }

    else:
        raise ValueError(f"unknown mode={mode}")

    return {
        "size": size,
        "track_total_hits": False,
        "_source": ["doc_id", "title", "text", "contents", "passage", "document"],
        "query": query,
    }


def maybe_postprocess_docs(docs: List[Dict[str, Any]], mode: str, demote_penalty: float) -> List[Dict[str, Any]]:
    if mode != "text_only_demote_title_prefix":
        return docs

    out = []
    for d in docs:
        dd = dict(d)
        score = float(dd.get("score", 0.0))
        if int(dd.get("starts_with_title", 0)) == 1:
            score -= float(demote_penalty)
        dd["adjusted_score"] = score
        out.append(dd)

    out.sort(key=lambda x: x.get("adjusted_score", x.get("score", 0.0)), reverse=True)
    return out


def eval_mode(
    items,
    es_url: str,
    index_name: str,
    mode: str,
    topk: int,
    candidate_k: int,
    timeout: float,
    demote_penalty: float,
):
    rows = []
    t0_all = time.time()

    # 如果需要后处理重排，ES 先多取一些候选。
    es_size = max(topk, candidate_k) if mode == "text_only_demote_title_prefix" else topk

    for i, qa in enumerate(items, 1):
        body = build_query(qa.q, mode, es_size)

        t0 = time.time()
        hits = es_search(es_url, index_name, body, timeout=timeout)
        rt = time.time() - t0

        docs = [hit_to_doc(h) for h in hits]
        docs = maybe_postprocess_docs(docs, mode=mode, demote_penalty=demote_penalty)
        docs = docs[:topk]

        hit10 = any(contains_any(d["full_text"], qa.a) for d in docs[:10])
        hit50 = any(contains_any(d["full_text"], qa.a) for d in docs[:50])
        hit100 = any(contains_any(d["full_text"], qa.a) for d in docs[:100])

        prefix_rate10 = sum(int(d["starts_with_title"]) for d in docs[:10]) / max(1, min(10, len(docs)))
        prefix_rate50 = sum(int(d["starts_with_title"]) for d in docs[:50]) / max(1, min(50, len(docs)))

        rows.append({
            "mode": mode,
            "i": i,
            "retrieve_time_s": rt,
            "n_docs": len(docs),
            "recall_at_10": int(hit10),
            "recall_at_50": int(hit50),
            "recall_at_100": int(hit100),
            "prefix_title_rate_at_10": prefix_rate10,
            "prefix_title_rate_at_50": prefix_rate50,
        })

        if i % 100 == 0:
            print(f"[{mode}] {i}/{len(items)}")

    df = pd.DataFrame(rows)
    summary = {
        "mode": mode,
        "n_questions": len(items),
        "mean_retrieve_time_s": df["retrieve_time_s"].mean(),
        "mean_n_docs": df["n_docs"].mean(),
        "recall_at_10": df["recall_at_10"].mean(),
        "recall_at_50": df["recall_at_50"].mean(),
        "recall_at_100": df["recall_at_100"].mean(),
        "prefix_title_rate_at_10": df["prefix_title_rate_at_10"].mean(),
        "prefix_title_rate_at_50": df["prefix_title_rate_at_50"].mean(),
        "total_time_s": time.time() - t0_all,
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--index-name", default="adarag_kb_v2_1m")
    ap.add_argument("--max-questions", type=int, default=1000)
    ap.add_argument("--dataset-max-examples", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--candidate-k", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--demote-penalty", type=float, default=0.25)
    ap.add_argument("--out", default="outputs_bm25_textonly_variant_probe.csv")
    args = ap.parse_args()

    items = load_nq_open_stream(
        split="validation",
        local_path=args.dataset,
        max_examples=args.dataset_max_examples,
        seed=args.seed,
    )
    items = items[: args.max_questions]

    modes = [
        "title_text_best_fields",
        "title_text_cross_fields",
        "title_text_strict_msm",
        "text_only_raw",
        "text_only_simple",
        "text_only_strict_msm",
        "text_only_demote_title_prefix",
    ]

    all_details = []
    summaries = []

    for mode in modes:
        detail, summary = eval_mode(
            items=items,
            es_url=args.es_url,
            index_name=args.index_name,
            mode=mode,
            topk=args.topk,
            candidate_k=args.candidate_k,
            timeout=args.timeout,
            demote_penalty=args.demote_penalty,
        )
        all_details.append(detail)
        summaries.append(summary)

    detail_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detail_path = args.out.replace(".csv", "_details.csv")
    summary_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary_df)
    print("\nSaved summary:", args.out)
    print("Saved details:", detail_path)


if __name__ == "__main__":
    main()