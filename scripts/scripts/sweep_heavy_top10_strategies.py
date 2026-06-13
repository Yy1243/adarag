# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from adarag.utils import load_yaml
from scripts.eval_light_heavy_retrieval_generation import build_light_retriever


# =========================
# 文本与答案匹配
# =========================

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


def docs_hit(docs: List["CandDoc"], answers: List[str], k: int = 10) -> bool:
    for d in docs[:k]:
        if doc_has_answer(d.text, answers):
            return True
    return False


# =========================
# 数据结构
# =========================

@dataclass
class CandDoc:
    key: str
    text: str
    title: str = ""
    doc_id: str = ""

    dense_rank: int = 10**9
    bm25_rank: int = 10**9

    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    source: str = ""


def make_key(text: str, title: str = "") -> str:
    if not title and "\n" in text:
        title = text.split("\n", 1)[0]
    s = norm_loose(title + "\n" + text[:500])
    return s


def get_title_from_text(text: str) -> str:
    if "\n" in text:
        return text.split("\n", 1)[0].strip()
    return ""


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


# =========================
# Dense 检索
# =========================

def retrieve_dense(light, q: str, topn: int) -> Tuple[List[CandDoc], float]:
    t0 = time.time()

    if hasattr(light, "retrieve_k"):
        docs, _ = light.retrieve_k(q, topn)
    else:
        old_topn = getattr(light, "top_n", None)
        try:
            if hasattr(light, "top_n"):
                light.top_n = topn
            docs, _ = light.retrieve(q)
            docs = docs[:topn]
        finally:
            if old_topn is not None and hasattr(light, "top_n"):
                light.top_n = old_topn

    elapsed = time.time() - t0

    out: List[CandDoc] = []
    for i, d in enumerate(docs, 1):
        text = getattr(d, "text", "") or ""
        title = get_title_from_text(text)
        score = float(getattr(d, "score", 0.0) or 0.0)
        key = make_key(text, title)

        out.append(CandDoc(
            key=key,
            text=text,
            title=title,
            dense_rank=i,
            dense_score=score,
            source="dense",
        ))

    return out, elapsed


# =========================
# BM25 检索
# =========================

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

    return {
        "size": size,
        "_source": ["doc_id", "title", "text"],
        "query": {
            "bool": {
                "should": [
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
                ],
                "minimum_should_match": 1,
            }
        },
    }


def retrieve_bm25(
    es_url: str,
    index_name: str,
    q: str,
    topn: int,
    timeout: int,
) -> Tuple[List[CandDoc], float]:
    url = f"{es_url.rstrip('/')}/{index_name}/_search"
    body = es_question_body(q, topn)

    t0 = time.time()
    r = requests.post(url, json=body, timeout=timeout)
    elapsed = time.time() - t0

    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])

    out: List[CandDoc] = []
    for i, h in enumerate(hits, 1):
        src = h.get("_source", {}) or {}
        doc_id = str(src.get("doc_id", h.get("_id", "")))
        title = str(src.get("title", "") or "")
        text = str(src.get("text", "") or "")
        full = (title + "\n" + text).strip() if title else text

        key = make_key(full, title)
        score = float(h.get("_score", 0.0) or 0.0)

        out.append(CandDoc(
            key=key,
            text=full,
            title=title,
            doc_id=doc_id,
            bm25_rank=i,
            bm25_score=score,
            source="bm25",
        ))

    return out, elapsed


# =========================
# 候选融合
# =========================

def merge_candidates(dense_docs: List[CandDoc], bm25_docs: List[CandDoc]) -> List[CandDoc]:
    by_key: Dict[str, CandDoc] = {}

    for d in dense_docs:
        if d.key not in by_key:
            by_key[d.key] = d
        else:
            old = by_key[d.key]
            old.dense_rank = min(old.dense_rank, d.dense_rank)
            old.dense_score = max(old.dense_score, d.dense_score)
            if "dense" not in old.source:
                old.source += "+dense"

    for d in bm25_docs:
        if d.key not in by_key:
            by_key[d.key] = d
        else:
            old = by_key[d.key]
            old.bm25_rank = min(old.bm25_rank, d.bm25_rank)
            old.bm25_score = max(old.bm25_score, d.bm25_score)
            if "bm25" not in old.source:
                old.source += "+bm25"

    return list(by_key.values())


def compute_rrf(
    docs: List[CandDoc],
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 0.35,
):
    for d in docs:
        s = 0.0
        if d.dense_rank < 10**9:
            s += dense_weight / (rrf_k + d.dense_rank)
        if d.bm25_rank < 10**9:
            s += bm25_weight / (rrf_k + d.bm25_rank)
        d.rrf_score = s


def unique_keep_order(docs: List[CandDoc]) -> List[CandDoc]:
    out = []
    seen = set()
    for d in docs:
        if d.key in seen:
            continue
        seen.add(d.key)
        out.append(d)
    return out


def top_dense(docs: List[CandDoc], n: int) -> List[CandDoc]:
    return sorted(
        [d for d in docs if d.dense_rank < 10**9],
        key=lambda x: x.dense_rank,
    )[:n]


def top_bm25(docs: List[CandDoc], n: int) -> List[CandDoc]:
    return sorted(
        [d for d in docs if d.bm25_rank < 10**9],
        key=lambda x: x.bm25_rank,
    )[:n]


def top_rrf(docs: List[CandDoc], n: int) -> List[CandDoc]:
    return sorted(docs, key=lambda x: x.rrf_score, reverse=True)[:n]


def top_rerank(docs: List[CandDoc], n: int) -> List[CandDoc]:
    return sorted(docs, key=lambda x: x.rerank_score, reverse=True)[:n]

def top_rerank_reverse(docs: List[CandDoc], n: int) -> List[CandDoc]:
    """
    反向检查 reranker 分数方向。
    注意：只对真正进入 reranker 的候选排序，排除 rerank_score=-1e9 的未打分文档。
    """
    scored_docs = [d for d in docs if d.rerank_score > -1e8]
    return sorted(scored_docs, key=lambda x: x.rerank_score, reverse=False)[:n]


def fill_to_10(primary: List[CandDoc], fallback: List[CandDoc]) -> List[CandDoc]:
    out = unique_keep_order(primary)
    for d in fallback:
        if len(out) >= 10:
            break
        if d.key not in {x.key for x in out}:
            out.append(d)
    return out[:10]


# =========================
# Reranker
# =========================

class BGEReranker:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 16,
        local_files_only: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, docs: List[CandDoc]) -> List[float]:
        scores: List[float] = []

        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i:i + self.batch_size]
            pairs = [(query, d.text) for d in batch_docs]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logits = self.model(**inputs).logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits[:, 0]
            elif logits.ndim == 2:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits

            scores.extend(batch_scores.detach().float().cpu().numpy().tolist())

        return scores


def rerank_candidates(
    reranker: BGEReranker,
    query: str,
    candidates: List[CandDoc],
):
    if not candidates:
        return
    scores = reranker.score(query, candidates)
    for d, s in zip(candidates, scores):
        d.rerank_score = float(s)


# =========================
# 策略生成
# =========================

def build_strategies(
    docs: List[CandDoc],
    dense_docs: List[CandDoc],
    bm25_docs: List[CandDoc],
) -> Dict[str, List[CandDoc]]:
    """
    注意：docs 里已经带 rerank_score。
    """
    strategies: Dict[str, List[CandDoc]] = {}

    dense10 = top_dense(docs, 10)
    dense20 = top_dense(docs, 20)
    dense50 = top_dense(docs, 50)
    dense100 = top_dense(docs, 100)

    bm25_1 = top_bm25(docs, 1)
    bm25_2 = top_bm25(docs, 2)

    rerank_all = top_rerank(docs, 200)
    rrf_all = top_rrf(docs, 200)

    strategies["dense_top10"] = dense10
    strategies["rerank_union_top10"] = top_rerank(docs, 10)
    strategies["rerank_union_reverse_top10"] = top_rerank_reverse(docs, 10)
    strategies["rrf_top10"] = top_rrf(docs, 10)

    # dense-preserving: 保留 dense 前排，后面用 rerank 补
    strategies["dense6_rerank4"] = fill_to_10(top_dense(docs, 6), rerank_all)
    strategies["dense7_rerank3"] = fill_to_10(top_dense(docs, 7), rerank_all)
    strategies["dense8_rerank2"] = fill_to_10(top_dense(docs, 8), rerank_all)

    # dense + BM25 rescue
    strategies["dense8_bm252"] = fill_to_10(top_dense(docs, 8) + bm25_2, rerank_all)
    strategies["dense9_bm251"] = fill_to_10(top_dense(docs, 9) + bm25_1, rerank_all)

    # dense + rerank + bm25
    strategies["dense6_rerank3_bm251"] = fill_to_10(top_dense(docs, 6) + top_rerank(docs, 3) + bm25_1, dense50)
    strategies["dense7_rerank2_bm251"] = fill_to_10(top_dense(docs, 7) + top_rerank(docs, 2) + bm25_1, dense50)
    strategies["dense8_rerank1_bm251"] = fill_to_10(top_dense(docs, 8) + top_rerank(docs, 1) + bm25_1, dense50)

    # RRF + rerank mixed
    strategies["dense6_rrf2_rerank2"] = fill_to_10(top_dense(docs, 6) + top_rrf(docs, 2) + top_rerank(docs, 2), dense50)
    strategies["dense7_rrf1_rerank2"] = fill_to_10(top_dense(docs, 7) + top_rrf(docs, 1) + top_rerank(docs, 2), dense50)

    # rerank 结果太激进时，保留 dense fallback
    strategies["rerank6_dense4"] = fill_to_10(top_rerank(docs, 6) + top_dense(docs, 4), dense50)
    strategies["rerank8_dense2"] = fill_to_10(top_rerank(docs, 8) + top_dense(docs, 2), dense50)

    # 兜底确保每个策略都是 10 个
    for k, v in list(strategies.items()):
        strategies[k] = fill_to_10(v, dense100)

    return strategies


# =========================
# 主流程
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)

    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--index-name", default="adarag_kb_v2_1m")
    ap.add_argument("--timeout", type=int, default=120)

    ap.add_argument("--max-questions", type=int, default=300)
    ap.add_argument("--dense-topn", type=int, default=200)
    ap.add_argument("--bm25-topn", type=int, default=500)

    # 实际进入 reranker 的候选数量。不要太大，否则很慢。
    ap.add_argument("--rerank-dense-n", type=int, default=100)
    ap.add_argument("--rerank-bm25-n", type=int, default=100)
    ap.add_argument("--rerank-candidate-limit", type=int, default=200)

    ap.add_argument("--reranker-model", required=True)
    ap.add_argument("--reranker-device", default="cuda")
    ap.add_argument("--reranker-batch-size", type=int, default=16)
    ap.add_argument("--reranker-max-length", type=int, default=512)

    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--bm25-weight", type=float, default=0.35)

    ap.add_argument("--out", default="outputs_sweep_heavy_top10_strategies.csv")
    ap.add_argument("--reverse-only", action="store_true")

    args = ap.parse_args()

    cfg = load_yaml(args.config)

    print("[init] loading dense/light retriever...", flush=True)
    light = build_light_retriever(cfg, topn=args.dense_topn)
    print("[init] dense/light retriever loaded.", flush=True)

    print("[init] loading reranker...", flush=True)
    reranker = BGEReranker(
        model_path=args.reranker_model,
        device=args.reranker_device,
        max_length=args.reranker_max_length,
        batch_size=args.reranker_batch_size,
        local_files_only=True,
    )
    print("[init] reranker loaded.", flush=True)

    data = load_questions(args.dataset, args.max_questions)
    print(f"[init] questions loaded: {len(data)}", flush=True)

    strategy_hits: Dict[str, int] = {}
    strategy_counts: Dict[str, int] = {}

    rows: List[Dict[str, Any]] = []

    total_dense_t = 0.0
    total_bm25_t = 0.0
    total_rerank_t = 0.0

    t_all = time.time()

    for i, item in enumerate(data, 1):
        qid = item["qid"]
        q = item["question"]
        answers = item["answers"]

        dense_docs, dense_t = retrieve_dense(light, q, args.dense_topn)
        bm25_docs, bm25_t = retrieve_bm25(
            es_url=args.es_url,
            index_name=args.index_name,
            q=q,
            topn=args.bm25_topn,
            timeout=args.timeout,
        )

        total_dense_t += dense_t
        total_bm25_t += bm25_t

        merged_all = merge_candidates(dense_docs, bm25_docs)
        compute_rrf(
            merged_all,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )

        # 只把较有希望的候选送入 reranker：dense topN + bm25 topN，再按 RRF 截断
        rerank_pool = merge_candidates(
            dense_docs[:args.rerank_dense_n],
            bm25_docs[:args.rerank_bm25_n],
        )
        compute_rrf(
            rerank_pool,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )
        rerank_pool = top_rrf(rerank_pool, args.rerank_candidate_limit)

        t_r0 = time.time()
        rerank_candidates(reranker, q, rerank_pool)
        rerank_t = time.time() - t_r0
        total_rerank_t += rerank_t

        # 把 rerank 分数写回 merged_all
        score_by_key = {d.key: d.rerank_score for d in rerank_pool}
        for d in merged_all:
            if d.key in score_by_key:
                d.rerank_score = score_by_key[d.key]
            else:
                d.rerank_score = -1e9

        strategies = build_strategies(merged_all, dense_docs, bm25_docs)
        if args.reverse_only:
            keep_names = [
                "dense_top10",
                "rerank_union_top10",
                "rerank_union_reverse_top10",
            ]
            strategies = {k: strategies[k] for k in keep_names if k in strategies}

        row = {
            "qid": qid,
            "question": q,
            "answers_json": json.dumps(answers, ensure_ascii=False),
            "dense_time_s": dense_t,
            "bm25_time_s": bm25_t,
            "rerank_time_s": rerank_t,
        }

        for name, docs in strategies.items():
            hit = docs_hit(docs, answers, 10)
            strategy_hits[name] = strategy_hits.get(name, 0) + int(hit)
            strategy_counts[name] = strategy_counts.get(name, 0) + 1
            row[name] = int(hit)

        rows.append(row)

        if i % 20 == 0:
            elapsed = time.time() - t_all
            current = {
                k: strategy_hits[k] / max(strategy_counts[k], 1)
                for k in sorted(strategy_hits)
            }
            top_items = sorted(current.items(), key=lambda x: x[1], reverse=True)[:5]
            top_s = " | ".join([f"{k}={v:.4f}" for k, v in top_items])

            print(
                f"[progress] {i}/{len(data)} elapsed={elapsed:.1f}s "
                f"dense_t={total_dense_t/i:.3f}s bm25_t={total_bm25_t/i:.3f}s rerank_t={total_rerank_t/i:.3f}s "
                f"best: {top_s}",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(data)
    summary = {
        "n_questions": n,
        "dense_topn": args.dense_topn,
        "bm25_topn": args.bm25_topn,
        "rerank_dense_n": args.rerank_dense_n,
        "rerank_bm25_n": args.rerank_bm25_n,
        "rerank_candidate_limit": args.rerank_candidate_limit,
        "mean_dense_time_s": total_dense_t / max(n, 1),
        "mean_bm25_time_s": total_bm25_t / max(n, 1),
        "mean_rerank_time_s": total_rerank_t / max(n, 1),
        "strategies": {
            k: strategy_hits[k] / max(strategy_counts[k], 1)
            for k in sorted(strategy_hits)
        },
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
