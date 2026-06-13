# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import time
import re
from typing import Any, List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

from adarag.utils import load_yaml
from adarag.data_hf import load_nq_open_stream
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever
from adarag.retrievers.heavy_es_bm25_rerank import ElasticBM25RerankRetriever
from adarag.retrievers.heavy_hybrid_dense_bm25 import HybridDenseBM25Retriever
from adarag.retrievers.heavy_bm25 import HeavyBM25Retriever


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _get_field(x: Any, names: List[str], default=None):
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
        return [str(x) for x in a if x is not None]
    return [str(a)]


def _norm_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = " ".join(s.split())
    return s


def _doc_text(d: Any) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        for key in ["text", "contents", "passage", "document", "body"]:
            if key in d and d[key] is not None:
                return str(d[key])
        return str(d)
    for attr in ["text", "contents", "passage", "document", "body"]:
        if hasattr(d, attr):
            v = getattr(d, attr)
            if v is not None:
                return str(v)
    return str(d)


def hit_at_k(docs: List[Any], answers: List[str], k: int) -> int:
    if not answers:
        return 0

    texts = [_norm_text(_doc_text(d)) for d in docs[:k]]
    ans_norms = [_norm_text(a) for a in answers if str(a).strip()]

    for ans in ans_norms:
        if not ans:
            continue
        for txt in texts:
            if ans in txt:
                return 1
    return 0


def load_questions(cfg: dict, max_questions: int | None):
    ds_cfg = cfg["dataset"]
    seed = int(ds_cfg.get("seed", cfg.get("seed", 42)))

    it = load_nq_open_stream(
        split=ds_cfg.get("split", "validation"),
        seed=seed,
        local_path=ds_cfg["local_path"],
        max_examples=ds_cfg.get("max_examples", None),
    )

    rows = []
    for idx, item in enumerate(it):
        q = _get_field(item, ["question", "q", "query"], "")
        a = _get_field(item, ["answer", "answers", "a", "gold"], [])
        qid = _get_field(item, ["qid", "id", "example_id"], f"q{idx}")

        rows.append(
            {
                "qid": str(qid),
                "question": str(q),
                "answers": _as_answers(a),
                "dataset_index": idx,
            }
        )

        if max_questions is not None and len(rows) >= max_questions:
            break

    return rows


def build_light_retriever(cfg: dict, topn: int):
    light_cfg = cfg["light_retriever"]

    light = FaissHNSWRetriever(
        corpus_path=light_cfg["corpus_path"],
        index_path=light_cfg["index_path"],
        embedding_model=light_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"),
        top_n=topn,
        device=light_cfg.get("device", "cpu"),
    )

    light.load_or_build(
        rebuild=bool(light_cfg.get("rebuild", False)),
        max_passages=light_cfg.get("max_passages", None),
    )

    return light


def build_heavy_retriever(cfg: dict, force_topn: int):
    heavy_cfg = cfg["heavy_retriever"]
    heavy_type = heavy_cfg.get("type", "bm25_es")

    if heavy_type == "bm25_es":
        return ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://127.0.0.1:9200"),
            index_name=heavy_cfg["index_name"],
            top_n=force_topn,
            bm25_k=int(heavy_cfg.get("bm25_k", 50)),
            reranker_model=heavy_cfg.get("reranker_model", None),
            device=heavy_cfg.get("device", "cpu"),
            collapse_field=heavy_cfg.get("collapse_field", None),
            max_per_title=int(heavy_cfg.get("max_per_title", 5)),
            minimum_should_match=heavy_cfg.get("minimum_should_match", None),
            request_timeout=float(heavy_cfg.get("request_timeout", 120)),
            rerank_k=heavy_cfg.get("rerank_k", 0),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 8)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 512)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

    if heavy_type == "bm25_local":
        return HeavyBM25Retriever(
            corpus_path=heavy_cfg["corpus_path"],
            top_n=force_topn,
            max_passages=heavy_cfg.get("max_passages", None),
            candidate_k=heavy_cfg.get("candidate_k", 200),
            use_rerank=heavy_cfg.get("use_rerank", True),
        )

    if heavy_type == "hybrid_dense_bm25":
        # hybrid 需要 dense/light 分支
        dense_topn = int(heavy_cfg.get("dense_top_n", 50))
        light_for_hybrid = build_light_retriever(cfg, topn=dense_topn)

        bm25 = ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://127.0.0.1:9200"),
            index_name=heavy_cfg["index_name"],
            top_n=int(heavy_cfg.get("bm25_top_n", force_topn)),
            bm25_k=int(heavy_cfg.get("bm25_k", 80)),
            reranker_model=heavy_cfg.get("reranker_model", None),
            device=heavy_cfg.get("device", "cpu"),
            collapse_field=heavy_cfg.get("collapse_field", None),
            max_per_title=int(heavy_cfg.get("max_per_title", 5)),
            minimum_should_match=heavy_cfg.get("minimum_should_match", None),
            request_timeout=float(heavy_cfg.get("request_timeout", 120)),
            rerank_k=heavy_cfg.get("rerank_k", 0),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 8)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 256)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

        return HybridDenseBM25Retriever(
            dense_retriever=light_for_hybrid,
            bm25_retriever=bm25,
            dense_top_n=int(heavy_cfg.get("dense_top_n", 50)),
            bm25_top_n=int(heavy_cfg.get("bm25_top_n", 80)),
            out_top_n=force_topn,
            merge_mode=str(heavy_cfg.get("merge_mode", "rrf")).strip().lower(),
            dense_keep_n=int(heavy_cfg.get("dense_keep_n", 50)),
            bm25_keep_n=int(heavy_cfg.get("bm25_keep_n", 50)),
            rrf_k=int(heavy_cfg.get("rrf_k", 60)),
            simplify_bm25=bool(heavy_cfg.get("simplify_bm25", False)),
            dedup_key_mode=str(heavy_cfg.get("dedup_key_mode", "title_text")).strip().lower(),
            score_fuse_norm=str(heavy_cfg.get("score_fuse_norm", "minmax")).strip().lower(),
            score_fuse_w_dense=float(heavy_cfg.get("score_fuse_w_dense", 0.8)),
            score_fuse_w_bm25=float(heavy_cfg.get("score_fuse_w_bm25", 0.2)),
            rewrite_mode=str(heavy_cfg.get("rewrite_mode", "none")).strip().lower(),
            rewrite_prf_docs=int(heavy_cfg.get("rewrite_prf_docs", 5)),
            rewrite_prf_terms=int(heavy_cfg.get("rewrite_prf_terms", 8)),
            rewrite_doc_max_chars=int(heavy_cfg.get("rewrite_doc_max_chars", 300)),
            rewrite_min_df=int(heavy_cfg.get("rewrite_min_df", 2)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

    raise ValueError(f"Unknown heavy_retriever.type = {heavy_type}")


def eval_one_retriever(
    label: str,
    retriever,
    questions: List[Dict[str, Any]],
    k_list: List[int],
):
    rows = []

    for item in tqdm(questions, desc=f"Eval {label}", unit="q"):
        qid = item["qid"]
        q = item["question"]
        answers = item["answers"]

        t0 = time.time()
        docs, _scores = retriever.retrieve(q)
        retrieve_time_s = time.time() - t0

        row = {
            "label": label,
            "qid": qid,
            "question": q,
            "retrieve_time_s": retrieve_time_s,
        }

        for k in k_list:
            row[f"hit_at_{k}"] = hit_at_k(docs, answers, k)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", required=True, type=str, help="old bm25_es config")
    ap.add_argument("--config-b", required=True, type=str, help="new hybrid_dense_bm25 config")
    ap.add_argument("--label-a", default="bm25_es_old", type=str)
    ap.add_argument("--label-b", default="hybrid_dense_prf", type=str)
    ap.add_argument("--max-questions", default=500, type=int)
    ap.add_argument("--topn", default=10, type=int)
    ap.add_argument("--k-list", default="1,3,5,10", type=str)
    ap.add_argument("--output-dir", default="outputs_compare_heavy_recall", type=str)
    args = ap.parse_args()

    _ensure_dir(args.output_dir)

    cfg_a = load_yaml(args.config_a)
    cfg_b = load_yaml(args.config_b)

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    topn = max(int(args.topn), max(k_list))

    # 用 config-a 的 dataset，保证两套检索器在同一批问题上比较
    questions = load_questions(cfg_a, max_questions=args.max_questions)

    print(f"Loaded questions: {len(questions)}")
    print(f"k_list = {k_list}, topn = {topn}")

    print("\nBuilding retriever A...")
    retriever_a = build_heavy_retriever(cfg_a, force_topn=topn)
    retriever_a.retrieve("warmup query")

    print("\nBuilding retriever B...")
    retriever_b = build_heavy_retriever(cfg_b, force_topn=topn)
    retriever_b.retrieve("warmup query")

    df_a = eval_one_retriever(args.label_a, retriever_a, questions, k_list)
    df_b = eval_one_retriever(args.label_b, retriever_b, questions, k_list)

    detailed = pd.concat([df_a, df_b], ignore_index=True)

    summary_rows = []
    for label, g in detailed.groupby("label"):
        row = {
            "label": label,
            "n_questions": len(g),
            "mean_retrieve_time_s": g["retrieve_time_s"].mean(),
            "median_retrieve_time_s": g["retrieve_time_s"].median(),
            "p95_retrieve_time_s": g["retrieve_time_s"].quantile(0.95),
        }
        for k in k_list:
            row[f"recall_at_{k}"] = g[f"hit_at_{k}"].mean()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # paired delta
    a_small = df_a[["qid", "retrieve_time_s"] + [f"hit_at_{k}" for k in k_list]].copy()
    b_small = df_b[["qid", "retrieve_time_s"] + [f"hit_at_{k}" for k in k_list]].copy()

    paired = a_small.merge(
        b_small,
        on="qid",
        suffixes=(f"_{args.label_a}", f"_{args.label_b}"),
    )

    paired["delta_time_b_minus_a"] = (
        paired[f"retrieve_time_s_{args.label_b}"]
        - paired[f"retrieve_time_s_{args.label_a}"]
    )

    paired_summary = {
        "n_questions": len(paired),
        "mean_delta_time_b_minus_a": paired["delta_time_b_minus_a"].mean(),
        "median_delta_time_b_minus_a": paired["delta_time_b_minus_a"].median(),
        "b_slower_ratio": (paired["delta_time_b_minus_a"] > 0).mean(),
    }

    for k in k_list:
        paired[f"delta_hit_at_{k}_b_minus_a"] = (
            paired[f"hit_at_{k}_{args.label_b}"]
            - paired[f"hit_at_{k}_{args.label_a}"]
        )
        paired_summary[f"delta_recall_at_{k}_b_minus_a"] = paired[
            f"delta_hit_at_{k}_b_minus_a"
        ].mean()

    detailed_path = os.path.join(args.output_dir, "heavy_recall_detailed.csv")
    summary_path = os.path.join(args.output_dir, "heavy_recall_summary.csv")
    paired_path = os.path.join(args.output_dir, "heavy_recall_paired.csv")
    paired_summary_path = os.path.join(args.output_dir, "heavy_recall_paired_summary.csv")

    detailed.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    summary.round(4).to_csv(summary_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([paired_summary]).round(4).to_csv(
        paired_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Summary ===")
    print(summary.round(4).to_string(index=False))

    print("\n=== Paired Summary: B - A ===")
    print(pd.DataFrame([paired_summary]).round(4).to_string(index=False))

    print("\nSaved:")
    print(detailed_path)
    print(summary_path)
    print(paired_path)
    print(paired_summary_path)


if __name__ == "__main__":
    main()