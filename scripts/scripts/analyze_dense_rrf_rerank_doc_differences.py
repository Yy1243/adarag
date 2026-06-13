# -*- coding: utf-8 -*-
"""
analyze_dense_rrf_rerank_doc_differences.py

用途：
分析 dense_top10 / rrf_top10 / Qwen3-rerank_top10 的文档差异，
用于解释为什么 rerank 提高 answer-string recall，却不一定提高 generation accuracy。

输出：
1) *.diff_cases.csv
   每个问题一行，给出三条路径的 hit、first_hit_rank、overlap、case_type 等。
2) *.diff_docs.jsonl
   每个问题一条 JSON，包含 dense/rrf/rerank top10 的详细文档、分数、rank、answer_hit、答案附近 snippet。
3) *.diff_summary.json
   总体统计。

运行：
PYTHONPATH=. python scripts/analyze_dense_rrf_rerank_doc_differences.py \
  --rrf-config scripts/config_weighted_rrf_top100_rerank_recall.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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


def get_rerank_text(d: Any) -> str:
    title = get_doc_title(d).strip()
    text = get_doc_text(d).strip()
    if title and text:
        if not norm_loose(text).startswith(norm_loose(title)):
            return title + "\n" + text
        return text
    if title:
        return title
    return text


def doc_key(d: Any) -> str:
    title = get_doc_title(d)
    text = get_doc_text(d)
    return norm_loose(title + "\n" + text[:500])


def short_text(s: str, n: int = 500) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:n]


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


def first_hit_rank(docs: List[Any], answers: List[str], k: int) -> int:
    for idx, d in enumerate(docs[:k], 1):
        if doc_has_answer(get_doc_text(d), answers):
            return idx
    return -1


def docs_hit(docs: List[Any], answers: List[str], k: int) -> bool:
    return first_hit_rank(docs, answers, k) != -1


def answer_snippet(text: str, answers: List[str], window: int = 120) -> str:
    text0 = str(text or "")
    text_norm = norm_basic(text0)

    best_pos = -1
    best_ans = ""

    for ans in answers:
        ans_b = norm_basic(ans)
        if not ans_b:
            continue
        pos = text_norm.find(ans_b)
        if pos >= 0:
            best_pos = pos
            best_ans = ans
            break

    if best_pos < 0:
        return ""

    start = max(0, best_pos - window)
    end = min(len(text0), best_pos + len(str(best_ans)) + window)
    snip = text0[start:end]
    snip = re.sub(r"\s+", " ", snip).strip()
    return snip


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
                        {"match_phrase": {"title": {"query": q_simple, "slop": 2, "boost": 4.0}}},
                        {"match_phrase": {"text": {"query": q_simple, "slop": 4, "boost": 2.0}}},
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
                "rerank_score": None,
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
                "rerank_score": None,
            }
        pool[key]["bm25_rank"] = rank
        pool[key]["score"] += beta * (1.0 / (rank + k))

    def sort_key(x: Dict[str, Any]):
        dense_rank = x["dense_rank"] if x["dense_rank"] is not None else 10**9
        bm25_rank = x["bm25_rank"] if x["bm25_rank"] is not None else 10**9
        return (-x["score"], dense_rank, bm25_rank)

    return sorted(pool.values(), key=sort_key)


class Qwen3NativeReranker:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 1024,
        dtype: str = "float16",
        instruction: str = "Given an open-domain question, retrieve passages that contain the exact answer or evidence needed to answer it.",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = int(max_length)
        self.instruction = instruction

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=True,
            trust_remote_code=False,
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device).eval()

        if hasattr(self.model, "config"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.token_false_id = self._safe_token_id("no")
        self.token_true_id = self._safe_token_id("yes")

        if self.token_false_id is None or self.token_true_id is None:
            raise ValueError(f"Cannot find yes/no token ids. yes={self.token_true_id}, no={self.token_false_id}")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        print(
            f"[init] Qwen3NativeReranker loaded. "
            f"yes_id={self.token_true_id}, no_id={self.token_false_id}, max_length={self.max_length}",
            flush=True,
        )

    def _safe_token_id(self, token: str):
        tid = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)

        if tid is not None and tid != unk_id:
            return tid

        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if ids:
            return ids[-1]

        ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
        if ids:
            return ids[-1]

        return None

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {str(query or '').strip()}\n"
            f"<Document>: {str(doc or '').strip()}"
        )

    def predict(self, pairs, batch_size: int = 1, show_progress_bar: bool = False):
        scores_all: List[float] = []
        bs = max(1, int(batch_size))

        body_max_len = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        body_max_len = max(32, body_max_len)

        for start in range(0, len(pairs), bs):
            batch_pairs = pairs[start:start + bs]
            texts = [self._format_pair(q, d) for q, d in batch_pairs]

            inputs = self.tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=body_max_len,
                return_attention_mask=False,
                add_special_tokens=False,
            )

            for i, ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

            batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            with self.torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
                true_logits = logits[:, self.token_true_id]
                false_logits = logits[:, self.token_false_id]
                scores = (true_logits - false_logits).detach().float().cpu().numpy().tolist()

            scores_all.extend([float(x) for x in scores])

        return scores_all


def build_reranker(model_path: str, device: str, max_length: int, instruction: str):
    model_path_str = str(model_path)
    if "Qwen3-Reranker" in model_path_str or "qwen3-reranker" in model_path_str.lower():
        print("[init] using Qwen3NativeReranker with yes/no logits scoring", flush=True)
        return Qwen3NativeReranker(
            model_path=model_path_str,
            device=device,
            max_length=max_length,
            dtype="float16",
            instruction=instruction,
        )

    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_path_str, device=device, max_length=max_length)


def rerank_docs(
    reranker,
    query: str,
    docs: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], float]:
    if not docs:
        return [], 0.0

    t0 = time.time()
    pairs = [(query, get_rerank_text(d) or "") for d in docs]
    scores = reranker.predict(pairs, batch_size=max(1, int(batch_size)), show_progress_bar=False)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    out = []
    for d, s in zip(docs, scores):
        nd = dict(d)
        nd["rerank_score"] = float(s)
        out.append(nd)

    out.sort(key=lambda x: (-float(x.get("rerank_score", 0.0)), -float(x.get("score", 0.0))))
    return out, time.time() - t0


def top_keys(docs: List[Any], k: int) -> List[str]:
    return [doc_key(d) for d in docs[:k]]


def overlap_ratio(a: List[Any], b: List[Any], k: int) -> float:
    sa = set(top_keys(a, k))
    sb = set(top_keys(b, k))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, k)


def added_removed_titles(base_docs: List[Any], new_docs: List[Any], k: int) -> Tuple[List[str], List[str]]:
    base_keys = set(top_keys(base_docs, k))
    new_keys = set(top_keys(new_docs, k))

    added = []
    removed = []

    for d in new_docs[:k]:
        if doc_key(d) not in base_keys:
            added.append(get_doc_title(d))

    for d in base_docs[:k]:
        if doc_key(d) not in new_keys:
            removed.append(get_doc_title(d))

    return added, removed


def enrich_doc(
    d: Any,
    rank: int,
    path: str,
    answers: List[str],
    max_text_chars: int,
) -> Dict[str, Any]:
    text = get_doc_text(d)
    title = get_doc_title(d)
    hit = doc_has_answer(text, answers)

    return {
        "path": path,
        "rank": rank,
        "title": title,
        "key": doc_key(d),
        "answer_hit": int(hit),
        "answer_snippet": answer_snippet(text, answers),
        "dense_rank": d.get("dense_rank") if isinstance(d, dict) else None,
        "bm25_rank": d.get("bm25_rank") if isinstance(d, dict) else None,
        "rrf_score": d.get("score") if isinstance(d, dict) else None,
        "rerank_score": d.get("rerank_score") if isinstance(d, dict) else None,
        "bm25_score": d.get("bm25_score") if isinstance(d, dict) else None,
        "text_preview": short_text(text, max_text_chars),
    }


def classify_case(dense_hit: bool, rrf_hit: bool, rerank_hit: bool) -> str:
    if dense_hit and rerank_hit and rrf_hit:
        return "all_hit"
    if (not dense_hit) and (not rrf_hit) and (not rerank_hit):
        return "all_miss"
    if (not dense_hit) and rerank_hit:
        return "rerank_rescue_dense"
    if dense_hit and (not rerank_hit):
        return "rerank_lost_dense"
    if (not rrf_hit) and rerank_hit:
        return "rerank_rescue_rrf"
    if rrf_hit and (not rerank_hit):
        return "rerank_lost_rrf"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf-config", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-text-chars", type=int, default=700)
    ap.add_argument("--only-cases", type=str, default="", help="comma separated case types, optional")
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
    beta = float(cfg["rrf"].get("beta", 0.2))
    rrf_k = float(cfg["rrf"].get("k", 60))
    rrf_pool_topk = int(cfg["rrf"].get("pool_topk", 100))
    rerank_topk = int((cfg.get("rerank", {}) or {}).get("final_topk", 10))

    rerank_cfg = cfg.get("rerank", {}) or {}
    heavy_cfg = base_cfg.get("heavy_retriever", {}) or {}
    reranker_model = str(rerank_cfg.get("model", heavy_cfg.get("reranker_model", ""))).strip()
    rerank_device = str(rerank_cfg.get("device", heavy_cfg.get("rerank_device", "cuda"))).strip()
    rerank_batch_size = int(rerank_cfg.get("batch_size", heavy_cfg.get("rerank_batch_size", 4)))
    rerank_max_length = int(rerank_cfg.get("max_length", heavy_cfg.get("rerank_max_length", 1024)))
    rerank_instruction = str(
        rerank_cfg.get(
            "instruction",
            "Given an open-domain question, retrieve passages that contain the exact answer or evidence needed to answer it.",
        )
    )

    out_base = Path(cfg["output"]["out_csv"])
    cases_csv = out_base.with_suffix(".diff_cases.csv")
    docs_jsonl = out_base.with_suffix(".diff_docs.jsonl")
    summary_json = out_base.with_suffix(".diff_summary.json")

    only_cases = set(x.strip() for x in args.only_cases.split(",") if x.strip())

    print("[init] loading dense/light retriever...", flush=True)
    light = build_light_retriever(base_cfg, topn=dense_topn)
    print("[init] dense/light retriever loaded.", flush=True)

    print(f"[init] loading reranker: {reranker_model}", flush=True)
    reranker = build_reranker(
        model_path=reranker_model,
        device=rerank_device,
        max_length=rerank_max_length,
        instruction=rerank_instruction,
    )
    print("[init] reranker loaded.", flush=True)

    questions = load_questions(dataset, max_questions)
    print(f"[init] questions loaded: {len(questions)}", flush=True)

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})

    case_rows: List[Dict[str, Any]] = []
    case_counter: Dict[str, int] = {}
    t_all = time.time()

    with docs_jsonl.open("w", encoding="utf-8") as jf:
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

            rrf_docs = weighted_rrf(
                dense_docs=dense_docs[:dense_topn],
                bm25_docs=bm25_docs[:bm25_topn],
                alpha=alpha,
                beta=beta,
                k=rrf_k,
            )

            rrf_pool_docs = rrf_docs[: min(rrf_pool_topk, len(rrf_docs))]
            rerank_docs_top, rerank_t = rerank_docs(
                reranker=reranker,
                query=q,
                docs=rrf_pool_docs,
                batch_size=rerank_batch_size,
            )

            topk = args.topk

            dense_hit = docs_hit(dense_docs, answers, topk)
            rrf_hit = docs_hit(rrf_docs, answers, topk)
            rerank_hit = docs_hit(rerank_docs_top, answers, topk)
            rrf100_hit = docs_hit(rrf_docs, answers, rrf_pool_topk)

            dense_first = first_hit_rank(dense_docs, answers, topk)
            rrf_first = first_hit_rank(rrf_docs, answers, topk)
            rerank_first = first_hit_rank(rerank_docs_top, answers, topk)
            rrf100_first = first_hit_rank(rrf_docs, answers, rrf_pool_topk)

            case_type = classify_case(dense_hit, rrf_hit, rerank_hit)
            case_counter[case_type] = case_counter.get(case_type, 0) + 1

            if only_cases and case_type not in only_cases:
                continue

            added_vs_dense, removed_vs_dense = added_removed_titles(dense_docs, rerank_docs_top, topk)
            added_vs_rrf, removed_vs_rrf = added_removed_titles(rrf_docs, rerank_docs_top, topk)

            dense_top_docs = [
                enrich_doc(d, rank=j, path="dense_top10", answers=answers, max_text_chars=args.max_text_chars)
                for j, d in enumerate(dense_docs[:topk], 1)
            ]
            rrf_top_docs = [
                enrich_doc(d, rank=j, path="rrf_top10", answers=answers, max_text_chars=args.max_text_chars)
                for j, d in enumerate(rrf_docs[:topk], 1)
            ]
            rerank_top_docs = [
                enrich_doc(d, rank=j, path="rerank_top10", answers=answers, max_text_chars=args.max_text_chars)
                for j, d in enumerate(rerank_docs_top[:topk], 1)
            ]

            row = {
                "qid": qid,
                "question": q,
                "answers_json": json.dumps(answers, ensure_ascii=False),
                "case_type": case_type,

                "dense_hit": int(dense_hit),
                "rrf_hit": int(rrf_hit),
                "rerank_hit": int(rerank_hit),
                "rrf100_hit": int(rrf100_hit),

                "dense_first_hit_rank_top10": dense_first,
                "rrf_first_hit_rank_top10": rrf_first,
                "rerank_first_hit_rank_top10": rerank_first,
                "rrf_first_hit_rank_top100": rrf100_first,

                "dense_rrf_overlap_top10": overlap_ratio(dense_docs, rrf_docs, topk),
                "dense_rerank_overlap_top10": overlap_ratio(dense_docs, rerank_docs_top, topk),
                "rrf_rerank_overlap_top10": overlap_ratio(rrf_docs, rerank_docs_top, topk),

                "rerank_added_vs_dense_titles": json.dumps(added_vs_dense, ensure_ascii=False),
                "rerank_removed_vs_dense_titles": json.dumps(removed_vs_dense, ensure_ascii=False),
                "rerank_added_vs_rrf_titles": json.dumps(added_vs_rrf, ensure_ascii=False),
                "rerank_removed_vs_rrf_titles": json.dumps(removed_vs_rrf, ensure_ascii=False),

                "dense_top10_answer_titles": json.dumps(
                    [x["title"] for x in dense_top_docs if x["answer_hit"] == 1],
                    ensure_ascii=False,
                ),
                "rrf_top10_answer_titles": json.dumps(
                    [x["title"] for x in rrf_top_docs if x["answer_hit"] == 1],
                    ensure_ascii=False,
                ),
                "rerank_top10_answer_titles": json.dumps(
                    [x["title"] for x in rerank_top_docs if x["answer_hit"] == 1],
                    ensure_ascii=False,
                ),

                "dense_time_s": dense_t,
                "bm25_time_s": bm25_t,
                "rerank_time_s": rerank_t,
                "rrf_all_len": len(rrf_docs),
                "rrf_pool_len": len(rrf_pool_docs),
            }
            case_rows.append(row)

            detail = {
                "qid": qid,
                "question": q,
                "answers": answers,
                "case_type": case_type,
                "metrics": {
                    "dense_hit": dense_hit,
                    "rrf_hit": rrf_hit,
                    "rerank_hit": rerank_hit,
                    "rrf100_hit": rrf100_hit,
                    "dense_first_hit_rank_top10": dense_first,
                    "rrf_first_hit_rank_top10": rrf_first,
                    "rerank_first_hit_rank_top10": rerank_first,
                    "rrf_first_hit_rank_top100": rrf100_first,
                    "dense_rrf_overlap_top10": overlap_ratio(dense_docs, rrf_docs, topk),
                    "dense_rerank_overlap_top10": overlap_ratio(dense_docs, rerank_docs_top, topk),
                    "rrf_rerank_overlap_top10": overlap_ratio(rrf_docs, rerank_docs_top, topk),
                    "rrf_all_len": len(rrf_docs),
                    "rrf_pool_len": len(rrf_pool_docs),
                },
                "dense_top10": dense_top_docs,
                "rrf_top10": rrf_top_docs,
                "rerank_top10": rerank_top_docs,
                "rerank_added_vs_dense_titles": added_vs_dense,
                "rerank_removed_vs_dense_titles": removed_vs_dense,
                "rerank_added_vs_rrf_titles": added_vs_rrf,
                "rerank_removed_vs_rrf_titles": removed_vs_rrf,
            }
            jf.write(json.dumps(detail, ensure_ascii=False) + "\n")

            if i % 50 == 0:
                n = i
                print(
                    f"[progress] {i}/{len(questions)} "
                    f"cases={case_counter} "
                    f"saved_rows={len(case_rows)} "
                    f"elapsed={time.time()-t_all:.1f}s",
                    flush=True,
                )

    cases_csv.parent.mkdir(parents=True, exist_ok=True)
    with cases_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()) if case_rows else [])
        writer.writeheader()
        writer.writerows(case_rows)

    n_all = len(questions)
    summary = {
        "n_questions": n_all,
        "saved_cases": len(case_rows),
        "case_counter": case_counter,
        "case_rate": {k: v / max(1, n_all) for k, v in case_counter.items()},
        "config": {
            "dataset": dataset,
            "dense_topn": dense_topn,
            "bm25_topn": bm25_topn,
            "rrf_pool_topk": rrf_pool_topk,
            "topk": args.topk,
            "reranker_model": reranker_model,
            "rerank_instruction": rerank_instruction,
            "rerank_max_length": rerank_max_length,
        },
        "outputs": {
            "cases_csv": str(cases_csv),
            "docs_jsonl": str(docs_jsonl),
            "summary_json": str(summary_json),
        },
        "elapsed_s": time.time() - t_all,
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== DIFF SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved cases CSV:", cases_csv)
    print("Saved docs JSONL:", docs_jsonl)
    print("Saved summary JSON:", summary_json)


if __name__ == "__main__":
    main()