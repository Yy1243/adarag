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


def first_hit_rank(docs: List[Any], answers: List[str], k: int) -> int:
    for idx, d in enumerate(docs[:k], 1):
        text = get_doc_text(d)
        if doc_has_answer(text or "", answers):
            return idx
    return -1


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
        # 避免重复拼接 title
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


class Qwen3NativeReranker:
    """
    Qwen3-Reranker 原生 yes/no logits 打分版本。

    目的：
    - 避免 sentence_transformers.CrossEncoder 把 Qwen3-Reranker 错误加载成
      Qwen3ForSequenceClassification，并随机初始化 score.weight。
    - 使用 Qwen3-Reranker 官方风格的 yes/no token logits 作为相关性分数。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 512,
        dtype: str = "float16",
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
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
            raise ValueError(
                f"Cannot find yes/no token ids. "
                f"yes={self.token_true_id}, no={self.token_false_id}"
            )

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
            f"yes_id={self.token_true_id}, no_id={self.token_false_id}, "
            f"max_length={self.max_length}",
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
        query = str(query or "").strip()
        doc = str(doc or "").strip()

        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
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

            batch = self.tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
            )

            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            with self.torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits[:, -1, :]

                true_logits = logits[:, self.token_true_id]
                false_logits = logits[:, self.token_false_id]

                pair_logits = self.torch.stack([false_logits, true_logits], dim=1)
                probs = self.torch.nn.functional.softmax(pair_logits, dim=1)
                yes_scores = probs[:, 1].detach().float().cpu().numpy().tolist()

            scores_all.extend([float(x) for x in yes_scores])

        return scores_all


def build_cross_encoder_reranker(
    model_path: str,
    device: str,
    max_length: int,
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
):
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

    try:
        reranker = CrossEncoder(
            model_path_str,
            device=device,
            trust_remote_code=False,
            local_files_only=True,
            max_length=max_length,
        )
    except TypeError:
        reranker = CrossEncoder(
            model_path_str,
            device=device,
            max_length=max_length,
        )

    tok = getattr(reranker, "tokenizer", None)
    mdl = getattr(reranker, "model", None)

    if tok is not None and tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            if mdl is not None and hasattr(mdl, "resize_token_embeddings"):
                mdl.resize_token_embeddings(len(tok))

    if tok is not None and mdl is not None and hasattr(mdl, "config"):
        mdl.config.pad_token_id = tok.pad_token_id

    return reranker


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

    scores = reranker.predict(
        pairs,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=False,
    )
    scores = np.asarray(scores, dtype=float).reshape(-1)

    out: List[Dict[str, Any]] = []
    for d, s in zip(docs, scores):
        nd = dict(d)
        nd["rerank_score"] = float(s)
        out.append(nd)

    out.sort(key=lambda x: (-float(x.get("rerank_score", 0.0)), -float(x.get("score", 0.0))))
    return out, time.time() - t0


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
    rrf_pool_topk = int(cfg["rrf"].get("pool_topk", 100))

    heavy_cfg = base_cfg.get("heavy_retriever", {}) or {}
    rerank_cfg = cfg.get("rerank", {}) or {}

    reranker_model = str(rerank_cfg.get("model", heavy_cfg.get("reranker_model", ""))).strip()
    rerank_device = str(rerank_cfg.get("device", heavy_cfg.get("rerank_device", "cuda"))).strip()
    rerank_batch_size = int(rerank_cfg.get("batch_size", heavy_cfg.get("rerank_batch_size", 8)))
    rerank_max_length = int(rerank_cfg.get("max_length", heavy_cfg.get("rerank_max_length", 512)))
    rerank_topk = int(rerank_cfg.get("final_topk", final_topk))
    rerank_instruction = str(
        rerank_cfg.get(
            "instruction",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
    )

    if not reranker_model:
        raise ValueError("No reranker model found. Set rerank.model or base_config.heavy_retriever.reranker_model.")

    out_csv = cfg["output"]["out_csv"]

    print("[init] loading dense/light retriever...", flush=True)
    light = build_light_retriever(base_cfg, topn=dense_topn)
    print("[init] dense/light retriever loaded.", flush=True)

    print(f"[init] loading reranker: {reranker_model} device={rerank_device}", flush=True)
    reranker = build_cross_encoder_reranker(
        model_path=reranker_model,
        device=rerank_device,
        max_length=rerank_max_length,
        instruction=rerank_instruction,
    )
    print("[init] reranker loaded.", flush=True)

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
        "rrf_top100": 0,
        "rerank_top10": 0,
        "union_d10_b300": 0,
        "union_d200_b300": 0,
        "bm25_rescue_over_dense10": 0,
        "rrf_rescue_over_dense10": 0,
        "rrf_lost_dense10": 0,
        "rerank_rescue_over_dense10": 0,
        "rerank_lost_rrf_top10": 0,
        "rerank_lost_rrf_top100": 0,
    }

    total_dense_t = 0.0
    total_bm25_t = 0.0
    total_rerank_t = 0.0

    total_dense_len = 0
    total_bm25_len = 0
    total_rrf_all_len = 0
    total_rrf_pool_len = 0

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

        rrf_pool_docs = rrf_docs[: min(rrf_pool_topk, len(rrf_docs))]

        dense_len = len(dense_docs)
        bm25_len = len(bm25_docs)
        rrf_all_len = len(rrf_docs)
        rrf_pool_len = len(rrf_pool_docs)

        total_dense_len += dense_len
        total_bm25_len += bm25_len
        total_rrf_all_len += rrf_all_len
        total_rrf_pool_len += rrf_pool_len

        rerank_docs_top, rerank_t = rerank_docs(
            reranker=reranker,
            query=q,
            docs=rrf_pool_docs,
            batch_size=rerank_batch_size,
        )
        total_rerank_t += rerank_t

        dense10 = docs_hit(dense_docs, answers, 10)
        dense50 = docs_hit(dense_docs, answers, 50)
        dense200 = docs_hit(dense_docs, answers, min(200, dense_topn))

        bm25_10 = docs_hit(bm25_docs, answers, 10)
        bm25_50 = docs_hit(bm25_docs, answers, 50)
        bm25_100 = docs_hit(bm25_docs, answers, min(100, bm25_topn))
        bm25_300 = docs_hit(bm25_docs, answers, min(300, bm25_topn))

        rrf_top10 = docs_hit(rrf_docs, answers, final_topk)
        rrf_top20 = docs_hit(rrf_docs, answers, 20)
        rrf_top100 = docs_hit(rrf_docs, answers, rrf_pool_topk)
        rerank_top10 = docs_hit(rerank_docs_top, answers, rerank_topk)

        rrf_hit_rank_100 = first_hit_rank(rrf_docs, answers, rrf_pool_topk)
        rerank_hit_rank_10 = first_hit_rank(rerank_docs_top, answers, rerank_topk)

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
        rerank_rescue = (not dense10) and rerank_top10
        rerank_lost_rrf10 = rrf_top10 and (not rerank_top10)
        rerank_lost_rrf100 = rrf_top100 and (not rerank_top10)

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
            "rrf_top100": rrf_top100,
            "rerank_top10": rerank_top10,
            "union_d10_b300": union_d10_b300,
            "union_d200_b300": union_d200_b300,
            "bm25_rescue_over_dense10": bm25_rescue,
            "rrf_rescue_over_dense10": rrf_rescue,
            "rrf_lost_dense10": rrf_lost,
            "rerank_rescue_over_dense10": rerank_rescue,
            "rerank_lost_rrf_top10": rerank_lost_rrf10,
            "rerank_lost_rrf_top100": rerank_lost_rrf100,
        }

        for k2, v in vals.items():
            acc[k2] += int(v)

        rows.append({
            "qid": qid,
            "question": q,
            "answers_json": json.dumps(answers, ensure_ascii=False),
            **{k2: int(v) for k2, v in vals.items()},
            "rrf_hit_rank_top100": rrf_hit_rank_100,
            "rerank_hit_rank_top10": rerank_hit_rank_10,
            "dense_len": dense_len,
            "bm25_len": bm25_len,
            "rrf_all_len": rrf_all_len,
            "rrf_pool_len": rrf_pool_len,
            "dense_time_s": dense_t,
            "bm25_time_s": bm25_t,
            "rerank_time_s": rerank_t,
            "rrf_top10_titles": json.dumps([x.get("title", "") for x in rrf_docs[:final_topk]], ensure_ascii=False),
            "rrf_top100_titles": json.dumps([x.get("title", "") for x in rrf_docs[:rrf_pool_topk]], ensure_ascii=False),
            "rerank_top10_titles": json.dumps([x.get("title", "") for x in rerank_docs_top[:rerank_topk]], ensure_ascii=False),
            "rerank_top10_scores": json.dumps([x.get("rerank_score", None) for x in rerank_docs_top[:rerank_topk]], ensure_ascii=False),
        })

        if i % 50 == 0:
            n = i
            print(
                f"[progress] {i}/{len(questions)} "
                f"d10={acc['dense10']/n:.4f} "
                f"b300={acc['bm25_300']/n:.4f} "
                f"rrf10={acc['rrf_top10']/n:.4f} "
                f"rrf100={acc['rrf_top100']/n:.4f} "
                f"rerank10={acc['rerank_top10']/n:.4f} "
                f"mean_rrf_all_len={total_rrf_all_len/n:.1f} "
                f"rerank_lost_rrf100={acc['rerank_lost_rrf_top100']/n:.4f} "
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
        "rrf_pool_topk": rrf_pool_topk,
        "rerank_topk": rerank_topk,
        "reranker_model": reranker_model,
        "rerank_device": rerank_device,
        "rerank_batch_size": rerank_batch_size,
        "rerank_max_length": rerank_max_length,
        "rerank_instruction": rerank_instruction,
        "max_per_title": max_per_title,
        "mean_dense_time_s": total_dense_t / max(n, 1),
        "mean_bm25_time_s": total_bm25_t / max(n, 1),
        "mean_rerank_time_s": total_rerank_t / max(n, 1),
        "mean_dense_len": total_dense_len / max(n, 1),
        "mean_bm25_len": total_bm25_len / max(n, 1),
        "mean_rrf_all_len": total_rrf_all_len / max(n, 1),
        "mean_rrf_pool_len": total_rrf_pool_len / max(n, 1),
        **{k2: v / max(n, 1) for k2, v in acc.items()},
        "rrf_delta_vs_dense10": (acc["rrf_top10"] - acc["dense10"]) / max(n, 1),
        "rerank_delta_vs_rrf10": (acc["rerank_top10"] - acc["rrf_top10"]) / max(n, 1),
        "rerank_delta_vs_dense10": (acc["rerank_top10"] - acc["dense10"]) / max(n, 1),
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