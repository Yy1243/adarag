# -*- coding: utf-8 -*-
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
from adarag.pipeline.prompt_builder import build_prompt, format_doc_for_prompt
from adarag.eval.evaluator import batch_accuracy
from scripts.eval_light_heavy_retrieval_generation import build_light_retriever, build_llm


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
        if doc_has_answer(get_doc_full_text(d) or "", answers):
            return True
    return False


def first_hit_rank(docs: List[Any], answers: List[str], k: int) -> int:
    for idx, d in enumerate(docs[:k], 1):
        if doc_has_answer(get_doc_full_text(d) or "", answers):
            return idx
    return -1


def get_doc_text(d: Any) -> str:
    """
    只返回正文 text，不拼 title。
    """
    if isinstance(d, dict):
        return str(d.get("text", "") or "")
    return str(getattr(d, "text", "") or "")


def get_doc_title(d: Any) -> str:
    """
    只从显式 title 字段取标题，不从 text 第一行推断。
    """
    if isinstance(d, dict):
        return str(d.get("title", "") or "")
    return str(getattr(d, "title", "") or "")


def get_doc_full_text(d: Any) -> str:
    """
    内部检索、rerank、answer hit 判断用完整文本。
    """
    if isinstance(d, dict):
        title = str(d.get("title", "") or "").strip()
        text = str(d.get("text", "") or "").strip()
        if title and text:
            return f"{title}\n{text}"
        return title or text

    if hasattr(d, "full_text"):
        return str(d.full_text or "")

    title = get_doc_title(d).strip()
    text = get_doc_text(d).strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def get_rerank_text(d: Any) -> str:
    return get_doc_full_text(d)


def doc_key(d: Any) -> str:
    return norm_loose(get_doc_full_text(d)[:800])


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
        docs.append({
            "doc_id": str(src.get("doc_id", h.get("_id", ""))),
            "title": title,
            "text": text,
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

                # 用 yes-logit minus no-logit 做排序分数，越大越相关
                scores = (true_logits - false_logits).detach().float().cpu().numpy().tolist()

            scores_all.extend([float(x) for x in scores])

        return scores_all


def build_cross_encoder_reranker(
    model_path: str,
    device: str,
    max_length: int,
    instruction: str = "Given an open-domain question, retrieve passages that contain the exact answer or evidence needed to answer it.",
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
        return CrossEncoder(
            model_path_str,
            device=device,
            trust_remote_code=False,
            local_files_only=True,
            max_length=max_length,
        )
    except TypeError:
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

    # 注意这里用 get_rerank_text，而不是 get_doc_text
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

def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("text", "output", "answer", "generated_text"):
            if key in x:
                return str(x[key])
        return str(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_text(x[0])
    return str(x)


def _postprocess_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    m = re.search(r"final\s*answer\s*[:：]\s*(.*)$", s, flags=re.I | re.S)
    if m:
        s = m.group(1)
    # 避免模型偶尔输出解释；只取第一行会更接近 NQ short-answer 评测
    s = s.strip().split("\n", 1)[0]
    s = " ".join(s.strip().split())
    return s


def contains_any(text: str, answers: List[str]) -> bool:
    return doc_has_answer(text or "", answers or [])

def strip_leading_title_once(text: str, title: str) -> str:
    """
    如果 text 开头已经带有 title，则删除一次。
    只删除开头，不删除正文中正常出现的标题/实体名。
    """
    text = str(text or "").strip()
    title = str(title or "").strip()

    if not text or not title:
        return text

    # 先处理最常见的精确前缀情况：
    # Love Yourself\nLove Yourself "Love Yourself" is ...
    # 或 Love Yourself Love Yourself "Love Yourself" is ...
    if text.lower().startswith(title.lower()):
        rest = text[len(title):]
        rest = rest.lstrip(" \n\t\r:：-—.,")
        return rest

    return text


def strip_repeated_leading_title(text: str, title: str, max_repeat: int = 3) -> str:
    """
    连续清理开头重复标题。
    例如：
    Love Yourself
    Love Yourself "Love Yourself" is ...
    -> "Love Yourself" is ...

    注意：不会删除正文中间的标题。
    """
    text = str(text or "").strip()
    title = str(title or "").strip()

    if not text or not title:
        return text

    for _ in range(max_repeat):
        new_text = strip_leading_title_once(text, title)
        if new_text == text:
            break
        text = new_text.strip()

    return text


def format_doc_for_generation_prompt(d: Any, max_doc_chars: int = 1600) -> str:
    return format_doc_for_prompt(d, max_doc_chars=max_doc_chars)

def build_generation_prompt(
    question: str,
    docs: List[Any],
    max_doc_chars: int = 1600,
    prompt_mode: str = "context_only",
) -> str:
    """
    prompt_mode:
      - context_only: 只允许使用 Context；Context 证据不足则输出 UNKNOWN。
      - original: 允许 Context 不足时使用模型自身知识。
    """
    mode = str(prompt_mode or "context_only").strip().lower()
    question = str(question or "").strip()

    lines = []

    if mode == "original":
        lines.append("You are an extractive QA system.")
        lines.append("Answer using ONLY the Context.")
        lines.append("If the context is empty or does not contain the answer, answer using your own knowledge.")
        lines.append("Return ONLY the short answer in ONE line. No explanation（1-10 words). No extra words")
        lines.append("")
        lines.append("Context:")
    else:
        lines.append("You are an extractive QA system.")
        lines.append("Answer using ONLY the provided Context.")
        lines.append("If the Context does not contain enough evidence to answer, return ONLY: UNKNOWN")
        lines.append("Return ONLY the short answer in ONE line. No explanation. No extra words.")
        lines.append("")
        lines.append("Context:")

    for i, d in enumerate(docs or [], 1):
        doc_block = format_doc_for_generation_prompt(d, max_doc_chars=max_doc_chars).strip()
        if not doc_block:
            continue

        lines.append(f"[Document {i}]")
        lines.append(doc_block)
        lines.append("")

    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def judge_correct(judge_llm, question: str, pred: str, gold: List[str]) -> Optional[bool]:
    if judge_llm is None:
        return None
    gold_list = [str(x) for x in (gold or []) if str(x).strip()]
    prompt = (
        "You are a strict QA evaluator.\n"
        "Decide whether the model answer is semantically equivalent to ANY gold answer.\n"
        "Return ONLY 1 or 0.\n\n"
        f"Question: {question}\n"
        f"Gold answers: {gold_list}\n"
        f"Model answer: {pred}\n"
    )
    out = _to_text(judge_llm.generate(prompt)).strip()
    m = re.search(r"[01]", out)
    return (m is not None) and (m.group(0) == "1")


def eval_generation_path(
    *,
    item: Dict[str, Any],
    path_name: str,
    docs: List[Any],
    retrieve_time_s: float,
    llm,
    judge_llm,
    prompt_max_doc_chars: int,
    acc_mode: str,
    gen_topk: int,
    prompt_mode: str,
) -> Dict[str, Any]:
    q = item["question"]
    answers = item["answers"]
    docs_k = docs[:gen_topk]

    t0 = time.time()
    prompt = build_generation_prompt(q, docs_k, max_doc_chars=prompt_max_doc_chars, prompt_mode=prompt_mode)
    prompt_build_time_s = time.time() - t0

    t1 = time.time()
    pred_raw = llm.generate(prompt)
    generate_time_s = time.time() - t1
    pred = _postprocess_answer(_to_text(pred_raw))

    t2 = time.time()
    judge_val = judge_correct(judge_llm, q, pred, answers) if judge_llm is not None else None
    judge_time_s = time.time() - t2 if judge_llm is not None else 0.0

    recall_at_gen_topk = docs_hit(docs_k, answers, gen_topk)
    hit_rank = first_hit_rank(docs_k, answers, gen_topk)
    prompt_has_gold = contains_any(prompt, answers)
    contains_correct = contains_any(pred, answers)
    acc_value = float(batch_accuracy([pred], [answers], mode=acc_mode)) if answers else None

    return {
        "qid": item["qid"],
        "question": q,
        "path": path_name,
        "answers_json": json.dumps(answers, ensure_ascii=False),
        "gen_topk": gen_topk,
        "retrieve_time_s": retrieve_time_s,
        "prompt_build_time_s": prompt_build_time_s,
        "generate_time_s": generate_time_s,
        "judge_time_s": judge_time_s,
        "total_time_s": retrieve_time_s + prompt_build_time_s + generate_time_s + judge_time_s,
        "recall_at_gen_topk": int(recall_at_gen_topk),
        "first_answer_doc_rank": hit_rank,
        "prompt_has_gold": int(prompt_has_gold),
        "acc_mode": acc_mode,
        "acc_value": acc_value,
        "contains_correct": int(contains_correct),
        "judge_correct": None if judge_val is None else int(bool(judge_val)),
        "pred": pred,
        "top_titles_json": json.dumps([get_doc_title(d) for d in docs_k], ensure_ascii=False),
    }


def summarize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    paths = sorted(set(r["path"] for r in records))
    out: List[Dict[str, Any]] = []
    for p in paths:
        rows = [r for r in records if r["path"] == p]
        n = len(rows)
        def mean_num(key: str) -> Optional[float]:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            if not vals:
                return None
            return float(sum(float(v) for v in vals) / len(vals))
        out.append({
            "path": p,
            "n_questions": n,
            "recall_at_gen_topk": mean_num("recall_at_gen_topk"),
            "prompt_gold_rate": mean_num("prompt_has_gold"),
            "token_accuracy": mean_num("acc_value"),
            "contains_accuracy": mean_num("contains_correct"),
            "judge_accuracy": mean_num("judge_correct"),
            "mean_retrieve_time_s": mean_num("retrieve_time_s"),
            "mean_generate_time_s": mean_num("generate_time_s"),
            "mean_judge_time_s": mean_num("judge_time_s"),
            "mean_total_time_s": mean_num("total_time_s"),
            "judge_correct_when_recall0": (
                sum(1 for r in rows if int(r.get("recall_at_gen_topk", 0)) == 0 and int(r.get("judge_correct") or 0) == 1)
                / max(1, sum(1 for r in rows if int(r.get("recall_at_gen_topk", 0)) == 0))
            ) if any(r.get("judge_correct") is not None for r in rows) else None,
            "n_recall0": sum(1 for r in rows if int(r.get("recall_at_gen_topk", 0)) == 0),
            "n_judge_correct_recall0": sum(1 for r in rows if int(r.get("recall_at_gen_topk", 0)) == 0 and int(r.get("judge_correct") or 0) == 1),
        })
    return out


def pairwise_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_qid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        by_qid.setdefault(str(r["qid"]), {})[r["path"]] = r

    def jc(row: Optional[Dict[str, Any]]) -> Optional[int]:
        if row is None or row.get("judge_correct") is None:
            return None
        return int(row["judge_correct"])

    pairs = [
        ("dense_top10", "rrf_top10"),
        ("dense_top10", "rerank_top10"),
        ("rrf_top10", "rerank_top10"),
    ]
    out: Dict[str, Any] = {}
    for a, b in pairs:
        both = [v for v in by_qid.values() if a in v and b in v and jc(v[a]) is not None and jc(v[b]) is not None]
        n = len(both)
        if n == 0:
            continue
        out[f"{b}_beats_{a}"] = sum(1 for v in both if jc(v[a]) == 0 and jc(v[b]) == 1) / n
        out[f"{b}_loses_to_{a}"] = sum(1 for v in both if jc(v[a]) == 1 and jc(v[b]) == 0) / n
        out[f"{a}_{b}_same_correct"] = sum(1 for v in both if jc(v[a]) == 1 and jc(v[b]) == 1) / n
        out[f"{a}_{b}_same_wrong"] = sum(1 for v in both if jc(v[a]) == 0 and jc(v[b]) == 0) / n
        out[f"{a}_{b}_n"] = n
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
    rrf_pool_topk = int(cfg["rrf"].get("pool_topk", 100))

    heavy_cfg = base_cfg.get("heavy_retriever", {}) or {}
    rerank_cfg = cfg.get("rerank", {}) or {}
    reranker_model = str(rerank_cfg.get("model", heavy_cfg.get("reranker_model", ""))).strip()
    rerank_device = str(rerank_cfg.get("device", heavy_cfg.get("rerank_device", "cuda"))).strip()
    rerank_batch_size = int(rerank_cfg.get("batch_size", heavy_cfg.get("rerank_batch_size", 8)))
    rerank_max_length = int(rerank_cfg.get("max_length", heavy_cfg.get("rerank_max_length", 512)))
    rerank_instruction = str(
       rerank_cfg.get(
        "instruction",
        "Given an open-domain question, retrieve passages that contain the exact answer or evidence needed to answer it.",
    )
)

    gen_cfg = cfg.get("generation", {}) or {}
    gen_topk = int(gen_cfg.get("gen_topk", 10))
    prompt_max_doc_chars = int(gen_cfg.get("prompt_max_doc_chars", (base_cfg.get("system", {}) or {}).get("prompt_max_doc_chars", 1600)))
    acc_mode = str(gen_cfg.get("acc_mode", base_cfg.get("acc_mode", "token")))
    prompt_mode = str(gen_cfg.get("prompt_mode", "context_only"))
    paths_to_eval = list(gen_cfg.get("paths", ["dense_top10", "rrf_top10", "rerank_top10"]))

    out_csv = cfg["output"]["out_csv"]

    if not reranker_model and "rerank_top10" in paths_to_eval:
        raise ValueError("No reranker model found. Set rerank.model or base_config.heavy_retriever.reranker_model.")

    print("[init] loading dense/light retriever...", flush=True)
    light = build_light_retriever(base_cfg, topn=dense_topn)
    print("[init] dense/light retriever loaded.", flush=True)

    reranker = None
    if "rerank_top10" in paths_to_eval:
        print(f"[init] loading reranker: {reranker_model} device={rerank_device}", flush=True)
        reranker = build_cross_encoder_reranker(
            reranker_model,
            rerank_device,
            rerank_max_length,
            instruction=rerank_instruction,
        )
        print("[init] reranker loaded.", flush=True)

    print("[init] loading generator llm...", flush=True)
    llm = build_llm(base_cfg.get("llm", {}) or {})
    print("[init] generator llm loaded.", flush=True)

    judge_llm = None
    if base_cfg.get("judge_llm", None):
        print("[init] loading judge llm...", flush=True)
        judge_llm = build_llm(base_cfg.get("judge_llm", {}) or {})
        print("[init] judge llm loaded.", flush=True)
    else:
        print("[warn] judge_llm not found in base config; judge_correct will be None.", flush=True)

    print("[init] loading questions...", flush=True)
    questions = load_questions(dataset, max_questions)
    print(f"[init] questions loaded: {len(questions)}", flush=True)

    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})

    records: List[Dict[str, Any]] = []
    t_all = time.time()

    for i, item in enumerate(questions, 1):
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
        rrf_docs = weighted_rrf(dense_docs[:dense_topn], bm25_docs[:bm25_topn], alpha, beta, rrf_k)

        rerank_docs_top: List[Dict[str, Any]] = []
        rerank_t = 0.0
        if reranker is not None:
            rrf_pool_docs = rrf_docs[: min(rrf_pool_topk, len(rrf_docs))]
            rerank_docs_top, rerank_t = rerank_docs(reranker, q, rrf_pool_docs, rerank_batch_size)

        path_docs: Dict[str, Tuple[List[Any], float]] = {
            "dense_top10": (dense_docs[:gen_topk], dense_t),
            "rrf_top10": (rrf_docs[:gen_topk], dense_t + bm25_t),
            "rerank_top10": (rerank_docs_top[:gen_topk], dense_t + bm25_t + rerank_t),
        }

        for path_name in paths_to_eval:
            docs, retrieve_time_s = path_docs[path_name]
            rec = eval_generation_path(
                item=item,
                path_name=path_name,
                docs=docs,
                retrieve_time_s=retrieve_time_s,
                llm=llm,
                judge_llm=judge_llm,
                prompt_max_doc_chars=prompt_max_doc_chars,
                acc_mode=acc_mode,
                gen_topk=gen_topk,
                prompt_mode=prompt_mode,
            )
            # 保存检索诊断信息，便于后续分析 rerank 是否 string-recall 下降但 judge 变好
            rec.update({
                "dense_recall_top10": int(docs_hit(dense_docs, answers, min(10, dense_topn))),
                "rrf_recall_top10": int(docs_hit(rrf_docs, answers, gen_topk)),
                "rrf_recall_top100": int(docs_hit(rrf_docs, answers, rrf_pool_topk)),
                "rerank_recall_top10": int(docs_hit(rerank_docs_top, answers, gen_topk)) if rerank_docs_top else None,
                "rrf_first_answer_rank_top100": first_hit_rank(rrf_docs, answers, rrf_pool_topk),
                "rerank_first_answer_rank_top10": first_hit_rank(rerank_docs_top, answers, gen_topk) if rerank_docs_top else -1,
            })
            records.append(rec)

        if i % 10 == 0:
            tmp_summary = summarize_records(records)
            msg = {r["path"]: r.get("judge_accuracy") for r in tmp_summary}
            print(f"[progress] {i}/{len(questions)} judge_acc={msg} elapsed={time.time()-t_all:.1f}s", flush=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys()) if records else []
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_rows = summarize_records(records)
    summary_csv = out_path.with_suffix(".summary.csv")
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = {
        "n_questions": len(questions),
        "dataset": dataset,
        "query_mode": query_mode,
        "dense_topn": dense_topn,
        "bm25_topn": bm25_topn,
        "alpha": alpha,
        "beta": beta,
        "rrf_k": rrf_k,
        "rrf_pool_topk": rrf_pool_topk,
        "gen_topk": gen_topk,
        "prompt_max_doc_chars": prompt_max_doc_chars,
        "prompt_mode": prompt_mode,
        "acc_mode": acc_mode,
        "paths": paths_to_eval,
        "reranker_model": reranker_model,
        "rerank_device": rerank_device,
        "generator_model": (base_cfg.get("llm", {}) or {}).get("model"),
        "judge_model": (base_cfg.get("judge_llm", {}) or {}).get("model"),
        "path_summary": summary_rows,
        "pairwise_summary": pairwise_summary(records),
        "out_csv": str(out_path),
        "summary_csv": str(summary_csv),
        "elapsed_s": time.time() - t_all,
    }
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))
    print("Saved detailed CSV:", out_path)
    print("Saved summary CSV :", summary_csv)
    print("Saved summary JSON:", summary_path)


if __name__ == "__main__":
    main()
