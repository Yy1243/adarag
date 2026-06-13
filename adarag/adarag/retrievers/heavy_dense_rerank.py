# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from adarag.data import Doc


_STOP = {
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


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-z0-9]+", text)
    return [t for t in toks if len(t) >= 2]


def _content_tokens(q: str) -> List[str]:
    return [t for t in _tokenize(q) if t not in _STOP]


def simplify_query(q: str) -> str:
    q = (q or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    toks = [t for t in q.split() if t and (t not in _STOP)]
    return " ".join(toks) if toks else (q.strip() or "")

def _doc_key(d: Doc, mode: str = "title_text") -> str:
    mode = (mode or "title_text").lower().strip()
    if mode == "doc_id":
        return str(d.doc_id or "")
    if mode == "title_text":
        return (d.full_text or "")[:256]

    raise ValueError(f"Unsupported dedup_key_mode={mode!r}")


def _minmax_norm_arr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) == 0:
        return x
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax <= xmin + 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


class DenseRerankHeavyRetriever:
    """
    heavy = dense top-N candidate recall + shallow soft rerank

    关键思想：
    - dense 先召回大候选池
    - 只对前面一小段做 rerank
    - rerank 不覆盖 dense 排序，而是和 dense 分数做 soft blend
    """

    def __init__(
        self,
        dense_retriever,
        reranker_model: str,
        candidate_pool_n: int = 120,
        rerank_head_n: int = 30,
        out_top_n: int = 50,
        rerank_batch_size: int = 8,
        rerank_max_length: int = 512,
        rerank_device: str = "cpu",
        dense_prior_weight: float = 0.75,
        max_per_title: int = 999,
        dedup_key_mode: str = "title_text",
        lexical_bonus_weight: float = 0.00,
        title_bonus: float = 0.00,
        phrase_bonus: float = 0.00,
        profile: bool = False,
    ):
        self.dense = dense_retriever
        self.reranker_model = reranker_model

        self.candidate_pool_n = int(candidate_pool_n)
        self.rerank_head_n = int(rerank_head_n)
        self.out_top_n = int(out_top_n)

        self.rerank_batch_size = int(rerank_batch_size)
        self.rerank_max_length = int(rerank_max_length)
        self.rerank_device = rerank_device
        self.dense_prior_weight = float(dense_prior_weight)

        self.max_per_title = int(max(1, max_per_title))
        self.dedup_key_mode = (dedup_key_mode or "title_text").lower().strip()

        self.lexical_bonus_weight = float(lexical_bonus_weight)
        self.title_bonus = float(title_bonus)
        self.phrase_bonus = float(phrase_bonus)

        self.profile = bool(profile)
        self._reranker = None

    def _print_profile(
        self,
        query: str,
        t_total: float,
        t_dense: float,
        t_rerank: float,
        out_n: int,
    ) -> None:
        if not self.profile:
            return
        print(
            "[DenseRerankHeavy][profile] "
            f"total={t_total:.3f}s "
            f"dense={t_dense:.3f}s "
            f"rerank={t_rerank:.3f}s "
            f"out_n={out_n} "
            f"q={query[:100]!r}",
            flush=True,
        )

    def _lazy_load_reranker(self):
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(
                    self.reranker_model,
                    device=self.rerank_device,
                    trust_remote_code=False,
                    local_files_only=True,
                    max_length=self.rerank_max_length,
                )
            except TypeError:
                self._reranker = CrossEncoder(
                    self.reranker_model,
                    device=self.rerank_device,
                    max_length=self.rerank_max_length,
                )
        return self._reranker

    def _lexical_bonus(self, query: str, doc: Doc) -> float:
        if (
            self.lexical_bonus_weight <= 0
            and self.title_bonus <= 0
            and self.phrase_bonus <= 0
        ):
            return 0.0

        q_simple = simplify_query(query)
        q_terms = set(_content_tokens(query))
        if not q_terms:
            return 0.0

        title = str(doc.title or "")
        body = str(doc.text or "")
        full = str(doc.full_text or "")

        title_l = title.lower()
        body_l = body.lower()
        full_l = full.lower()

        doc_terms = set(_content_tokens(full[:512]))
        overlap = len(q_terms & doc_terms) / float(max(1, len(q_terms)))
        bonus = self.lexical_bonus_weight * overlap

        title_terms = set(_content_tokens(title))
        title_hit = len(q_terms & title_terms) / float(max(1, len(q_terms)))
        bonus += self.title_bonus * title_hit

        if q_simple and len(q_simple.split()) <= 8:
            if q_simple in title_l:
                bonus += self.phrase_bonus
            elif q_simple in body_l or q_simple in full_l:
                bonus += 0.5 * self.phrase_bonus

        return float(bonus)

    def retrieve(self, query: str) -> Tuple[List[Doc], np.ndarray]:
        t_total0 = time.perf_counter()

        if not hasattr(self.dense, "retrieve_k"):
            raise RuntimeError("dense_retriever must implement retrieve_k(query, top_n).")

        # 1) dense candidate recall
        t0 = time.perf_counter()
        docs_d, scores_d = self.dense.retrieve_k(query, self.candidate_pool_n)
        t_dense = time.perf_counter() - t0
        scores_d = np.asarray(scores_d, dtype=float).reshape(-1)

        if not docs_d:
            return [], np.asarray([], dtype=float)

        # 2) 只对前面一小段做 rerank
        t0 = time.perf_counter()
        head_n = min(self.rerank_head_n, len(docs_d))
        head_docs = docs_d[:head_n]
        tail_docs = docs_d[head_n:]

        head_dense_scores = scores_d[:head_n]
        head_dense_norm = _minmax_norm_arr(head_dense_scores)

        reranker = self._lazy_load_reranker()
        pairs = [(query, d.full_text or "") for d in head_docs]
        rr_scores = reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False,
        )
        rr_scores = np.asarray(rr_scores, dtype=float).reshape(-1)
        rr_norm = _minmax_norm_arr(rr_scores)

        lexical_bonus = np.asarray([self._lexical_bonus(query, d) for d in head_docs], dtype=float)

        # 关键：soft blend，不让 reranker 完全覆盖 dense
        mixed_scores = (
            self.dense_prior_weight * head_dense_norm
            + (1.0 - self.dense_prior_weight) * rr_norm
            + lexical_bonus
        )

        order = np.argsort(-mixed_scores)
        reranked_head_docs = [head_docs[i] for i in order]
        reranked_head_scores = mixed_scores[order]

        ranked_docs = reranked_head_docs + tail_docs

        # tail 保持原 dense 顺序
        tail_dense_norm = _minmax_norm_arr(scores_d[head_n:])
        ranked_scores_map: Dict[str, float] = {}
        for d, s in zip(reranked_head_docs, reranked_head_scores):
            ranked_scores_map[_doc_key(d, self.dedup_key_mode)] = 2.0 + float(s)
        for d, s in zip(tail_docs, tail_dense_norm):
            ranked_scores_map[_doc_key(d, self.dedup_key_mode)] = float(s)

        # 3) dedup + final output
        out_docs: List[Doc] = []
        out_scores: List[float] = []
        cnt = Counter()

        for d in ranked_docs:
            k = _doc_key(d, self.dedup_key_mode)
            title = str(d.title or "").strip()
            group_key = title.lower() if title else k

            if cnt[group_key] >= self.max_per_title:
                continue

            cnt[group_key] += 1
            out_docs.append(d)
            out_scores.append(float(ranked_scores_map.get(k, 0.0)))

            if len(out_docs) >= self.out_top_n:
                break

        t_rerank = time.perf_counter() - t0
        t_total = time.perf_counter() - t_total0
        self._print_profile(
            query=query,
            t_total=t_total,
            t_dense=t_dense,
            t_rerank=t_rerank,
            out_n=len(out_docs),
        )
        return out_docs, np.asarray(out_scores, dtype=float)