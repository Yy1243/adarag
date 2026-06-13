# -*- coding: utf-8 -*-
"""
AdaRAG system prototype.

This module implements the main runtime behavior described in the paper:
- For each slot t, process a batch of queries A_t
- Light retrieval for all queries -> similarity score vectors
- Decide heavy retrieval proportion p_t:
    choose bottom ceil(p_t * |A_t|) queries (low similarity sum) for heavy retrieval
- Prompt optimizer decides document selection by probability vectors x_t, y_t:
    for each query, sample each of top-n docs with prob x_i (or y_i)
- Pipeline parallelism:
    start LLM inference for light queries immediately,
    while heavy retrieval runs concurrently and feeds results in groups

We measure wall-clock latency as end-to-end delay proxy d_t(·).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import math
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from adarag.retrievers.light_tfidf import LightRetrieverTFIDF, RetrievedDoc
from adarag.retrievers.heavy_bm25 import HeavyRetrieverBM25
from adarag.llm.extractive import ExtractiveLLM
from adarag.eval.evaluator import batch_accuracy
from adarag.utils import soft_normalize_nonneg


@dataclass
class SlotResult:
    latency_s: float
    accuracy: float
    p: float
    avg_docs_light: float
    avg_docs_heavy: float
    n_queries: int


class AdaRAGSystem:
    def __init__(
        self,
        light_retriever: LightRetrieverTFIDF,
        heavy_retriever: HeavyRetrieverBM25,
        *,
        n_docs: int,
        llm_mode: str = "extractive",
        hf_model_name: str = "google/flan-t5-small",
        group_size: int = 16,
        max_workers_retrieval: int = 4,
        seed: int = 0,
    ) -> None:
        self.light = light_retriever
        self.heavy = heavy_retriever
        self.n_docs = int(n_docs)
        self.group_size = int(group_size)
        self.max_workers_retrieval = int(max_workers_retrieval)
        self.rng = np.random.RandomState(seed)

        if llm_mode == "hf":
            from adarag.llm.hf_llm import HFLlm  # 按需导入，避免 extractive 模式依赖 torch
            self.llm = HFLlm(model_name=hf_model_name)
        else:
            self.llm = ExtractiveLLM()


    # -----------------------------
    # Prompt selection
    # -----------------------------
    def _select_docs_by_prob(self, docs: List[RetrievedDoc], probs: np.ndarray) -> List[RetrievedDoc]:
        """
        Given ranked docs and a probability vector probs (length n_docs),
        sample each doc i independently with probability probs[i].

        If none selected, return empty -> "no RAG" case.
        """
        k = min(len(docs), probs.shape[0])
        chosen = []
        for i in range(k):
            if self.rng.rand() < float(probs[i]):
                chosen.append(docs[i])
        return chosen

    # -----------------------------
    # Heavy query selection
    # -----------------------------
    def _choose_heavy_queries(self, score_sums: np.ndarray, p: float) -> np.ndarray:
        """
        Choose indices of queries to go heavy retrieval.

        Paper behavior (Sec. III-A):
        - Sum similarity scores (vector) per query
        - Sort by accumulated scores (descending)
        - Low-score queries are subject to heavy retrieval
        """
        n = score_sums.shape[0]
        k = int(math.ceil(float(p) * n))
        if k <= 0:
            return np.zeros(n, dtype=bool)
        if k >= n:
            return np.ones(n, dtype=bool)
        # bottom-k by score
        idx = np.argpartition(score_sums, k - 1)[:k]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask

    # -----------------------------
    # Slot execution
    # -----------------------------
    def run_slot(
        self,
        questions: List[str],
        gold_answers: List[str],
        *,
        p: float,
        x: np.ndarray,
        y: np.ndarray,
    ) -> SlotResult:
        """
        Execute one time-slot.

        Returns:
            latency_s: wall time for the entire batch
            accuracy:  exact-match accuracy over the batch
        """
        assert len(questions) == len(gold_answers)
        n_q = len(questions)
        t0 = time.time()

        # 1) Light retrieval for all queries
        light_docs: List[List[RetrievedDoc]] = []
        score_sums = np.zeros(n_q, dtype=float)
        for i, q in enumerate(questions):
            docs_i, score_vec = self.light.retrieve(q, top_k=self.n_docs)
            light_docs.append(docs_i)
            score_sums[i] = float(np.sum(score_vec))

        # 2) Choose heavy queries by p
        heavy_mask = self._choose_heavy_queries(score_sums, p)
        light_idx = [i for i in range(n_q) if not heavy_mask[i]]
        heavy_idx = [i for i in range(n_q) if heavy_mask[i]]

        # 3) Start heavy retrieval concurrently (pipeline parallelism)
        #    We submit tasks for heavy queries; once a group is ready we can infer them.
        heavy_results: Dict[int, List[RetrievedDoc]] = {}

        def _heavy_task(i: int) -> Tuple[int, List[RetrievedDoc]]:
            q = questions[i]
            docs = self.heavy.retrieve(q, top_k=self.n_docs, candidate_k=50, use_rerank=True)
            return i, docs

        # 4) Infer light queries immediately
        preds: Dict[int, str] = {}

        # Light inference uses x
        for i in light_idx:
            docs = self._select_docs_by_prob(light_docs[i], x)
            preds[i] = self.llm.generate(questions[i], docs)

        # Heavy retrieval and inference:
        # - Use a threadpool for retrieval (CPU)
        # - As each retrieval completes, we buffer results
        # - Once buffered count reaches group_size (or all done), run inference for that group
        buffer_ready: List[int] = []

        with ThreadPoolExecutor(max_workers=self.max_workers_retrieval) as ex:
            futures = [ex.submit(_heavy_task, i) for i in heavy_idx]
            for fut in as_completed(futures):
                i, docs = fut.result()
                heavy_results[i] = docs
                buffer_ready.append(i)

                # If enough for one group, infer that group now
                if len(buffer_ready) >= self.group_size:
                    group = buffer_ready[: self.group_size]
                    buffer_ready = buffer_ready[self.group_size :]
                    for j in group:
                        docs_sel = self._select_docs_by_prob(heavy_results[j], y)
                        preds[j] = self.llm.generate(questions[j], docs_sel)

            # Infer remaining
            for j in buffer_ready:
                docs_sel = self._select_docs_by_prob(heavy_results[j], y)
                preds[j] = self.llm.generate(questions[j], docs_sel)

        # 5) Compute accuracy
        pred_list = [preds[i] for i in range(n_q)]
        acc = float(batch_accuracy(pred_list, gold_answers))

        latency = float(time.time() - t0)

        # Stats: how many docs selected on average
        # (since we sample docs by probability vectors, report expected or realized)
        # We'll report realized selection size.
        doc_counts_light = []
        for i in light_idx:
            doc_counts_light.append(len(self._select_docs_by_prob(light_docs[i], x)))
        doc_counts_heavy = []
        for i in heavy_idx:
            doc_counts_heavy.append(len(self._select_docs_by_prob(heavy_results[i], y))) if i in heavy_results else None

        avg_light = float(np.mean(doc_counts_light)) if doc_counts_light else 0.0
        avg_heavy = float(np.mean(doc_counts_heavy)) if doc_counts_heavy else 0.0

        return SlotResult(
            latency_s=latency,
            accuracy=acc,
            p=float(p),
            avg_docs_light=avg_light,
            avg_docs_heavy=avg_heavy,
            n_queries=n_q,
        )
