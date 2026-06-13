# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from adarag.data import Doc

def _doc_key(d: Doc, mode: str = "title_text") -> str:
    mode = (mode or "title_text").lower().strip()

    if mode == "doc_id":
        return str(d.doc_id or "")
    if mode == "title_text":
        return (d.full_text or "")[:256]

    raise ValueError(f"Unsupported dedup_key_mode={mode!r}")


def _norm_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = np.asarray(list(score_map.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin + 1e-12:
        return {k: 0.0 for k in score_map.keys()}
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in score_map.items()}


def _weighted_rrf(rank_lists: List[Tuple[List[str], float]], k: int = 60) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for keys, w in rank_lists:
        if not keys or w <= 0:
            continue
        seen = set()
        uniq = []
        for key in keys:
            if key not in seen:
                uniq.append(key)
                seen.add(key)

        for r, key in enumerate(uniq):
            out[key] = out.get(key, 0.0) + float(w) / float(k + r + 1)
    return out


class HybridDenseBM25CrossRerankRetriever:
    """
    Dense + BM25 rescue + rerank heavy retriever.

    Main idea:
      1. Dense provides the main semantic ranking.
      2. BM25 cross_fields provides lexical/entity/number rescue candidates.
      3. BM25 top candidates are separately reranked.
      4. Final top10 explicitly mixes dense evidence and BM25 rescue evidence.
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        reranker_model: str,
        dense_top_n: int = 60,
        bm25_top_n: int = 100,
        out_top_n: int = 50,
        candidate_pool_n: int = 80,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 0.15,
        rerank_k: int = 50,
        rerank_weight: float = 0.15,
        rerank_batch_size: int = 8,
        rerank_max_length: int = 512,
        rerank_device: str = "cpu",
        max_per_title: int = 2,
        dedup_key_mode: str = "title_text",
        dense_prompt_n: int = 8,
        rescue_top_n: int = 2,
        rescue_rerank_k: int = 100,
        mix_top_n: int = 10,
        profile: bool = False,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.reranker_model = reranker_model

        self.dense_top_n = int(dense_top_n)
        self.bm25_top_n = int(bm25_top_n)
        self.out_top_n = int(out_top_n)
        self.candidate_pool_n = int(candidate_pool_n)

        self.rrf_k = int(rrf_k)
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)

        self.rerank_k = int(rerank_k)
        self.rerank_weight = float(rerank_weight)
        self.rerank_batch_size = int(max(1, rerank_batch_size))
        self.rerank_max_length = int(rerank_max_length)
        self.rerank_device = rerank_device

        self.max_per_title = int(max(1, max_per_title))
        self.dedup_key_mode = (dedup_key_mode or "title_text").lower().strip()

        self.dense_prompt_n = int(max(0, dense_prompt_n))
        self.rescue_top_n = int(max(0, rescue_top_n))
        self.rescue_rerank_k = int(max(0, rescue_rerank_k))
        self.mix_top_n = int(max(1, mix_top_n))

        self.profile = bool(profile)
        self._reranker = None

        if hasattr(self.bm25, "top_n"):
            try:
                self.bm25.top_n = max(int(getattr(self.bm25, "top_n")), self.bm25_top_n)
            except Exception:
                pass

    def _lazy_load_reranker(self):
        if self._reranker is not None:
            return self._reranker

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

    def _rerank_key_docs(
        self,
        query: str,
        keys: List[str],
        by_key: Dict[str, Doc],
        limit: int,
    ) -> List[str]:
        if not keys or limit <= 0 or not self.reranker_model:
            return keys[:limit]

        cand_keys = []
        seen = set()
        for k in keys:
            if k in by_key and k not in seen:
                cand_keys.append(k)
                seen.add(k)
            if len(cand_keys) >= limit:
                break

        if not cand_keys:
            return []

        reranker = self._lazy_load_reranker()
        pairs = [(query, by_key[k].full_text or "") for k in cand_keys]
        scores = reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(scores, dtype=float).reshape(-1)

        ranked = sorted(zip(cand_keys, scores), key=lambda x: -x[1])
        return [k for k, _ in ranked]

    def _title_key(self, d: Doc) -> str:
        title = str(d.title or "").strip()
        return title.lower() if title else _doc_key(d, self.dedup_key_mode)

    def _print_profile(
        self,
        query: str,
        t_total: float,
        t_dense: float,
        t_bm25: float,
        t_fuse: float,
        t_main_rerank: float,
        t_rescue_rerank: float,
        dense_n: int,
        bm25_n: int,
        out_n: int,
    ) -> None:
        if not self.profile:
            return
        print(
            "[HybridDenseBM25CrossRerank][profile] "
            f"total={t_total:.3f}s "
            f"dense={t_dense:.3f}s "
            f"bm25={t_bm25:.3f}s "
            f"fuse={t_fuse:.3f}s "
            f"main_rerank={t_main_rerank:.3f}s "
            f"rescue_rerank={t_rescue_rerank:.3f}s "
            f"dense_n={dense_n} "
            f"bm25_n={bm25_n} "
            f"out_n={out_n} "
            f"q={query[:100]!r}",
            flush=True,
        )

    def retrieve(self, query: str) -> Tuple[List[Doc], np.ndarray]:
        t_total0 = time.perf_counter()

        if not hasattr(self.dense, "retrieve_k"):
            raise RuntimeError("dense_retriever must implement retrieve_k(query, top_n).")

        # 1. Dense main recall
        t0 = time.perf_counter()
        docs_d, scores_d = self.dense.retrieve_k(query, self.dense_top_n)
        t_dense = time.perf_counter() - t0
        docs_d = docs_d or []

        # 2. BM25 lexical rescue recall
        t0 = time.perf_counter()
        docs_b, scores_b = self.bm25.retrieve(query)
        t_bm25 = time.perf_counter() - t0
        docs_b = (docs_b or [])[: self.bm25_top_n]

        # 3. Build rank lists and doc map
        t0 = time.perf_counter()

        by_key: Dict[str, Doc] = {}
        dense_keys: List[str] = []
        bm25_keys: List[str] = []

        for d in docs_d:
            k = _doc_key(d, self.dedup_key_mode)
            if k not in by_key:
                by_key[k] = d
            if k not in dense_keys:
                dense_keys.append(k)

        for d in docs_b:
            k = _doc_key(d, self.dedup_key_mode)
            if k not in by_key:
                by_key[k] = d
            if k not in bm25_keys:
                bm25_keys.append(k)

        fused_scores = _weighted_rrf(
            [
                (dense_keys, self.dense_weight),
                (bm25_keys, self.bm25_weight),
            ],
            k=self.rrf_k,
        )

        base_ranked = sorted(fused_scores.items(), key=lambda x: -x[1])
        base_keys = [k for k, _ in base_ranked]

        candidate_keys = base_keys[: min(self.candidate_pool_n, len(base_keys))]
        t_fuse = time.perf_counter() - t0

        if not candidate_keys:
            return [], np.asarray([], dtype=float)

        # 4. Conservative main rerank on fused candidates
        t0 = time.perf_counter()

        rrf_norm = _norm_map({k: fused_scores.get(k, 0.0) for k in candidate_keys})

        rerank_n = min(self.rerank_k, len(candidate_keys))
        head_keys = candidate_keys[:rerank_n]
        tail_keys = candidate_keys[rerank_n:]

        final_score_map: Dict[str, float] = {}

        if self.reranker_model and rerank_n > 0 and self.rerank_weight > 0:
            reranker = self._lazy_load_reranker()
            pairs = [(query, by_key[k].full_text or "") for k in head_keys]
            rr = reranker.predict(
                pairs,
                batch_size=self.rerank_batch_size,
                show_progress_bar=False,
            )
            rr = np.asarray(rr, dtype=float).reshape(-1)
            rr_norm = _norm_map({k: float(s) for k, s in zip(head_keys, rr)})

            main_items = []
            for k in head_keys:
                base_s = float(rrf_norm.get(k, 0.0))
                rr_s = float(rr_norm.get(k, 0.0))
                final_s = (1.0 - self.rerank_weight) * base_s + self.rerank_weight * rr_s
                final_score_map[k] = final_s
                main_items.append((k, final_s))

            main_items.sort(key=lambda x: -x[1])
            main_ranked_keys = [k for k, _ in main_items] + tail_keys
        else:
            main_ranked_keys = head_keys + tail_keys
            for k in main_ranked_keys:
                final_score_map[k] = float(rrf_norm.get(k, 0.0))

        # Candidate-pool outside fallback
        candidate_set = set(candidate_keys)
        rest_keys = [k for k in base_keys if k not in candidate_set]
        main_ranked_keys = main_ranked_keys + rest_keys

        t_main_rerank = time.perf_counter() - t0

        # 5. BM25 top100 separate rerank rescue
        t0 = time.perf_counter()

        bm25_reranked_keys = self._rerank_key_docs(
            query=query,
            keys=bm25_keys,
            by_key=by_key,
            limit=min(self.rescue_rerank_k, len(bm25_keys)),
        )

        t_rescue_rerank = time.perf_counter() - t0

        # 6. Explicit top10 evidence mixing:
        #    dense/rerank main evidence + BM25 rescue evidence
        mixed_top: List[str] = []
        seen = set()
        title_cnt = Counter()

        def _can_add(k: str) -> bool:
            if k not in by_key or k in seen:
                return False
            tk = self._title_key(by_key[k])
            return title_cnt[tk] < self.max_per_title

        def _add(k: str) -> bool:
            if not _can_add(k):
                return False
            mixed_top.append(k)
            seen.add(k)
            tk = self._title_key(by_key[k])
            title_cnt[tk] += 1
            return True

        # A. main top evidence
        for k in main_ranked_keys:
            if len(mixed_top) >= self.dense_prompt_n:
                break
            _add(k)

        # B. BM25 rescue evidence
        rescue_added = 0
        main_front = set(mixed_top)
        for k in bm25_reranked_keys:
            if rescue_added >= self.rescue_top_n:
                break
            if k in main_front:
                continue
            if _add(k):
                rescue_added += 1

        # C. Fill top10 with main ranking
        for k in main_ranked_keys:
            if len(mixed_top) >= self.mix_top_n:
                break
            _add(k)

        # D. Final ranking = mixed top10 + remaining main + remaining bm25
        mixed_set = set(mixed_top)
        ranked_keys = (
            mixed_top
            + [k for k in main_ranked_keys if k not in mixed_set]
            + [k for k in bm25_reranked_keys if k not in mixed_set]
        )

        # 7. Output with title diversity
        out_docs: List[Doc] = []
        out_scores: List[float] = []
        out_seen = set()
        out_title_cnt = Counter()

        for k in ranked_keys:
            if k not in by_key or k in out_seen:
                continue
            d = by_key[k]
            tk = self._title_key(d)
            if out_title_cnt[tk] >= self.max_per_title:
                continue

            out_seen.add(k)
            out_title_cnt[tk] += 1
            out_docs.append(d)

            if k in mixed_set:
                out_scores.append(float(2.0 + final_score_map.get(k, fused_scores.get(k, 0.0))))
            else:
                out_scores.append(float(final_score_map.get(k, fused_scores.get(k, 0.0))))

            if len(out_docs) >= self.out_top_n:
                break

        t_total = time.perf_counter() - t_total0
        self._print_profile(
            query=query,
            t_total=t_total,
            t_dense=t_dense,
            t_bm25=t_bm25,
            t_fuse=t_fuse,
            t_main_rerank=t_main_rerank,
            t_rescue_rerank=t_rescue_rerank,
            dense_n=len(docs_d),
            bm25_n=len(docs_b),
            out_n=len(out_docs),
        )

        return out_docs, np.asarray(out_scores, dtype=float)