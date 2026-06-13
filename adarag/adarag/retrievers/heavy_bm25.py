# -*- coding: utf-8 -*-
"""
Heavy retriever: BM25 (rank-bm25) + optional simple reranking.

Service-free local implementation:
- BM25 candidates via rank-bm25
- Optional rerank by TF-IDF cosine similarity on the candidate set

Scripts expect:
  HeavyBM25Retriever(corpus_path=..., top_n=..., max_passages=...).retrieve(query)
    -> (docs: List[Doc], scores: np.ndarray)
"""

from __future__ import annotations

import json
import re
import numpy as np
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from adarag.data import Doc
from .light_tfidf import RetrievedDoc

_STOP = set(ENGLISH_STOP_WORDS)
_QWORDS = {
    "what","which","who","whom","whose","when","where","why","how",
    "is","are","was","were","be","been","being",
    "do","does","did","done",
    "the","a","an"
}

def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    # 过滤：停用词 + 问句常见词 + 太短的 token
    toks = [t for t in toks if len(t) > 1 and t not in _STOP and t not in _QWORDS]
    return toks


def rewrite_query_heuristic(q: str) -> str:
    toks = _simple_tokenize(q)
    return " ".join(toks) if toks else q


class HeavyRetrieverBM25:
    def __init__(self, docs: List[Doc]) -> None:
        self.docs = docs
        self.corpus_tokens = []
        for d in docs:
            title = str(d.title or "")
            body = str(d.text or "")
            tt = _simple_tokenize(title)
            bt = _simple_tokenize(body)
            self.corpus_tokens.append(tt * 3 + bt)
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.vectorizer = TfidfVectorizer(
                tokenizer=_simple_tokenize,
                preprocessor=None,
                token_pattern=None,
                lowercase=False,
                max_features=50000,
                ngram_range=(1, 2),
            )
        texts = []
        for d in docs:
            title = str(d.title or "")
            body = str(d.text or "")
            texts.append((title + " " + title + " " + body).strip())
        self.doc_tfidf = self.vectorizer.fit_transform(texts)
        self.doc_tfidf = normalize(self.doc_tfidf, norm="l2", axis=1)

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        candidate_k: int = 50,
        use_rerank: bool = True
    ) -> List[RetrievedDoc]:
        q_rw = rewrite_query_heuristic(query)
        q_tokens = _simple_tokenize(q_rw)
        if not q_tokens:
            q_tokens = re.findall(r"[a-z0-9]+", query.lower())
        bm25_scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=float)
        num_docs = len(self.docs)
        candidate_k = max(1, min(candidate_k, num_docs))

        cand_idx = np.argpartition(-bm25_scores, candidate_k - 1)[:candidate_k]
        cand_idx = cand_idx[np.argsort(-bm25_scores[cand_idx])]

        if not use_rerank:
            top_idx = cand_idx[:min(top_k, len(cand_idx))]
            return [
                RetrievedDoc(
                    doc_id=self.docs[i].doc_id,
                    title=self.docs[i].title,
                    text=self.docs[i].text,
                    score=float(bm25_scores[i]),
                )
                for i in top_idx
            ]

        # Rerank by cosine similarity on candidates
        qv = self.vectorizer.transform([q_rw])
        qv = normalize(qv, norm="l2", axis=1)
        sims = (self.doc_tfidf[cand_idx] @ qv.T).toarray().reshape(-1)

        # ---- fusion: normalized BM25 + normalized TF-IDF cosine ----
        bm = bm25_scores[cand_idx].astype(float)
        bm = (bm - bm.min()) / (bm.max() - bm.min() + 1e-9)

        sim = sims.astype(float)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-9)

        final = 0.5 * bm + 0.5 * sim

        order = np.argsort(-final)
        top_local = order[:min(top_k, len(order))]
        top_idx = cand_idx[top_local]

        return [
            RetrievedDoc(
                doc_id=self.docs[i].doc_id,
                title=self.docs[i].title,
                text=self.docs[i].text,
                score=float(final[j]),
            )
            for j, i in zip(top_local, top_idx)
        ]


# -----------------------------
# Scripts-compatible wrapper
# -----------------------------
def _read_jsonl_corpus_as_docs(
    corpus_path: str, max_passages: Optional[int] = None
) -> List[Doc]:
    docs: List[Doc] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_passages is not None and i >= max_passages:
                break
            o = json.loads(line)

            pid = o.get("id", o.get("pid", o.get("doc_id", i)))
            title = o.get("title", "") or ""
            text = o.get("text", o.get("contents", "")) or ""
            full_text = f"{title}\n{text}".strip() if title else str(text).strip()
            docs.append(
                Doc(
                    doc_id=str(pid),
                    title=str(title).strip(),
                    text=str(text).strip(),
                )
            )
    return docs


class HeavyBM25Retriever:
    """
    Used by scripts/run_adarag_real.py
    Must return: (docs, scores)
    """
    def __init__(self, corpus_path: str, top_n: int = 5, max_passages: int | None = None,candidate_k: int = 200, use_rerank: bool = True):
        self.top_n = int(top_n)
        self.candidate_k = int(candidate_k)
        self.use_rerank = bool(use_rerank)
        docs = _read_jsonl_corpus_as_docs(corpus_path, max_passages=max_passages)
        if len(docs) == 0:
            raise ValueError(f"Empty corpus: {corpus_path}")
        self.inner = HeavyRetrieverBM25(docs)

    def retrieve(self, query: str) -> Tuple[List[Doc], np.ndarray]:
        rds = self.inner.retrieve(query, top_k=self.top_n, candidate_k=self.candidate_k, use_rerank=self.use_rerank)
        docs = [
            Doc(
                doc_id=str(rd.doc_id),
                title=str(getattr(rd, "title", "") or ""),
                text=str(rd.text),
            )
            for rd in rds
        ]
        scores = np.asarray([float(rd.score) for rd in rds], dtype=np.float32)
        return docs, scores
