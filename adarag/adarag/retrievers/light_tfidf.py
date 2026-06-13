# -*- coding: utf-8 -*-
"""
Light retriever: TF-IDF cosine similarity.

This module is designed to be:
- service-free (no Faiss/Elastic needed)
- fast enough for prototype testing
- easy to swap with Faiss-HNSW later

In the paper:
- light retriever is HNSW on a vector DB (Sec. III-A).
Here we implement a faithful replacement:
- build TF-IDF matrix for docs
- for a query, compute cosine similarity to all docs
- return top-K docs and their similarity scores
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from adarag.data import Doc


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    title: str = ""


class LightRetrieverTFIDF:
    def __init__(self, docs: List[Doc], *, max_features: int = 50000, ngram_range=(1, 2)) -> None:
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words="english",
        )
        # Fit & transform docs
        doc_texts = [d.full_text for d in docs]
        self.doc_tfidf = self.vectorizer.fit_transform(doc_texts)
        # Normalize so dot-product equals cosine similarity
        self.doc_tfidf = normalize(self.doc_tfidf, norm="l2", axis=1)

    def retrieve(self, query: str, top_k: int) -> Tuple[List[RetrievedDoc], np.ndarray]:
        """
        Returns:
            docs: list of RetrievedDoc length top_k
            score_vec: np.ndarray of length top_k (similarity scores)
        """
        q = self.vectorizer.transform([query])
        q = normalize(q, norm="l2", axis=1)
        # cosine similarity via dot product
        sims = (self.doc_tfidf @ q.T).toarray().reshape(-1)  # shape (num_docs,)
        if top_k >= len(self.docs):
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        out = []
        score_vec = []
        for i in top_idx[:top_k]:
            d = self.docs[int(i)]
            s = float(sims[int(i)])
            out.append(
                RetrievedDoc(
                    doc_id=d.doc_id,
                    title=d.title,
                    text=d.text,
                    score=s,
                )
            )
            score_vec.append(s)
        return out, np.asarray(score_vec, dtype=float)
