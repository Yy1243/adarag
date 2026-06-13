# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from adarag.data import Doc


@dataclass
class Passage:
    pid: str
    title: str
    text: str


def _read_jsonl(path: str, max_passages: Optional[int] = None) -> List[Passage]:
    out: List[Passage] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_passages is not None and i >= max_passages:
                break
            obj = json.loads(line)
            out.append(
                Passage(
                    pid=str(obj.get("id", obj.get("pid", i))),
                    title=str(obj.get("title", "")),
                    text=str(obj.get("text", obj.get("contents", ""))),
                )
            )
    return out


class FaissHNSWRetriever:
    """
    Light retriever: HNSW index over dense embeddings.

    Corpus format: JSONL, each line:
      {"id": "...", "title": "...", "text": "..."}
    """

    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        top_n: int = 10,
        device: str = "cuda",
        profile: bool = False,
    ):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.top_n = int(top_n)
        self.device = device

        self.profile = bool(profile) or os.environ.get("ADARAG_FAISS_PROFILE", "0") == "1"

        self._passages: List[Passage] = []
        self._index = None
        self._encoder = None

    def _lazy_imports(self):
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401

    def _print_profile(
        self,
        query: str,
        k: int,
        total_s: float,
        encode_s: float,
        search_s: float,
        build_docs_s: float,
        n_docs: int,
    ) -> None:
        if not self.profile:
            return
        print(
            "[FaissHNSW][profile] "
            f"total={total_s:.3f}s "
            f"encode={encode_s:.3f}s "
            f"search={search_s:.3f}s "
            f"build_docs={build_docs_s:.3f}s "
            f"k={k} "
            f"n_docs={n_docs} "
            f"device={self.device} "
            f"q={query[:100]!r}",
            flush=True,
        )

    def load_or_build(self, rebuild: bool = False, max_passages: Optional[int] = None) -> None:
        self._lazy_imports()
        import faiss
        from sentence_transformers import SentenceTransformer

        self._passages = _read_jsonl(self.corpus_path, max_passages=max_passages)
        if len(self._passages) == 0:
            raise ValueError(f"Empty corpus: {self.corpus_path}")

        # 只有 Jina 这类依赖本地自定义实现的模型才需要 trust_remote_code=True
        is_jina = "jina" in str(self.embedding_model).lower()

        self._encoder = SentenceTransformer(
            self.embedding_model,
            device=self.device,
            trust_remote_code=is_jina,
            local_files_only=True,
        )

        if (not rebuild) and os.path.exists(self.index_path):
            self._index = faiss.read_index(self.index_path)
            try:
                self._index.hnsw.efSearch = 128
            except Exception:
                pass
            return

        texts = [f"{p.title}\n{p.text}".strip() for p in self._passages]
        emb = self._encoder.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dim = emb.shape[1]
        index = faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(emb)

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(index, self.index_path)
        self._index = index

    def _search(self, query: str, k: int) -> Tuple[List[Doc], np.ndarray]:
        if self._index is None or self._encoder is None:
            raise RuntimeError("Call load_or_build() first.")

        k = int(k)
        t_total0 = time.perf_counter()

        t0 = time.perf_counter()
        q_emb = self._encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores, idx = self._index.search(q_emb, k)
        t_search = time.perf_counter() - t0

        scores = scores[0]
        idx = idx[0]

        t0 = time.perf_counter()
        docs: List[Doc] = []
        for pi in idx.tolist():
            if pi < 0 or pi >= len(self._passages):
                continue

            p = self._passages[pi]
            full_text = f"{p.title}\n{p.text}".strip() if p.title else p.text

            docs.append(
                Doc(
                    doc_id=p.pid,
                    text=full_text,
                    title=p.title,
                )
            )
        t_build_docs = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total0
        self._print_profile(
            query=query,
            k=k,
            total_s=t_total,
            encode_s=t_encode,
            search_s=t_search,
            build_docs_s=t_build_docs,
            n_docs=len(docs),
        )

        return docs, np.asarray(scores, dtype=float)

    def retrieve(self, query: str) -> Tuple[List[Doc], np.ndarray]:
        return self._search(query, self.top_n)

    def retrieve_k(self, query: str, top_n: int) -> Tuple[List[Doc], np.ndarray]:
        return self._search(query, int(top_n))