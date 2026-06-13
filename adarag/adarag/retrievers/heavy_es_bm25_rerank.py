# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import time
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import requests

from adarag.data import Doc

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def _simplify_query(q: str) -> str:             #将自然语言问题简化为更适合 BM25 的关键词查询。
    q = (q or "").lower()
    toks = re.findall(r"[a-z0-9]+", q)          #正则表达式从字符串里"抓"出所有连续的字母和数字。
    toks = [t for t in toks if len(t) >= 2 and t not in _STOP]
    return " ".join(toks) if toks else ((q or "").strip())      #用空格隔开然后拼成一个完整的字符串


class ElasticBM25RerankRetriever:
    def __init__(
        self,
        es_url: str,
        index_name: str,
        top_n: int = 10,
        bm25_k: int = 50,       #从 ES 召回的候选池大小（默认 50）。ES 实际会返回 50 篇，然后可能经过重排/去重，最后只输出 top_n 篇。
        reranker_model: Optional[str] = None,
        device: str = "cuda",
        collapse_field: Optional[str] = None,           #ES 的 field collapsing 字段。如果指定（如 "title"），ES 会对该字段值相同的文档进行折叠，每个值只返回最相关的一篇。用于避免同一来源的文档刷屏。
        max_per_title: int = 5,                         #同一标题最多保留几篇
        minimum_should_match: Optional[str] = None,         #最少应该匹配多少查询词。比如 "2<80%" 表示少于 2 个词时必须全匹配，多于 2 个词时至少匹配 80%。用于控制召回的严格程度。
        request_timeout: float = 30.0,  
        query_mode: str = "cross_fields",
        # --------- bool phrase knobs ---------
        phrase_slop: int = 2,           #短语匹配的宽松度（默认 2）。match_phrase 允许查询词之间最多间隔几个词
        # --------- rerank knobs ---------
        rerank_k: Optional[int] = 50,
        rerank_batch_size: int = 32,
        rerank_max_doc_chars: int = 2000,       #	重排序时截断文档的最大字符数（默认 2000）。防止超长文档把显存撑爆，也加快推理速度。
        profile: bool = False,
    ):
        self.es_url = es_url.rstrip("/")
        self.index_name = index_name
        self.top_n = int(top_n)
        self.bm25_k = int(bm25_k)
        self.reranker_model = reranker_model
        self.device = device
        self.collapse_field = collapse_field
        self.max_per_title = int(max_per_title)
        self.minimum_should_match = minimum_should_match
        self.request_timeout = float(request_timeout)
        self.query_mode = (query_mode or "cross_fields").strip().lower()
        if self.query_mode not in {
            "cross_fields",
            "text_only_raw",
            "text_only_simple",
        }:
            raise ValueError(
                "query_mode must be one of: cross_fields|text_only_raw|text_only_simple, "
                f"got: {query_mode}"
            )

        self.phrase_slop = int(max(0, phrase_slop))
        self.rerank_k = None if rerank_k is None else int(rerank_k)
        self.rerank_batch_size = int(max(1, rerank_batch_size))
        self.rerank_max_doc_chars = int(max(0, rerank_max_doc_chars))
        self.profile = bool(profile)
        self._sess: Optional[requests.Session] = None
        self._reranker = None

    def _lazy_init(self):
        if self._sess is None:
            self._sess = requests.Session()             #sess是Session对象，理解他是一条"不打断的专线"。第一次用时花点时间"拨号接通"，之后所有请求都走这条老线路，省去了反复"握手打招呼"的开销。
            self._sess.headers.update({"Content-Type": "application/json"})             #headers.update(...)给所有请求加上默认请求头，声明发送的是 JSON 数据，这样后面发 ES 查询时就不用每次都手动带 Content-Type 了。
        if self.reranker_model and self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model, device=self.device)
            try:
                self._reranker.model.eval()
            except Exception:
                pass
            print(f"[HeavyES] reranker={self.reranker_model} device={self.device} loaded=True")

    @staticmethod
    def _diversify_hits_by_title(hits: List[dict], max_per_title: int = 5) -> List[dict]:           #按标题对 ES 召回结果做"去重"
        if max_per_title is None or max_per_title <= 0:
            return hits
        cnt = defaultdict(int)      #访问一个不存在的key时会自动调用 int() 来生成默认值,是一个字典形式作用是统计每个标题出现的次数，初始的时候为，他是会边跑边长的
        out = []
        for h in hits:             
            src = h.get("_source", {}) or {}            #遍历ES返回的每一个 hit（命中结果），从 hit 里取出 _source 字段（ES中返回的字段）
            t = (src.get("title", "") or "").strip()
            if cnt[t] >= max_per_title:
                continue
            cnt[t] += 1
            out.append(h)
        return out

    def _es_search(self, body: dict) -> dict:
        assert self._sess is not None
        url = f"{self.es_url}/{self.index_name}/_search"

        r = self._sess.post(url, json=body, timeout=(3.0, self.request_timeout))            #第一个参数表示连接最多等3秒，ES计算最多等30秒

        if r.status_code >= 400:
            print("\n[HeavyES][ERROR] ES returned HTTP", r.status_code)
            print("[HeavyES] URL:", url)
            try:
                body_txt = json.dumps(body, ensure_ascii=False)
            except Exception:
                body_txt = str(body)
            print("[HeavyES] Request body (head 4000 chars):")
            print(body_txt[:4000])
            print("[HeavyES] Response text (head 4000 chars):")
            print((r.text or "")[:4000])
            r.raise_for_status()

        return r.json()         #把ES返回的 JSON 字符串解析成 Python 字典，转化成什么形式取决于JSON 里的写法，json里面是{...} 花括号，键值对则转的就是字典，json里面是[...] 方括号，则转为列表。ES 的 _search API 返回的 JSON 最外层是花括号 {}，所以这里转成了字典。这个字典里通常会有一个 "hits" 键，对应的值又是一个字典，里面有个 "hits" 键，对应的值才是我们真正关心的那个列表，里面每个元素就是一个命中结果（hit）。所以后面我们经常看到 resp.get("hits", {}).get("hits", []) 这样的写法，就是在从 ES 的响应里取出那个命中结果列表。

    def _make_doc(self, h: dict) -> Doc:
        src = h.get("_source", {}) or {}

        title = str(src.get("title", "") or "").strip()
        text = str(src.get("text", "") or "").strip()

        return Doc(
            doc_id=str(h.get("_id", src.get("doc_id", ""))),
            title=title,
            text=text,
        )

    def _build_es_query(self, query: str) -> dict:
        """
        构造 ES 查询。

        raw_best_fields:
          旧版逻辑，直接用原始自然语言问题做 best_fields。

        simple_best_fields:
          先去掉 question words / stop words，再 best_fields。

        cross_fields:
          推荐模式。把 title/text 当成联合字段，让标题和正文共同贡献匹配。
          你的 probe 里该模式显著强于 raw_best_fields。

        strict_msm:
          简化 query + minimum_should_match，减少噪声。

        bool_phrase_boost:
          效果通常更强但明显更慢，不建议在线主流程默认使用。
        """
        q_raw = query or ""
        q_simple = _simplify_query(q_raw)

        #模式不同分数计算方式不同，可能导致哪些文档能进 top-k 也不同"。
        if self.query_mode == "cross_fields":               #multi_match 查询，query 是简化后的查询词，fields 指定了要搜索的字段和权重（标题权重是正文的3倍），type 是 cross_fields 表示把 title/text 当成一个联合字段来匹配(区别于best field将title和text分别计算得分)，operator 是 or 表示查询词之间是或关系。最后如果设置了 minimum_should_match 就加上这个参数。
            mm: Dict[str, object] = {
                "query": q_simple,
                "fields": ["title^3", "text"],
                "type": "cross_fields",
                "operator": "or",
            }
            if self.minimum_should_match:
                mm["minimum_should_match"] = self.minimum_should_match
            return {"multi_match": mm}

        if self.query_mode == "text_only_raw":
            return {
                "match": {
                    "text": {
                        "query": q_raw,
                        "operator": "or",
                    }
                }
            }

        if self.query_mode == "text_only_simple":
            return {
                "match": {
                    "text": {
                        "query": q_simple,
                        "operator": "or",
                    }
                }
            }

        raise ValueError(f"Unknown query_mode={self.query_mode}")

    def retrieve(self, query: str) -> Tuple[List[Doc], np.ndarray]:
        self._lazy_init()

        # -------------------------
        # ES BM25
        # -------------------------
        t_es0 = time.time()

        body = {            
            "_source": ["title", "text"],           #返回字段
            "query": self._build_es_query(query),     # 构造具体的查询 DSL
            "track_total_hits": False,                  #会统计总共有多少文档匹配查询
        }   

        hits: List[dict] = []       #空列表用于接受ES返回的文档。

        if self.collapse_field:         #让 ES 按某个字段把结果“分组”，每组只返回若干条最相关的 passage，从es服务层面进行处理
            per_title = max(1, self.max_per_title)
            outer_size = max(1, int(np.ceil(self.bm25_k / per_title)))

            body["size"] = outer_size
            body["collapse"] = {
                "field": self.collapse_field,
                "inner_hits": {
                    "name": "passages",
                    "size": per_title,
                    "_source": ["title", "text"],
                    "sort": [{"_score": "desc"}],
                },
            }

            try:
                resp = self._es_search(body)                #封装请求ES的函数，发送请求并返回结果。这里是第一次真正调用 ES 的地方，之前只是构造了查询语句。
            except requests.HTTPError:
                print("[HeavyES] collapse failed -> fallback to non-collapse BM25 search.")
                self.collapse_field = None
                body.pop("collapse", None)
                body["size"] = self.bm25_k
                resp = self._es_search(body)

            raw_hits = resp.get("hits", {}).get("hits", []) or []       #解析响应
            if not raw_hits:
                return [], np.zeros((0,), dtype=float)

            for h in raw_hits:
                ih = (
                    (h.get("inner_hits", {}) or {})
                    .get("passages", {})
                    .get("hits", {})
                    .get("hits", [])
                )
                if ih:
                    hits.extend(ih)
                else:
                    hits.append(h)

            hits = hits[: self.bm25_k]

        else:
            body["size"] = self.bm25_k
            resp = self._es_search(body)
            hits = resp.get("hits", {}).get("hits", []) or []
            if not hits:
                return [], np.zeros((0,), dtype=float)

            if self.max_per_title and self.max_per_title > 0:
                hits = self._diversify_hits_by_title(
                    hits,
                    max_per_title=self.max_per_title,
                )
            hits = hits[: self.bm25_k]

        t_es = time.time() - t_es0

        if not hits:
            return [], np.zeros((0,), dtype=float)

        docs = [self._make_doc(h) for h in hits]
        bm25_scores = np.asarray([float(h.get("_score", 0.0)) for h in hits], dtype=float)

        # -------------------------
        # No reranker -> top_n by BM25
        # -------------------------
        if self._reranker is None:                                          #重排序器如果没有加载，就直接根据 BM25 分数排序，取 top_n 返回。注意这里的 top_n 是最终输出的文档数量，而 bm25_k 是从 ES 召回的候选池大小，通常 bm25_k 会大于 top_n，这样才有重排序的意义。
            order = np.argsort(-bm25_scores)[: min(self.top_n, len(docs))]
            if self.profile:
                print(
                    f"[HeavyES][profile] mode={self.query_mode} "
                    f"es={t_es:.3f}s rerank=0.000s K={len(docs)}"
                )
            return [docs[i] for i in order], bm25_scores[order]

        # -------------------------
        # CrossEncoder rerank
        # -------------------------
        t_rr0 = time.time()                 #有重排序

        rk = self.bm25_k if self.rerank_k is None else min(self.bm25_k, self.rerank_k)
        rk = min(rk, len(docs))
        cand_idx = np.argsort(-bm25_scores)[:rk]

        pairs = []
        for i in cand_idx:
            txt = docs[i].full_text
            if self.rerank_max_doc_chars > 0 and len(txt) > self.rerank_max_doc_chars:
                txt = txt[: self.rerank_max_doc_chars]
            pairs.append((query, txt))

        rr = self._reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False,
        )
        rr = np.asarray(rr, dtype=float).reshape(-1)

        t_rr = time.time() - t_rr0

        take = min(self.top_n, rr.size)
        top_local = np.argsort(-rr)[:take]
        picked_idx = cand_idx[top_local]

        if self.profile:
            print(
                f"[HeavyES][profile] mode={self.query_mode} "
                f"es={t_es:.3f}s rerank={t_rr:.3f}s "
                f"bm25_k={self.bm25_k} rerank_k={rk} batch={self.rerank_batch_size}"
            )

        return [docs[i] for i in picked_idx], rr[top_local]