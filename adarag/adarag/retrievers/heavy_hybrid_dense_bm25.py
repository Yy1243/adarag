# -*- coding: utf-8 -*-                                             
from __future__ import annotations                                   

from typing import List, Tuple, Dict, Optional                       
import numpy as np                                                 
import re                                                            # 导入正则表达式模块
import time                                                          
from collections import Counter                                      # 导入计数器，用于词频统计

from adarag.data import Doc                                         
from sentence_transformers import CrossEncoder                       # 导入CrossEncoder用于重排序

_STOP = {                                                            # 定义停用词集合（过滤无意义词汇）
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

_REL_MARKERS = {"of", "in", "for", "from", "by", "on", "about", "with", "into", "over", "under"}  # 定义关系介词标记集合，用于提取查询焦点

_VERB_HINTS = {                                                      # 定义关系动词提示集合
    "wrote", "write", "written", "sang", "sing", "sung", "directed", "direct", "director",         
    "starred", "starring", "stars", "star", "played", "play", "plays", "founded", "founder",      
    "married", "marry", "invented", "invent", "inventor", "discovered", "discover", "owned", "own", "owner",  
    "born", "died", "lead", "leads", "leading",                    
}                                                                   


def simplify_query(q: str) -> str:                                   # 定义基础查询简化函数
    q = (q or "").lower()                                               
    q = re.sub(r"[^a-z0-9\s]", " ", q)                                      # 非字母数字替换为空格
    toks = [t for t in q.split() if t and (t not in _STOP)]                     # 分词并过滤停用词，展开过后相当于先创建一个空列表，然后逐个分词然后不是停用词就放进去
    return " ".join(toks) if toks else (q.strip() or "")                            # 返回处理后的查询或原查询，展开过后相当于先判断toks列表是否非空，非空的话用空格连接所有词，返回字符串。如果为空的话先去掉q首尾空白，然后判断是否为空，空扔有留


def _tokenize(text: str) -> List[str]:                               # 定义基础分词函数（模块内部使用），模块内部使用表示只在当前文件内部用
    text = (text or "").lower()                                         
    toks = re.findall(r"[a-z0-9]+", text)                                   # 正则提取连续字母数字序列，比如abcd,3452,1c78这种都可以，返回值是一个列表用逗号进行隔开
    return [t for t in toks if len(t) >= 2]                                     


def _unique_keep_order(xs: List[str]) -> List[str]:                  # 定义去重保序函数（模块内部使用），用于重写和组装扩展去重
    seen = set()                                                        
    out = []                                                                # 创建输出列表，用 set 快速判断"是否见过"，用 list 保证"第一次见的顺序"。
    for x in xs:                                                              
        if x and x not in seen:                                              
            out.append(x)                                                           # 添加到输出
            seen.add(x)                                                                 # 标记为已见
    return out                                                       


def _extract_number_tokens(q: str) -> List[str]:                     # 定义提取数字token函数
    return re.findall(r"\b\d[\d\-/:,.]*\b", q or "")                        # 匹配数字开头，后可跟数字、横杠、斜杠、冒号、逗号、点，碰到空格会停，一句话里面可能会提取出多个数字token


def _extract_quoted_phrases(q: str) -> List[str]:                    # 定义提取引号短语函数,捕获如书名，歌名，电影名这些实体
    out = []                                                        
    for pat in [r'"([^"]+)"', r"'([^']+)'"]:                         # 匹配双引号和单引号内容
        out.extend(re.findall(pat, q or ""))                                
    return _unique_keep_order([" ".join(_tokenize(x)) for x in out if x.strip()])       # 分词处理后去重保序返回


def _content_tokens(q: str) -> List[str]:                            # 定义提取内容词函数（去除停用词），去掉虚词后剩下的实义词，用于计算每个词在文档中的出现次数	
    return [t for t in _tokenize(q) if t not in _STOP]                   


def _extract_tail_focus_spans(q: str, max_len: int = 5) -> List[str]:  
    toks = _tokenize(q)                                              
    spans: List[str] = []                                         
    for i, tok in enumerate(toks):                                   
        if tok in _REL_MARKERS and i + 1 < len(toks):                # 策略1介词后面的名词可能会比较关键进行提取
            tail = [t for t in toks[i + 1:] if t not in _STOP]          # 提取介词后的非停用词尾部
            if 1 <= len(tail) <= max_len:                                   
                spans.append(" ".join(tail))                                    
            elif len(tail) > max_len:                                
                spans.append(" ".join(tail[-max_len:]))              # 只取最后max_len个词

    cue_patterns = [                                                 # 定义提示词模式列表
        ["song"], ["book"], ["movie"], ["film"], ["tv", "series"], ["season"], ["episode"],  
        ["lead", "singer"], ["capital"], ["setting"],                
    ]                                                                
    for pat in cue_patterns:                                         
        m = None                                                     
        for i in range(0, len(toks) - len(pat) + 1):                 # 滑动窗口匹配，len(toks) - len(pat) + 1表示滑动窗口的最后一个起始位置，逐个匹配提示词
            if toks[i:i + len(pat)] == pat:                             # 完全匹配模式
                m = i + len(pat)                                            #记录模式结束后的第一个位置
        if m is not None and m < len(toks):                          # 如果匹配成功且后面有词
            tail = [t for t in toks[m:] if t not in _STOP]                  # 提取模式后的非停用词
            if tail:                                                
                spans.append(" ".join(tail[:max_len]))           
                spans.append(" ".join(tail[-max_len:]))                

    cont = [t for t in toks if t not in _STOP]                      # 获取全部非停用词
    for n in [4, 3, 2]:                                                 # 取尾部2/3/4-gram
        if len(cont) >= n:                                                  # 长度足够
            spans.append(" ".join(cont[-n:]))                                   # 加入最后n个词
    return _unique_keep_order([s for s in spans if s.strip()])                      #去重保序返回非空片段


def _extract_relation_phrase(q: str, max_len: int = 4) -> List[str]:  # 定义提取关系短语函数
    toks = _tokenize(q)                                              # 对查询分词
    out = []                                                         # 初始化输出
    cont = [t for t in toks if t not in _STOP]                      # 获取非停用词
    for i in range(len(cont)):                                       # 滑动窗口起点
        for j in range(i + 1, min(len(cont), i + max_len) + 1):     # 滑动窗口终点
            span = cont[i:j]                                         # 获取窗口内词
            if any(t in _VERB_HINTS for t in span):                  # 如果包含关系动词
                out.append(" ".join(span))                           # 作为关系短语加入
    patterns = [["lead", "singer"], ["turn", "vampire"], ["tv", "series"], ["season"], ["episode"]]  # 固定模式
    for pat in patterns:                                             # 遍历固定模式
        for i in range(0, len(toks) - len(pat) + 1):                 # 在原token序列中匹配
            if toks[i:i + len(pat)] == pat:                          # 完全匹配
                out.append(" ".join(pat))                            # 加入模式
    return _unique_keep_order(out)                                   # 去重保序返回


def rewrite_query_keyword_v2(q: str, max_terms: int = 12) -> str:    # 定义关键词查询重写函数V2
    q = q or ""                                                      # 处理None为空字符串
    numbers = _extract_number_tokens(q)                              # ①提取数字
    quoted = _extract_quoted_phrases(q)                              # ②提取引号短语
    tail_spans = _extract_tail_focus_spans(q, max_len=5)            # ③提取尾部焦点
    relation_spans = _extract_relation_phrase(q, max_len=4)         # ④提取关系短语
    content = _content_tokens(q)                                     # ⑤提取内容词
    pieces = _unique_keep_order(quoted + tail_spans + relation_spans + numbers + content[:max_terms])  # 按优先级组合
    final_terms = []                                                 # 初始化最终词项
    token_budget = 0                                                 # 初始化token预算
    for p in pieces:                                                 # 遍历每个片段
        toks = _tokenize(p)                                          # 对片段分词
        if not toks:                                                 # 空则跳过
            continue
        if token_budget + len(toks) > max_terms:                     # 超出预算则截断
            break
        final_terms.append(" ".join(toks))                           # 加入最终词项
        token_budget += len(toks)                                    # 更新预算
    return " ".join(final_terms).strip() or simplify_query(q)        # 返回结果或简化查询

def _collect_prf_terms(
    docs: List[Doc],
    base_query: str,
    prf_docs: int = 5,
    prf_terms: int = 8,
    min_df: int = 2,
    max_doc_chars: int = 300,
) -> List[str]:
    docs = (docs or [])[: max(1, prf_docs)]
    if not docs:
        return []

    base_terms = set(_tokenize(base_query))
    df = Counter()
    tf = Counter()

    for d in docs:
        title = str(d.title or "")
        body = str(d.text or "")

        if max_doc_chars > 0:
            body = body[:max_doc_chars]

        title_toks = [t for t in _tokenize(title) if t not in _STOP]
        body_toks = [t for t in _tokenize(body) if t not in _STOP]

        weighted = title_toks * 3 + body_toks

        uniq_doc_terms = set()
        for t in weighted:
            if t in base_terms or len(t) < 3:
                continue
            tf[t] += 1
            uniq_doc_terms.add(t)

        for t in uniq_doc_terms:
            df[t] += 1

    scored = []
    for t in tf:
        if df[t] < min_df:
            continue
        score = 10.0 * df[t] + 1.0 * tf[t]
        scored.append((t, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [t for t, _ in scored[: max(1, prf_terms)]]


def _build_rewrite_from_dense_docs(                                  # 定义基于Dense文档构建重写查询函数
    query: str,                                                     # 原始查询
    dense_docs: List[Doc],                                          # Dense检索结果文档
    prf_docs: int,                                                  # PRF文档数
    prf_terms: int,                                                 # PRF扩展词数
    min_df: int,                                                    # 最小文档频率
    max_doc_chars: int,                                             # 文档最大字符数
    max_terms: int = 14,                                            # 最终查询最大词数
) -> str:                                                            # 返回重写后的查询字符串
    base_q = rewrite_query_keyword_v2(query, max_terms=max_terms)   # 先重写基础查询
    exp_terms = _collect_prf_terms(                                  # 收集PRF扩展词
        docs=dense_docs,
        base_query=base_q,
        prf_docs=prf_docs,
        prf_terms=prf_terms,
        min_df=min_df,
        max_doc_chars=max_doc_chars,
    )
    if not exp_terms:                                                # 无扩展词则返回基础查询
        return base_q
    return " ".join(_unique_keep_order([base_q] + exp_terms)).strip()  # 基础查询+扩展词去重


def rrf_fuse(
    rank_lists: List[List[str]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Weighted Reciprocal Rank Fusion.

    score(d) = sum_i weight_i / (k + rank_i(d) + 1)

    如果 weights=None，则退化为普通等权 RRF。
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)

    if len(weights) != len(rank_lists):
        raise ValueError(
            f"weights length must match rank_lists length: "
            f"{len(weights)} vs {len(rank_lists)}"
        )

    score: Dict[str, float] = {}

    for lst, w in zip(rank_lists, weights):
        w = float(w)
        if w <= 0:
            continue

        for r, doc_id in enumerate(lst):
            score[doc_id] = score.get(doc_id, 0.0) + w / (k + r + 1)

    return score                                                 # 返回融合分数


def _doc_key(d: Doc, mode: str = "doc_id") -> str:
    mode = (mode or "doc_id").lower().strip()

    if mode == "doc_id":
        return str(d.doc_id or "")

    if mode == "title_text":
        return (d.full_text or "")[:256]

    raise ValueError(f"Unsupported dedup_key_mode={mode!r}")


def _minmax_norm(score_map: Dict[str, float]) -> Dict[str, float]:   # 定义Min-Max归一化函数
    if not score_map:                                                # 空字典直接返回
        return {}
    vals = np.asarray(list(score_map.values()), dtype=float)         # 转为numpy数组
    vmin = float(vals.min())                                         # 最小值
    vmax = float(vals.max())                                         # 最大值
    if vmax <= vmin + 1e-12:                                         # 几乎无差异
        return {k: 0.0 for k in score_map.keys()}                    # 全部设为0
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in score_map.items()}  # Min-Max归一化


def _rank_norm(keys_in_order: List[str], score_map: Dict[str, float]) -> Dict[str, float]:  # 定义排名归一化函数
    uniq = []                                                        # 初始化唯一键列表
    seen = set()                                                     # 已见集合
    for k in keys_in_order:                                          # 按输入顺序遍历
        if k in score_map and k not in seen:                         # 在分数映射中且未处理
            uniq.append(k)                                           # 加入唯一列表
            seen.add(k)                                              # 标记已见
    n = len(uniq)                                                    # 唯一键数量
    if n <= 1:                                                       # 0或1个键
        return {k: 0.0 for k in uniq}                                # 全部设为0
    out: Dict[str, float] = {}                                       # 初始化输出
    for r, k in enumerate(uniq):                                     # 按顺序遍历
        out[k] = 1.0 - (r / (n - 1))                                 # 第1名=1.0，最后=0.0
    return out                                                       # 返回归一化排名


class HybridDenseBM25Retriever:                                      # 定义混合Dense+BM25检索器类
    def __init__(                                                    # 构造函数
        self,
        dense_retriever,                                             # Dense检索器（需实现retrieve_k）
        bm25_retriever,                                              # BM25检索器（需实现retrieve）
        dense_top_n: int = 50,                                       # Dense检索返回数量
        bm25_top_n: int = 50,                                        # BM25检索返回数量
        out_top_n: int = 50,                                         # 最终输出数量
        merge_mode: str = "rrf",                                     # 融合模式
        dense_keep_n: int = 10,                                      # dense_cap模式保留Dense数
        rrf_k: int = 60,                                             # RRF平滑参数
        dense_weight: float = 1.0,
        bm25_weight: float = 0.2,
        simplify_bm25: bool = True,                                  # 是否简化BM25查询
        dedup_key_mode: str = "title_text",                          # 去重键模式
        bm25_keep_n: int = 3,                                        # dense_cap模式保留BM25数
        score_fuse_norm: str = "minmax",                             # score_fuse归一化方式
        score_fuse_w_dense: float = 0.8,                             # Dense分数权重
        score_fuse_w_bm25: float = 0.2,                              # BM25分数权重
        rewrite_mode: str = "dense_prf",                             # 查询重写模式
        rewrite_prf_docs: int = 5,                                   # 重写PRF文档数
        rewrite_prf_terms: int = 8,                                  # 重写PRF扩展词数
        rewrite_doc_max_chars: int = 300,                            # 重写文档最大字符数
        rewrite_min_df: int = 2,                                     # 重写最小文档频率
        llm_rewriter=None,                                           # LLM重写器实例
        reranker_model: Optional[str] = None,                        # 重排序模型路径
        rerank_k: int = 30,                                          # 重排序前K个文档
        rerank_batch_size: int = 8,                                  # 重排序批次大小
        rerank_max_doc_chars: int = 256,                             # 重排序文档最大字符数
        profile: bool = False,                                       # 是否打印性能分析
    ):
        self.dense = dense_retriever                                  # 保存Dense检索器
        self.bm25 = bm25_retriever                                    # 保存BM25检索器
        self.dense_top_n = int(dense_top_n)                           # 转为整数
        self.bm25_top_n = int(bm25_top_n)                             # 转为整数
        self.out_top_n = int(out_top_n)                               # 转为整数
        self.merge_mode = (merge_mode or "rrf").lower().strip()       # 规范化融合模式
        if self.merge_mode not in ("dense_first", "dense_cap", "interleave", "rrf", "protected_rrf","score_fuse"):  # 校验模式
            raise ValueError(f"merge_mode must be dense_first|dense_cap|interleave|rrf|protected_rrf|score_fuse, got: {merge_mode}")  # 非法模式报错

        self.dense_keep_n = int(max(0, dense_keep_n))                # 非负整数
        self.rrf_k = int(rrf_k)                                       # RRF参数
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)
        self.simplify_bm25 = bool(simplify_bm25)                      # 布尔转换
        self.dedup_key_mode = (dedup_key_mode or "title_text").lower().strip()  # 规范化
        self.bm25_keep_n = int(max(0, bm25_keep_n))                  # 非负整数

        self.score_fuse_norm = (score_fuse_norm or "minmax").lower().strip()  # 规范化
        if self.score_fuse_norm not in ("minmax", "rank"):            # 校验归一化方式
            raise ValueError("score_fuse_norm must be minmax|rank")   # 非法方式报错
        self.score_fuse_w_dense = float(score_fuse_w_dense)           # 转为浮点数
        self.score_fuse_w_bm25 = float(score_fuse_w_bm25)             # 转为浮点数

        self.rewrite_mode = (rewrite_mode or "dense_prf").lower().strip()  # 规范化重写模式
        if self.rewrite_mode not in ("none", "keyword", "dense_prf", "bm25_prf", "llm"):  # 校验
            raise ValueError("rewrite_mode must be none|keyword|dense_prf|bm25_prf|llm")  # 非法报错

        self.rewrite_prf_docs = int(max(1, rewrite_prf_docs))        # 至少1篇
        self.rewrite_prf_terms = int(max(1, rewrite_prf_terms))      # 至少1个词
        self.rewrite_doc_max_chars = int(max(0, rewrite_doc_max_chars))  # 非负
        self.rewrite_min_df = int(max(1, rewrite_min_df))            # 至少1

        self.llm_rewriter = llm_rewriter                              # 保存LLM重写器

        # 这里的 reranker 是"融合后的 union reranker"                     # 注释说明reranker作用阶段
        self.reranker_model = reranker_model                          # 保存模型路径
        self.rerank_k = int(max(0, rerank_k))                        # 非负
        self.rerank_batch_size = int(max(1, rerank_batch_size))      # 至少1
        self.rerank_max_doc_chars = int(max(64, rerank_max_doc_chars))  # 至少64
        self.rerank_device = getattr(self.bm25, "device", "cpu")     # 从BM25获取设备，默认CPU
        self._reranker = None                                         # 延迟加载标记

        self.profile = bool(profile)                                  # 布尔转换

        if hasattr(self.bm25, "top_n"):                               # 如果BM25有top_n属性
            try:
                self.bm25.top_n = max(int(getattr(self.bm25, "top_n")), self.bm25_top_n)  # 确保足够大
            except Exception:                                          # 忽略异常
                pass

    def _lazy_load_reranker(self):                                    # 定义延迟加载重排序器方法
        if not self.reranker_model:                                   # 无模型配置
            return None                                                 # 返回None
        if self._reranker is None:                                    # 尚未加载
            try:
                self._reranker = CrossEncoder(                         # 尝试加载（本地优先）
                    self.reranker_model,
                    device=self.rerank_device,
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except TypeError:                                          # 旧版本不支持local_files_only
                self._reranker = CrossEncoder(                         # 回退加载
                    self.reranker_model,
                    device=self.rerank_device,
                    trust_remote_code=False,
                )
        return self._reranker                                          # 返回重排序器实例

    def _postprocess_llm_rewrite(self, text: str, fallback_query: str) -> str:  # 定义LLM重写后处理方法
        s = (text or "").strip()                                      # 去空白
        s = re.sub(r"```.*?```", "", s, flags=re.S)                   # 去除代码块
        s = re.sub(r"^(rewritten query|query|search query)\s*[:：]\s*", "", s, flags=re.I)  # 去除前缀
        s = " ".join(s.split())                                       # 规范化空白
        if "\n" in s:                                                 # 有多行
            s = s.splitlines()[0].strip()                             # 只取第一行
        s = s.strip().strip('"').strip("'").strip()                   # 去除引号

        if not s:                                                     # 结果为空
            return fallback_query                                     # 回退到fallback

        toks = _tokenize(s)                                           # 分词
        if len(toks) < 2:                                             # 少于2个词
            return fallback_query                                     # 回退

        toks = toks[:12]                                              # 最多12个词
        return " ".join(toks)                                         # 返回处理结果

    def _rewrite_query_with_llm(self, query: str) -> str:             # 定义LLM查询重写方法
        if self.llm_rewriter is None:                                 # 无LLM重写器
            return rewrite_query_keyword_v2(query)                    # 回退到关键词重写

        base_q = rewrite_query_keyword_v2(query)                      # 先得到基础重写作为fallback
        prompt = (                                                    # 构建提示词
            "Rewrite the question into a short keyword-rich search query for retrieval.\n"  # 任务说明
            "Rules:\n"                                                # 规则开始
            "- Output ONLY the rewritten search query.\n"             # 只输出查询
            "- Keep named entities, titles, years, places, and numbers.\n"  # 保留实体
            "- Remove filler words and question words.\n"             # 去除填充词
            "- Use 3 to 12 words.\n"                                  # 长度限制
            "- Do not answer the question.\n\n"                       # 不要回答
            f"Question: {query}\n"                                    # 原始问题
            f"Fallback query: {base_q}\n"                             # fallback查询
            "Rewritten search query:"                                 # 输出提示
        )

        try:
            out = self.llm_rewriter.generate(prompt)                  # 调用LLM生成
        except Exception:                                              # 异常回退
            return base_q

        if isinstance(out, dict):                                     # 输出是字典
            out = out.get("text") or out.get("output") or out.get("answer") or out.get("generated_text") or str(out)  # 提取文本
        elif isinstance(out, (list, tuple)) and len(out) > 0:        # 输出是序列
            out = out[0]                                              # 取第一个
            if isinstance(out, dict):                                 # 第一个还是字典
                out = out.get("text") or out.get("output") or out.get("answer") or out.get("generated_text") or str(out)  # 提取
            else:
                out = str(out)                                        # 转字符串
        else:
            out = str(out)                                            # 其他类型转字符串

        return self._postprocess_llm_rewrite(out, fallback_query=base_q)  # 后处理

    def _build_bm25_query(self, query: str, dense_docs: Optional[List[Doc]] = None, bm25_query: Optional[str] = None) -> str:  # 构建BM25查询
        if bm25_query is not None and str(bm25_query).strip():        # 外部传入BM25查询
            return str(bm25_query).strip()                            # 直接使用

        q = query or ""                                               # 处理None

        if self.rewrite_mode == "keyword":                            # 关键词重写模式
            return rewrite_query_keyword_v2(q)                        # 返回关键词重写结果

        if self.rewrite_mode == "dense_prf":                          # Dense_PRF模式
            return _build_rewrite_from_dense_docs(                     # 基于Dense文档PRF扩展
                query=q,
                dense_docs=dense_docs or [],
                prf_docs=self.rewrite_prf_docs,
                prf_terms=self.rewrite_prf_terms,
                min_df=self.rewrite_min_df,
                max_doc_chars=self.rewrite_doc_max_chars,
            )

        if self.rewrite_mode == "bm25_prf":                           # BM25_PRF模式
            base_q = rewrite_query_keyword_v2(q)                      # 先基础重写
            docs_b0, _scores_b0 = self.bm25.retrieve(base_q)          # BM25初检索
            exp_terms = _collect_prf_terms(                            # 收集PRF扩展词
                docs=docs_b0,
                base_query=base_q,
                prf_docs=self.rewrite_prf_docs,
                prf_terms=self.rewrite_prf_terms,
                min_df=self.rewrite_min_df,
                max_doc_chars=self.rewrite_doc_max_chars,
            )
            return (base_q + " " + " ".join(exp_terms)).strip() if exp_terms else base_q  # 拼接或回退

        return simplify_query(q) if self.simplify_bm25 else q         # 默认简化或原样返回

    def _rerank_union(self, query: str, docs: List[Doc], scores: np.ndarray) -> Tuple[List[Doc], np.ndarray]:  # 融合后重排序
        reranker = self._lazy_load_reranker()                         # 获取重排序器
        if reranker is None or self.rerank_k <= 0 or not docs:        # 无需重排序
            return docs, scores                                       # 直接返回

        k = min(self.rerank_k, len(docs))                            # 实际重排序数量
        head_docs = docs[:k]                                          # 头部待重排序
        tail_docs = docs[k:]                                          # 尾部保持原序

        pairs = []                                                    # 初始化query-doc对
        for d in head_docs:                                           # 遍历头部文档
            txt = (d.full_text or "")[: self.rerank_max_doc_chars]        # 截取文本
            pairs.append((query, txt))                                # 构建pair

        rr_scores = reranker.predict(                                 # CrossEncoder打分
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False,
        )
        rr_scores = np.asarray(rr_scores, dtype=float).reshape(-1)   # 转为numpy数组

        order = np.argsort(-rr_scores)                                # 按分数降序排序索引
        reranked_head_docs = [head_docs[i] for i in order]           # 重排头部文档
        reranked_head_scores = rr_scores[order]                       # 对应分数

        if tail_docs:                                                 # 有尾部文档
            tail_scores = np.asarray(scores[k:], dtype=float)        # 尾部原分数
            out_docs = reranked_head_docs + tail_docs                 # 拼接文档
            out_scores = np.concatenate([reranked_head_scores, tail_scores], axis=0)  # 拼接分数
        else:                                                         # 无尾部
            out_docs = reranked_head_docs                             # 仅头部
            out_scores = reranked_head_scores                         # 仅头部分数

        return out_docs, out_scores                                   # 返回结果

    def _print_profile(                                               # 定义性能分析打印方法
        self,
        query: str,                                                   # 原始查询
        q_dense: str,                                                 # Dense查询
        q_bm25: str,                                                  # BM25查询
        t_total: float,                                               # 总耗时
        t_rewrite: float,                                             # 重写耗时
        t_dense: float,                                               # Dense耗时
        t_bm25: float,                                                # BM25耗时
        t_fusion: float,                                              # 融合耗时
        t_rerank: float,                                              # 重排序耗时
        dense_n: int,                                                 # Dense结果数
        bm25_n: int,                                                  # BM25结果数
        out_n: int,                                                   # 输出结果数
    ) -> None:                                                        # 无返回值
        if not self.profile:                                          # 未开启分析
            return                                                    # 直接返回
        print(                                                        # 打印分析信息
            "[HybridDenseBM25][profile] "                             # 前缀标签
            f"total={t_total:.3f}s "                                  # 总时间
            f"rewrite={t_rewrite:.3f}s "                              # 重写时间
            f"dense={t_dense:.3f}s "                                  # Dense时间
            f"bm25={t_bm25:.3f}s "                                    # BM25时间
            f"fusion={t_fusion:.3f}s "                                # 融合时间
            f"rerank={t_rerank:.3f}s "                                # 重排序时间
            f"dense_n={dense_n} "                                     # Dense数量
            f"bm25_n={bm25_n} "                                       # BM25数量
            f"out_n={out_n} "                                         # 输出数量
            f"mode={self.merge_mode} "                                # 融合模式
            f"rewrite_mode={self.rewrite_mode} "                      # 重写模式
            f"q={query[:80]!r} "                                      # 原始查询（截断）
            f"q_dense={q_dense[:120]!r} "                             # Dense查询（截断）
            f"q_bm25={q_bm25[:120]!r}",                               # BM25查询（截断）
            flush=True,                                               # 立即刷新
        )

    def retrieve(self, query: str, bm25_query: Optional[str] = None) -> Tuple[List[Doc], np.ndarray]:
        t_total0 = time.perf_counter()

        if not hasattr(self.dense, "retrieve_k"):
            raise RuntimeError("dense_retriever must implement retrieve_k(query, top_n).")

        # 1) Dense 永远用原始 query
        q_dense = query
        t_rewrite = 0.0

        t0 = time.perf_counter()
        docs_d, scores_d = self.dense.retrieve_k(q_dense, self.dense_top_n)
        t_dense = time.perf_counter() - t0
        scores_d = np.asarray(scores_d, dtype=float).reshape(-1)

        # 2) BM25 三路 query
        t0 = time.perf_counter()
        if self.rewrite_mode == "llm":
            # A: 原问题关键词版
            q_bm25_a = rewrite_query_keyword_v2(query)

            # B: 小模型 LLM rewrite
            q_bm25_b = self._rewrite_query_with_llm(query)

            # C: dense-PRF 扩展版
            q_bm25_c = _build_rewrite_from_dense_docs(
                query=query,
                dense_docs=docs_d,
                prf_docs=self.rewrite_prf_docs,
                prf_terms=self.rewrite_prf_terms,
                min_df=self.rewrite_min_df,
                max_doc_chars=self.rewrite_doc_max_chars,
            )

            bm25_queries: List[str] = []
            for qx in [q_bm25_a, q_bm25_b, q_bm25_c]:
                qx = (qx or "").strip()
                if qx and qx not in bm25_queries:
                    bm25_queries.append(qx)
        else:
            q_bm25 = self._build_bm25_query(query, dense_docs=docs_d, bm25_query=bm25_query)
            bm25_queries = [q_bm25]

        t_rewrite += time.perf_counter() - t0

        # 3) 对一个或多个 BM25 query 分别检索，再用 RRF 融合为统一 BM25 排名
        t0 = time.perf_counter()

        bm25_doc_map: Dict[str, Doc] = {}
        bm25_rank_lists: List[List[str]] = []

        for qx in bm25_queries:
            docs_tmp, scores_tmp = self.bm25.retrieve(qx)
            scores_tmp = np.asarray(scores_tmp, dtype=float).reshape(-1)

            if docs_tmp and len(docs_tmp) > self.bm25_top_n:
                docs_tmp = docs_tmp[: self.bm25_top_n]
                scores_tmp = scores_tmp[: self.bm25_top_n]

            rank_keys: List[str] = []
            seen_local = set()

            for d in docs_tmp or []:
                k = _doc_key(d, self.dedup_key_mode)
                if k not in bm25_doc_map:
                    bm25_doc_map[k] = d
                if k not in seen_local:
                    rank_keys.append(k)
                    seen_local.add(k)

            bm25_rank_lists.append(rank_keys)

        t_bm25 = time.perf_counter() - t0

        bm25_rrf_scores = rrf_fuse(bm25_rank_lists, k=self.rrf_k)
        bm25_ranked = sorted(bm25_rrf_scores.items(), key=lambda x: -x[1])[: self.bm25_top_n]

        docs_b = [bm25_doc_map[k] for k, _ in bm25_ranked]
        scores_b = np.asarray([s for _, s in bm25_ranked], dtype=float)

        q_bm25 = " || ".join(bm25_queries)

        # 4) dense + bm25 外层融合
        t_fusion0 = time.perf_counter()

        by_key: Dict[str, Doc] = {}
        dense_keys: List[str] = []
        bm25_keys: List[str] = []
        dense_score_by_key: Dict[str, float] = {}
        bm25_score_by_key: Dict[str, float] = {}

        for d, s in zip(docs_d or [], scores_d.tolist() if scores_d.size else []):
            k = _doc_key(d, self.dedup_key_mode)
            if k not in by_key:
                by_key[k] = d
            dense_keys.append(k)
            dense_score_by_key[k] = max(dense_score_by_key.get(k, -1e30), float(s))

        for d, s in zip(docs_b or [], scores_b.tolist() if scores_b.size else []):
            k = _doc_key(d, self.dedup_key_mode)
            if k not in by_key:
                by_key[k] = d
            bm25_keys.append(k)
            bm25_score_by_key[k] = max(bm25_score_by_key.get(k, -1e30), float(s))

        def _append_unique(dst: List[str], src: List[str], seen: set):
            for k in src:
                if k in by_key and k not in seen:
                    dst.append(k)
                    seen.add(k)
                    if len(dst) >= self.out_top_n:
                        break

        union_keys: List[str] = []
        seen_u = set()
        for k in dense_keys:
            if k in by_key and k not in seen_u:
                union_keys.append(k)
                seen_u.add(k)
        for k in bm25_keys:
            if k in by_key and k not in seen_u:
                union_keys.append(k)
                seen_u.add(k)

        out_docs: List[Doc]
        out_scores: np.ndarray

        if self.merge_mode == "dense_first":
            seen = set()
            ranked_keys: List[str] = []
            _append_unique(ranked_keys, dense_keys, seen)
            _append_unique(ranked_keys, bm25_keys, seen)
            out_docs = [by_key[k] for k in ranked_keys]
            out_scores = np.asarray(list(reversed(range(len(out_docs)))), dtype=float)

        elif self.merge_mode == "dense_cap":
            seen = set()
            ranked_keys: List[str] = []
            head_dense = dense_keys[: self.dense_keep_n] if self.dense_keep_n > 0 else []
            tail_dense = dense_keys[self.dense_keep_n:] if self.dense_keep_n > 0 else dense_keys
            head_bm25 = bm25_keys[: self.bm25_keep_n] if self.bm25_keep_n > 0 else []
            tail_bm25 = bm25_keys[self.bm25_keep_n:] if self.bm25_keep_n > 0 else bm25_keys
            _append_unique(ranked_keys, head_dense, seen)
            _append_unique(ranked_keys, head_bm25, seen)
            _append_unique(ranked_keys, tail_dense, seen)
            _append_unique(ranked_keys, tail_bm25, seen)
            ranked_keys = ranked_keys[: self.out_top_n]
            out_docs = [by_key[k] for k in ranked_keys]
            out_scores = np.asarray(list(reversed(range(len(out_docs)))), dtype=float)

        elif self.merge_mode == "interleave":
            seen = set()
            ranked_keys: List[str] = []
            i = 0
            while len(ranked_keys) < self.out_top_n and (i < len(dense_keys) or i < len(bm25_keys)):
                if i < len(dense_keys):
                    k = dense_keys[i]
                    if k in by_key and k not in seen:
                        ranked_keys.append(k)
                        seen.add(k)
                        if len(ranked_keys) >= self.out_top_n:
                            break
                if i < len(bm25_keys):
                    k = bm25_keys[i]
                    if k in by_key and k not in seen:
                        ranked_keys.append(k)
                        seen.add(k)
                        if len(ranked_keys) >= self.out_top_n:
                            break
                i += 1
            out_docs = [by_key[k] for k in ranked_keys]
            out_scores = np.asarray(list(reversed(range(len(out_docs)))), dtype=float)

        elif self.merge_mode == "rrf":
            fused = rrf_fuse(
                [dense_keys, bm25_keys],
                k=self.rrf_k,
                weights=[self.dense_weight, self.bm25_weight],
            )
            ranked = sorted(fused.items(), key=lambda x: -x[1])[: self.out_top_n]
            out_docs = [by_key[k] for k, _ in ranked if k in by_key]
            out_scores = np.asarray([s for _, s in ranked[: len(out_docs)]], dtype=float)

        elif self.merge_mode == "protected_rrf":
            # 1) 先保留 dense 前 dense_keep_n 个作为语义主干
            ranked_keys: List[str] = []
            seen = set()

            head_dense = dense_keys[: self.dense_keep_n]
            head_bm25 = bm25_keys[: self.bm25_keep_n]

            _append_unique(ranked_keys, head_dense, seen)

            # 2) 对 dense 与 BM25 做 weighted RRF
            fused = rrf_fuse(
                [dense_keys, bm25_keys],
                k=self.rrf_k,
                weights=[self.dense_weight, self.bm25_weight],
            )
            fused_ranked = sorted(fused.items(), key=lambda x: -x[1])

            bm25_set = set(bm25_keys)

            # 3) rescue 只从 BM25 支持过的候选中选，避免退化成 dense tail
            rescue_keys: List[str] = []
            for k, _score in fused_ranked:
                if k in seen:
                    continue
                if k not in bm25_set:
                    continue
                rescue_keys.append(k)
                if len(rescue_keys) >= self.bm25_keep_n:
                    break

            _append_unique(ranked_keys, rescue_keys, seen)

            # 4) 如果 rescue 不够，再用 BM25 head 补齐
            if len(ranked_keys) < self.dense_keep_n + self.bm25_keep_n:
                _append_unique(ranked_keys, head_bm25, seen)

            # 5) 后续位置用 dense tail 和 fused tail 补齐
            _append_unique(ranked_keys, dense_keys[self.dense_keep_n :], seen)
            _append_unique(ranked_keys, [k for k, _ in fused_ranked], seen)
            _append_unique(ranked_keys, bm25_keys[self.bm25_keep_n :], seen)

            ranked_keys = ranked_keys[: self.out_top_n]
            out_docs = [by_key[k] for k in ranked_keys if k in by_key]

            out_scores = np.asarray(
                [fused.get(k, 0.0) for k in ranked_keys[: len(out_docs)]],
                dtype=float,
            )

        else:
            if self.score_fuse_norm == "rank":
                nd = _rank_norm(dense_keys, dense_score_by_key)
                nb = _rank_norm(bm25_keys, bm25_score_by_key)
            else:
                nd = _minmax_norm(dense_score_by_key)
                nb = _minmax_norm(bm25_score_by_key)

            wd = self.score_fuse_w_dense
            wb = self.score_fuse_w_bm25
            dense_rank = {k: i for i, k in enumerate(dense_keys)}
            bm25_rank = {k: i for i, k in enumerate(bm25_keys)}

            fused_items: List[Tuple[str, float]] = []
            for k in union_keys:
                sd = nd.get(k, 0.0)
                sb = nb.get(k, 0.0)
                fused_items.append((k, float(wd * sd + wb * sb)))

            fused_items.sort(
                key=lambda x: (-x[1], dense_rank.get(x[0], 10**9), bm25_rank.get(x[0], 10**9))
            )
            fused_items = fused_items[: self.out_top_n]
            out_docs = [by_key[k] for k, _ in fused_items]
            out_scores = np.asarray([s for _, s in fused_items], dtype=float)

        t_fusion = time.perf_counter() - t_fusion0

        # 5) 融合后统一 rerank（如果配置里没开，会自动跳过）
        t0 = time.perf_counter()
        out_docs, out_scores = self._rerank_union(query, out_docs, out_scores)
        t_rerank = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total0
        self._print_profile(
            query=query,
            q_dense=q_dense,
            q_bm25=q_bm25,
            t_total=t_total,
            t_rewrite=t_rewrite,
            t_dense=t_dense,
            t_bm25=t_bm25,
            t_fusion=t_fusion,
            t_rerank=t_rerank,
            dense_n=len(docs_d or []),
            bm25_n=len(docs_b or []),
            out_n=len(out_docs),
        )
        return out_docs, out_scores